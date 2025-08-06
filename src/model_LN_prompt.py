import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision
import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts

from src.encoders.clip_encoder import ClipEncoder
from src.encoders.dinov2_encoder import DinoV2Encoder

def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts

        if self.opts.encoder == 'clip':
            self.encoder = ClipEncoder()
        elif self.opts.encoder == 'dinov2':
            self.encoder = DinoV2Encoder()
        else:
            raise ValueError(f"Unknown model_type {self.opts.encoder}")
        
        self.encoder.apply(freeze_all_but_bn)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn_triplet = nn.TripletMarginWithDistanceLoss(
             distance_function=self.distance_fn, margin=0.2)
        
        self.emb_cos_loss = nn.CosineEmbeddingLoss(margin=0.2)
        self.loss_kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.best_metric = -1e3
        self.val_step_outputs = []

    def configure_optimizers(self):
        if self.opts.model_type == 'one_encoder':
            model_params = list(self.encoder.parameters())
        else:
            model_params = list(self.encoder.parameters()) + list(self.clip_sk.parameters())

        optimizer = torch.optim.Adam([
            {'params': model_params, 'lr': self.opts.LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer
    
    def loss_fn_nx(self, z1, z2, temperature=0.5):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N, Z = z1.shape
        device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        diag = torch.eye(2*N, dtype=torch.bool, device=device)
        diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

        negatives = similarity_matrix[~diag].view(2*N, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= temperature

        labels = torch.zeros(2*N, device=device, dtype=torch.int64)

        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * N)
    
    def loss_clip(self, emb_sketch, emb_photo):
        norm_emb_sketch = F.normalize(emb_sketch, dim=1)
        norm_emb_photo = F.normalize(emb_photo, dim=1)

        similarity_matrix = norm_emb_sketch @ norm_emb_photo.T
        loss = F.cross_entropy(similarity_matrix, torch.arange(similarity_matrix.shape[0], device=self.device), reduction='none')
        return loss.mean()
    
    def loss_fn_nc(self, emb_sketch, emb_photo):
        sketch_soft = F.softmax(emb_sketch, dim=1)
        photo_soft = F.softmax(emb_photo, dim=1)

        loss = F.kl_div(sketch_soft.log(), photo_soft, reduction='batchmean') # -torch.sum(photo_soft * torch.log(sketch_soft))

        return loss
    
    def info_nce_loss(self, emb_sketch, emb_image, temperature=0.5):
        img_norm = F.normalize(emb_image)
        sketch_norm = F.normalize(emb_sketch)
        
        sim = img_norm @ sketch_norm.T
        mask = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        sim.masked_fill_(mask, -9e15) # los positivos == -9e15
        pos_mask = mask.roll(shifts=sim.shape[0] // 2, dims=0)

        # InfoNCE loss
        sim = sim / temperature
        nll = -sim[pos_mask] + torch.logsumexp(sim, dim=-1)
        nll = nll.mean()
        return nll
    

    def info_nce_loss_v2(self, emb_sketch, emb_image):
        ground_truth = torch.arange(emb_sketch.shape[0], device=emb_sketch.device)
        total_loss = (F.cross_entropy(emb_sketch, ground_truth) + F.cross_entropy(emb_image, ground_truth)) / 2
        return total_loss

    
    def loss_fn(self, sketch_emb, image_emb):
        n_loss_terms = 0
        total_loss = 0
        sketch_emb_out = F.softmax(sketch_emb / 0.5, dim=1)

        for iq, q in enumerate(image_emb):
            for v in range(len(sketch_emb_out)):
                if iq == v:
                    continue
                loss = torch.sum(-q * F.log_softmax(sketch_emb_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        return total_loss / n_loss_terms

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.encoder(data, prompt=self.img_prompt)
        else:
            feat = self.encoder(data, prompt=self.sk_prompt)
        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn_triplet(sk_feat, img_feat, neg_feat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        # loss = self.loss_fn(sk_feat, img_feat)
        loss = self.loss_fn_triplet(sk_feat, img_feat, neg_feat)
        #loss = self.loss_fn(img_feat, sk_feat)
        self.log('val_loss', loss)
        return sk_feat, img_feat, category
    
    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]

        with torch.no_grad():
            img_feat = self.forward(img_tensor, dtype='image')
            sk_feat = self.forward(sk_tensor, dtype='sketch')
            neg_feat = self.forward(neg_tensor, dtype='image')

            loss = self.loss_fn_triplet(sk_feat, img_feat, neg_feat)
            self.log('val_loss', loss)
            self.val_step_outputs.append((sk_feat, img_feat, category))
        
        return sk_feat, img_feat, category

    def on_validation_epoch_end(self):
        with torch.no_grad():
            Len = len(self.val_step_outputs)
            if Len == 0:
                return
            query_feat_all = torch.cat([self.val_step_outputs[i][0] for i in range(Len)])
            gallery_feat_all = torch.cat([self.val_step_outputs[i][1] for i in range(Len)])
            all_category = np.array(sum([list(self.val_step_outputs[i][2]) for i in range(Len)], []))

            # mAP category-level SBIR Metrics
            gallery = gallery_feat_all
            ap = torch.zeros(len(query_feat_all))
            for idx, sk_feat in enumerate(query_feat_all):
                category = all_category[idx]
                distance = -1 * self.distance_fn(sk_feat.unsqueeze(0), gallery)
                target = torch.zeros(len(gallery), dtype=torch.bool)
                target[np.where(all_category == category)] = True
                ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
            
            mAP = torch.mean(ap)
            self.log('mAP', mAP)
            if self.global_step > 0:
                self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
            print ('mAP: {}, Best mAP: {}'.format(mAP.item(), self.best_metric))
            self.val_step_outputs = []
            del query_feat_all, gallery_feat_all, all_category, gallery, ap, mAP
