import torch
import torch.nn as nn

from src.dinov2.models.vision_transformer import vit_base

class DinoV2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_base(patch_size=14, block_chunks=0, init_values=1.0) 
        self.model.load_state_dict(torch.load('/home/chr/Sketch_LVM_DINO/dinov2_vitb14_pretrain.pth'))

    def forward(self, x, prompt=None):
        return self.model(x, prompt=prompt.expand(x.shape[0], -1, -1))

    def parameters(self):
        return self.model.parameters()
