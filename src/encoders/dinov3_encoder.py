import torch
import torch.nn as nn

from src.dinov3.models.vision_transformer import vit_base

class DinoV3Encoder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.model = vit_base(n_storage_tokens=4, mask_k_bias=True, layerscale_init=12) 
        self.model.load_state_dict(torch.load('/home/chr/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', map_location=device))
        print("DINOv3 model loaded")
        print(self.model)

    def forward(self, x, prompt=None):
        return self.model(x, prompt=None)

    def parameters(self):
        return self.model.parameters()
