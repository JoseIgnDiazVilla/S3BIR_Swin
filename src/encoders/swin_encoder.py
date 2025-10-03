from src.swin import swin
import torch.nn as nn

class SwinEncoder(nn.Module):
    def __init__(self, model_name='Swin-T', device='cuda'):
        super().__init__()
        self.model, _ = swin.load(model_name, device=device)

    def forward(self, x, prompt):
        return self.model.encode_image(
            x,
            prompt
        )

    def parameters(self):
        return self.model.parameters()
