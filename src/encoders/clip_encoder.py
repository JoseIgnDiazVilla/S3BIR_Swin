from src.clip import clip
import torch.nn as nn

class ClipEncoder(nn.Module):
    def __init__(self, model_name='ViT-B/16', device='cuda'):
        super().__init__()
        self.model, _ = clip.load(model_name, device=device)

    def forward(self, x, prompt):
        return self.model.encode_image(
            x,
            prompt
        )

    def parameters(self):
        return self.model.parameters()
