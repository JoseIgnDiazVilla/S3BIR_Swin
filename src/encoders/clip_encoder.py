from src.clip import clip
import torch.nn as nn

class ClipEncoder(nn.Module):
    def __init__(self, model_name='ViT-B/16'):
        super().__init__()
        self.model, _ = clip.load(model_name)
        self.out_dim = self.model.visual.output_dim

    def forward(self, x, prompt=None):
        return self.model.encode_image(x, prompt.expand(x.shape[0], -1, -1)) if prompt is not None else self.model.encode_image(x)

    def parameters(self):
        return self.model.visual.parameters()
