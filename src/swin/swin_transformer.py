# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from .swin_backbone import SwinTransformer  as SwinViT


logger = logging.getLogger("swin")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class SwinBackbone(nn.Module):
    def __init__(self, args, embed_dim, patch_size, window_size):
        super().__init__()
        self.swin = SwinViT(
            in_chans=args.in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
            window_size=window_size,
            depths=[2,2,2,2],      
            num_heads=[3,6,12,24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.drop_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
        )
        self.cls_norm = nn.LayerNorm(embed_dim)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.in_channels = args.in_chans
        self.embed_dim = embed_dim

    def forward(self, x, masks=None, is_training=False):
        feats = self.swin(x)
        
        f = feats[-1]               # (B, C, H', W')
        B, C, H, W = f.shape

        cls_token = f.flatten(2).mean(dim=2)   # (B, C)
        cls_token = self.cls_norm(cls_token)

        patch_tokens = f.flatten(2).transpose(1,2)  # (B, H'*W', C)
        patch_tokens = patch_tokens.reshape(B*H*W, C)
        patch_tokens = self.patch_norm(patch_tokens)

        return {
            "x_norm_clstoken": cls_token,
            "x_norm_patchtokens": patch_tokens
        }
    

    def forward_features_list(self, x_list, masks_list, prompt_list):
        x = [self.prepare_tokens_with_masks(x, masks, prompt) for x, masks, prompt in zip(x_list, masks_list, prompt_list)]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None, prompt=None):
        # print("forward_features prompt: ", prompt)
        if isinstance(x, list):
            return self.forward_features_list(x, masks, prompt)

        x = self.prepare_tokens_with_masks(x, masks, prompt)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }
    


def swin_base(
    args,
    embed_dim: int = 96,
    patch_size: Union[int, Tuple[int, int]] = (4,4),
    window_size: Union[int, Tuple[int, int]] = (7,7),
) -> SwinBackbone:

    return SwinBackbone(args, embed_dim, patch_size, window_size)

