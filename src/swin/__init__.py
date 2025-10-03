# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import swin_transformer as swin
from torch import nn
from types import SimpleNamespace


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "swin" in args.arch:
        vit_kwargs = dict(
            in_chans=args.in_channels,
            #embed_dim=embed_dim,
            patch_size=(4,4),
            window_size=(7,7),
            depths=[2,2,2,2],      
            num_heads=[3,6,12,24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
        )
        args_obj = SimpleNamespace(**vit_kwargs)
        teacher = swin.swin_base(args_obj) #vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = swin.swin_base(args_obj) #vits.__dict__[args.arch](
            #**vit_kwargs,
            #drop_path_rate=args.drop_path_rate,
            #drop_path_uniform=args.drop_path_uniform,
        #)
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
