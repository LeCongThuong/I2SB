# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from IQA_pytorch import LPIPSvgg
from guided_diffusion.script_util import create_model


from . import util
from .ckpt_util import (
    I2SB_IMG256_UNCOND_PKL,
    I2SB_IMG256_UNCOND_CKPT,
    I2SB_IMG256_COND_PKL,
    I2SB_IMG256_COND_CKPT,
)

from ipdb import set_trace as debug

class Image256Net(torch.nn.Module):
    def __init__(self, log, noise_levels, use_fp16=False, cond=False, pretrained_adm=False, ckpt_dir="data/"):
        super(Image256Net, self).__init__()

        # initialize model
        ckpt_pkl = os.path.join(ckpt_dir, I2SB_IMG256_COND_PKL if cond else I2SB_IMG256_UNCOND_PKL)
        with open(ckpt_pkl, "rb") as f:
            kwargs = pickle.load(f)
        kwargs["use_fp16"] = use_fp16
        kwargs["image_size"] = 512
        kwargs["in_channels"] = 2  
        kwargs["num_channels"] = 128
          
        self.diffusion_model = create_model(**kwargs)
        log.info(f"[Net] Initialized network from {ckpt_pkl=}! Size={util.count_parameters(self.diffusion_model)}!")

        # load (modified) adm ckpt
        if pretrained_adm:
            ckpt_pt = os.path.join(ckpt_dir, I2SB_IMG256_COND_CKPT if cond else I2SB_IMG256_UNCOND_CKPT)
            out = torch.load(ckpt_pt, map_location="cpu")
            self.diffusion_model.load_state_dict(out)
            log.info(f"[Net] Loaded pretrained adm {ckpt_pt=}!")

        self.diffusion_model.eval()
        self.cond = cond
        self.noise_levels = noise_levels

    def forward(self, x, steps, cond=None):

        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        x = torch.cat([x, cond], dim=1) if self.cond else x
        return self.diffusion_model(x, t)


class AuxLoss(nn.Module):
    def __init__(self, feat_coeff=1.0, pixel_coeff=1.0):
        super(AuxLoss, self).__init__()
        self.feat_loss_fn = LPIPSvgg()
        self.feat_coeff = feat_coeff
        self.pixel_coeff = pixel_coeff

    def forward(self, output, target):
        output = (output + 1) / 2.0
        target = (target + 1) / 2.0
        pixel_loss = F.mse_loss(output, target)
        output= output.repeat_interleave(3, dim=1)
        target = target.repeat_interleave(3, dim=1)
        feat_loss = self.feat_loss_fn(output, target)
        return self.pixel_coeff * pixel_loss + self.feat_coeff * feat_loss