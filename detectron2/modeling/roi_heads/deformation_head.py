# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

"""
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

class Deformation(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = 3
        fc_dim_regr= [256, 256, 2]
        fc_dim_cls = [256, 256, cfg.K * cfg.K * 2]
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_fc > 0

        self._output_size_regr = (input_shape.channels, input_shape.height, input_shape.width)
        self._output_size_cls = (input_shape.channels, input_shape.height, input_shape.width)

        self.fcs_regr = []
        self.fcs_cls  = []
        self.fc_shared = Linear(np.prod(self._output_size_regr), fc_dim_regr[0])
        self._output_size_regr = fc_dim_regr[0]
        self._output_size_cls  = fc_dim_cls[0]
        for k in range(num_fc - 1):
            fc_regr = Linear(np.prod(self._output_size_regr), fc_dim_regr[k+1])
            fc_cls = Linear(np.prod(self._output_size_cls), fc_dim_cls[k+1])
            self.add_module("fc_regr{}".format(k + 1), fc_regr)
            self.add_module("fc_cls{}".format(k + 1), fc_cls)
            self.fcs_regr.append(fc_regr)
            self.fcs_cls.append(fc_cls)
            self._output_size_regr = fc_dim_regr[k+1]
            self._output_size_cls  = fc_dim_cls[k+1]

        weight_init.c2_xavier_fill(self.fc_shared)
        for layer in self.fcs_regr:
            weight_init.c2_xavier_fill(layer)
        for layer in self.fcs_cls:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        if len(self.fcs_regr and self.fcs_cls):

            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            shared = F.relu(self.fc_shared(x))
            x = F.relu(self.fcs_regr[0](shared))
            x = self.fcs_regr[1](x)
            y = F.relu(self.fcs_cls[0](shared))
            y = self.fcs_cls[1](y)
        return x, y

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])
