# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, xavier_init
from mmcv.runner import BaseModule, ModuleList

from ..builder import NECKS, build_backbone
from .fpn import FPN
from .fpn_post import FPN_POST
from mmcv.cnn import ConvModule


class ASPP(BaseModule):
    """ASPP (Atrous Spatial Pyramid Pooling)

    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations=(1, 3, 6, 1),
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super().__init__(init_cfg)
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            temp=F.relu(self.aspp[aspp_idx](inp))
            out.append(temp)
            # out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


@NECKS.register_module()
class RFP_POSTFUSION(FPN):
    """RFP (Recursive Feature Pyramid)

    This is an implementation of RFP in `DetectoRS
    <https://arxiv.org/pdf/2006.02334.pdf>`_. Different from standard FPN, the
    input of RFP should be multi level features along with origin input image
    of backbone.

    Args:
        rfp_steps (int): Number of unrolled steps of RFP.
        rfp_backbone (dict): Configuration of the backbone for RFP.
        aspp_out_channels (int): Number of output channels of ASPP module.
        aspp_dilations (tuple[int]): Dilation rates of four branches.
            Default: (1, 3, 6, 1)
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 rfp_steps,
                 rfp_backbone,
                 aspp_out_channels,
                 aspp_dilations=(1, 3, 6, 1),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.rfp_steps = rfp_steps
        # Be careful! Pretrained weights cannot be loaded when use
        # nn.ModuleList
        self.rfp_modules = ModuleList()
        self.lateral=ModuleList()
        self.postlateral=ModuleList()
        self.up4_convs=ModuleList()
        self.down3_convs=ModuleList()
        self.lateral3_convs=ModuleList()
        for i in range(5):
            l_conv = ConvModule(
                256,
                256,
                1,
                conv_cfg=dict(type='ConvAWS'),
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='Swish'),
                inplace=False)
            up_conv = ConvModule(
                256,
                256,
                4,
                stride=2,
                padding=1,
                conv_cfg=dict(type='ConvTranspose2d'),
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='Swish'),
                inplace=False)
            down_conv = ConvModule(
                256,
                256,
                3,
                stride=2,
                padding=1,
                conv_cfg=dict(type='ConvAWS'),
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='Swish'),
                inplace=False)
            # self.lateral.append(l_conv)
        #     self.postlateral.append(l_conv)
        # for i in range(6):
        #     self.up4_convs.append(up_conv)
        #     self.down3_convs.append(down_conv)
        # for i in range(12):
        #     self.lateral3_convs.append(l_conv)
        
        for rfp_idx in range(1, rfp_steps):
            rfp_module = build_backbone(rfp_backbone)
            self.rfp_modules.append(rfp_module)
        self.rfp_aspp = ASPP(self.out_channels, aspp_out_channels,
                             aspp_dilations)
        self.rfp_weight = nn.Conv2d(
            self.out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def init_weights(self):
        # Avoid using super().init_weights(), which may alter the default
        # initialization of the modules in self.rfp_modules that have missing
        # keys in the pretrained checkpoint.
        for convs in [self.lateral_convs, self.fpn_convs]:
            for m in convs.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        for rfp_idx in range(self.rfp_steps - 1):
            self.rfp_modules[rfp_idx].init_weights()
        constant_init(self.rfp_weight, 0)

    def forward(self, inputs):
        inputs = list(inputs)
        assert len(inputs) == len(self.in_channels) + 1  # +1 for input image
        img = inputs.pop(0)
        # FPN forward
        x = super().forward(tuple(inputs))
        # print(x[0].size(),x[1].size(),x[2].size(),x[3].size(),x[4].size())
        for rfp_idx in range(self.rfp_steps - 1):
            rfp_feats = [x[0]] + list(
                self.rfp_aspp(x[i]) for i in range(1, len(x)))
            x_idx = self.rfp_modules[rfp_idx].rfp_forward(img, rfp_feats)
            # FPN forward
            x_idx = super().forward(x_idx)
            x_new = []
            for ft_idx in range(len(x_idx)):
                add_weight = torch.sigmoid(self.rfp_weight(x_idx[ft_idx]))
                x_new.append(add_weight * x_idx[ft_idx] +
                             (1 - add_weight) * x[ft_idx])
            x = x_new


        ####################################################
        # Post Fusion
        # laterals=x[:-1]
        # c43=self.up4_convs[0](laterals[-1])+self.lateral3_convs[0](laterals[2])
        # c42=self.up4_convs[1](c43)+self.lateral3_convs[1](laterals[1])
        # c41=self.up4_convs[2](c42)+self.lateral3_convs[2](laterals[0])
        # c32=self.up4_convs[3](laterals[-2])+self.lateral3_convs[3](laterals[1])
        # c31=self.up4_convs[4](c32)+self.lateral3_convs[4](laterals[0])
        # c21=self.up4_convs[5](laterals[-3])+self.lateral3_convs[5](laterals[0])
        # c12=self.down3_convs[0](laterals[0])+self.lateral3_convs[6](laterals[1])
        # c13=self.down3_convs[1](c12)+self.lateral3_convs[7](laterals[2])
        # c14=self.down3_convs[2](c13)+self.lateral3_convs[8](laterals[3])
        # c23=self.down3_convs[3](laterals[1])+self.lateral3_convs[9](laterals[2])
        # c24=self.down3_convs[4](c23)+self.lateral3_convs[10](laterals[3])
        # c34=self.down3_convs[5](laterals[2])+self.lateral3_convs[11](laterals[3])
        # f1=c41+c31+c21
        # f2=c42+c32+c12
        # f3=c43+c23+c13
        # f4=c14+c24+c34
        # x[0]=f1
        # x[1]=f2
        # x[2]=f3
        # x[3]=f4
        # for i in range(len(x)):
        #     x[i]=self.postlateral[i](x[i])

        ####################################################
        return x
