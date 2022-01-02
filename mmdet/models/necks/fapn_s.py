# Copyright (c) OpenMMLab. All rights reserved.
## FaPN
from __future__ import absolute_import, division, print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule
from mmcv.cnn import constant_init, xavier_init
from mmcv.runner import BaseModule, auto_fp16
from mmcv.ops.deform_conv import DeformConv2d

from ..builder import NECKS

# from __future__ import absolute_import, division, print_function
# from .dcn_v2 import DCN as dcn_v2
import math

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import PIL
from PIL import Image
import os
import numpy as np


# import _ext as _backend


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        # self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False,).to('cuda')
        # self.groupnorm=nn.GroupNorm(2,256).to('cuda')
        self.batchnorm=nn.BatchNorm2d(288).to('cuda')
        # self.instancenorm=nn.InstanceNorm2d(256).to('cuda')

        self.sigmoid = nn.Sigmoid()
        # self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False,).to('cuda')
        xavier_init(self.conv_atten,distribution='uniform')
        xavier_init(self.conv,distribution='uniform')

    def forward(self, x):
        # atten = self.sigmoid(self.bn(self.conv_atten(F.avg_pool2d(x, x.size()[2:])).to('cuda')))
        # atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])).to('cuda'))
        temp=self.batchnorm(self.conv_atten(F.avg_pool2d(x, x.size()[2:])).to('cuda'))
        atten = self.sigmoid(temp)
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=288, out_nc=144, norm=None):
        super(FeatureAlign_V2, self).__init__()
        self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="GB").to('cuda')
        # self.offset = nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.offset = nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False,).to('cuda')
        # self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
        #                         extra_offset_mask=True).to('cuda')
        self.dcpack_L2= DeformConv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, dilation=1, deform_groups=16,).to('cuda')
        self.relu = nn.ReLU(inplace=True)
        # self.groupnorm=nn.GroupNorm(2,256).to('cuda')
        self.batchnorm=nn.BatchNorm2d(288).to('cuda')
        self.fapn_weight = nn.Conv2d(
            out_nc,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        # self.instancenorm=nn.InstanceNorm2d(256).to('cuda')
        xavier_init(self.offset,distribution='uniform')
        xavier_init(self.dcpack_L2,distribution='uniform')
        xavier_init(self.fapn_weight,distribution='uniform')

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False).to('cuda')
        else:
            feat_up = feat_s.to('cuda')
        feat_arm = self.lateral_conv(feat_l).to('cuda')  # 0~1 * feats
        offset = self.batchnorm(self.offset(torch.cat([feat_arm, feat_up * 2], dim=1).to('cuda')))  # concat for offset by compute the dif
        # feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
        # print(feat_up.size(),self.dcpack_L2.weight.size())
        feat_align = self.relu(self.dcpack_L2(feat_up, offset))  # [feat, offset]
        add_weight = torch.sigmoid(self.fapn_weight(feat_arm))
        
        # return feat_align + feat_arm
        return add_weight * feat_arm + (1 - add_weight) * feat_align

@NECKS.register_module()
class FaPN_SF(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FaPN_SF, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.l_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fapn=FeatureAlign_V2(288,288)
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                288,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            l2_conv = ConvModule(
                in_channels[i],
                in_channels[i-1],
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                288,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.fpn_convs.append(fpn_conv)
            self.lateral_convs.append(l_conv)
            # if i > 0:
            #     self.l_convs.append(l2_conv)
            # else:
            #     self.l_convs.append(l_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # fsm=FeatureSelectionModule(256,256,)
        # laterals.append(fsm(inputs[self.start_level]))
        # fsm=FeatureSelectionModule(512,256,)
        # laterals.append(fsm(inputs[1+self.start_level]))
        # fsm=FeatureSelectionModule(1024,256,)
        # laterals.append(fsm(inputs[2+self.start_level]))
        # fsm=FeatureSelectionModule(2048,256,)
        # laterals.append(fsm(inputs[3+self.start_level]))
        # fapn1=FeatureAlign_V2(256,512)
        # fapn2=FeatureAlign_V2(512,512)
        # fapn3=FeatureAlign_V2(1024,1024)
        # laterals.append(fapn(inputs[self.start_level],inputs[self.start_level+1]))
        # fapn=FeatureAlign_V2(512,256)
        # laterals.append(fapn(inputs[self.start_level+1],inputs[self.start_level+2]))
        # fapn=FeatureAlign_V2(1024,256)
        # laterals.append(fapn(inputs[self.start_level+2],inputs[self.start_level+3]))
        # fapn=FeatureAlign_V2(2048,256)
        # laterals.append(fapn(inputs[self.start_level+3],inputs[self.start_level+4]))
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # laterals = [
        #     inputs[i + self.start_level]
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]
        # laterals[2] = fapn3(laterals[2],laterals[3])
        # laterals[1] = fapn2(laterals[1],laterals[2])
        # laterals[0] = fapn1(laterals[0],laterals[1])
        # laterals = [
        #     lateral_conv(laterals[i])
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]
        # # build laterals
        # laterals = [
        #     lateral_conv(inputs[i + self.start_level])
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = self.fapn(laterals[i-1],laterals[i])
                # laterals[i - 1] += F.interpolate(laterals[i],
                #                                  **self.upsample_cfg)
                # print(laterals[i-1])
            else:
                # prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = self.fapn(laterals[i-1],laterals[i])
                # laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape,
                #                                  **self.upsample_cfg)
                # print(laterals[i-1])

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
