# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import torch

from ..builder import NECKS


@NECKS.register_module()
class FPN_POST(BaseModule):
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
                 act_cfg=dict(type='Swish'),
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN_POST, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.act_cfg=dict(type='Swish')

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
        self.lateral1_convs = nn.ModuleList()
        self.lateral2_convs = nn.ModuleList()
        self.lateral3_convs = nn.ModuleList()
        self.lateral4_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.upl_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up2_convs = nn.ModuleList()
        self.up3_convs = nn.ModuleList()
        self.up4_convs = nn.ModuleList()
        self.up5_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        self.down2_convs = nn.ModuleList()
        self.down3_convs = nn.ModuleList()


        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            l1_conv = ConvModule(
                out_channels*4,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            l2_conv = ConvModule(
                out_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            l3_conv = ConvModule(
                out_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            l4_conv = ConvModule(
                out_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            up_conv = ConvModule(
                out_channels,
                out_channels*3,
                4,
                stride=2,
                padding=1,
                conv_cfg=dict(type='ConvTranspose2d'),
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            up2_conv = ConvModule(
                out_channels,
                out_channels,
                4,
                stride=2,
                padding=1,
                conv_cfg=dict(type='ConvTranspose2d'),
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            up3_conv = ConvModule(
                in_channels[i],
                out_channels,
                4,
                stride=2,
                padding=1,
                conv_cfg=dict(type='ConvTranspose2d'),
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            down_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=None,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            # self.lateral3_convs.append(l3_conv)
            if i > 0:
                # self.lateral1_convs.append(l1_conv)
                # self.lateral2_convs.append(l2_conv)
                # self.up_convs.append(up_conv)
                self.up2_convs.append(up2_conv)
                self.up3_convs.append(up3_conv)
                #self.down2_convs.append(down_conv)
                #self.down3_convs.append(down_conv)
            # self.lateral2_convs.append(l2_conv)
            if i == self.start_level:
                self.down_convs.append(down_conv)
                self.lateral4_convs.append(l2_conv)
                # self.upl_convs.append(up_conv)
        for i in range(self.num_outs):
            self.fpn_convs.append(fpn_conv)
        #     self.lateral1_convs.append(l1_conv)
        #     self.lateral4_convs.append(l4_conv)
        # for i in range(26):
        #     self.lateral3_convs.append(l3_conv)
        for i in range(6):
            # self.up2_convs.append(up2_conv)
            self.up4_convs.append(up2_conv)
            # self.up5_convs.append(up2_conv)
        #     self.down2_convs.append(down_conv)
            self.down3_convs.append(down_conv)
        #     self.lateral2_convs.append(l2_conv)
        #     self.lateral2_convs.append(l2_conv)
            self.lateral3_convs.append(l3_conv)
            self.lateral3_convs.append(l3_conv)
            # self.down2_convs.append(down_conv)
            # self.lateral3_convs.append(l3_conv)
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
        temp=[
            inputs[i + self.start_level]
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # build laterals
        laterals =[
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # laterals =[
        #     conv(laterals[i])
        #     for i, conv in enumerate(self.up2_convs)
        # ] + laterals
        # temps=laterals

        
        # c43=self.up2_convs[0](laterals[-1])+self.lateral3_convs[0](laterals[2])
        # c42=self.up2_convs[1](c43)+self.lateral3_convs[1](laterals[1])
        # c41=self.up2_convs[2](c42)+self.lateral3_convs[2](laterals[0])
        # c32=self.up2_convs[3](laterals[-2])+self.lateral3_convs[3](laterals[1])
        # c31=self.up2_convs[4](c32)+self.lateral3_convs[4](laterals[0])
        # c21=self.up2_convs[5](laterals[-3])+self.lateral3_convs[5](laterals[0])
        # c12=self.down3_convs[0](laterals[0])+self.lateral3_convs[6](laterals[1])
        # c13=self.down3_convs[1](c12)+self.lateral3_convs[7](laterals[2])
        # c14=self.down3_convs[2](c13)+self.lateral3_convs[8](laterals[3])
        # c23=self.down3_convs[3](laterals[1])+self.lateral3_convs[9](laterals[2])
        # c24=self.down3_convs[4](c23)+self.lateral3_convs[10](laterals[3])
        # c34=self.down3_convs[5](laterals[2])+self.lateral3_convs[11](laterals[3])
        # f1=self.lateral4_convs[0](laterals[0])+self.lateral1_convs[0](torch.cat((self.lateral3_convs[12](laterals[0]),self.lateral2_convs[0](c41),self.lateral2_convs[1](c31),self.lateral2_convs[2](c21)),1))
        # f2=self.lateral4_convs[1](laterals[1])+self.lateral1_convs[1](torch.cat((self.lateral3_convs[13](laterals[1]),self.lateral2_convs[3](c42),self.lateral2_convs[4](c32),self.lateral2_convs[5](c12)),1))
        # f3=self.lateral4_convs[2](laterals[2])+self.lateral1_convs[2](torch.cat((self.lateral3_convs[14](laterals[2]),self.lateral2_convs[6](c43),self.lateral2_convs[7](c23),self.lateral2_convs[8](c13)),1))
        # f4=self.lateral4_convs[3](laterals[3])+self.lateral1_convs[3](torch.cat((self.lateral3_convs[15](laterals[3]),self.lateral2_convs[9](c14),self.lateral2_convs[10](c24),self.lateral2_convs[11](c34)),1))
        # d12=self.down2_convs[0](f1)+self.lateral3_convs[16](laterals[1])
        # d13=self.down2_convs[1](d12)+self.lateral3_convs[17](laterals[2])
        # d14=self.down2_convs[2](d13)+self.lateral3_convs[18](laterals[3])
        # d23=self.down2_convs[3](f2)+self.lateral3_convs[19](laterals[2])
        # d24=self.down2_convs[4](d23)+self.lateral3_convs[20](laterals[3])
        # d34=self.down2_convs[5](f3)+self.lateral3_convs[21](laterals[3])
        # f5=self.lateral4_convs[4](f4)+self.lateral1_convs[4](torch.cat((self.lateral3_convs[22](f4),self.lateral3_convs[23](d14),self.lateral3_convs[24](d24),self.lateral3_convs[25](d34)),1))
        # laterals=[f1,f2,f3,f4]
        
        # laterals=[f1,f2,f3,f5,f5]
        # laterals[-1] = self.down_convs[0](laterals[-1])
        # laterals =laterals + [
        #     conv(laterals[-1])
        #     for i, conv in enumerate(self.down_convs)
        # ]

        # for i, lateral_conv in enumerate(self.lateral3_convs):
        #     laterals[i]=laterals[i]+F.relu(lateral_conv(F.softmax(laterals[i])))*laterals[i]
        # build top-down path
        c43=self.up4_convs[0](laterals[-1])+self.lateral3_convs[0](laterals[2])
        c42=self.up4_convs[1](c43)+self.lateral3_convs[1](laterals[1])
        c41=self.up4_convs[2](c42)+self.lateral3_convs[2](laterals[0])
        c32=self.up4_convs[3](laterals[-2])+self.lateral3_convs[3](laterals[1])
        c31=self.up4_convs[4](c32)+self.lateral3_convs[4](laterals[0])
        c21=self.up4_convs[5](laterals[-3])+self.lateral3_convs[5](laterals[0])
        c12=self.down3_convs[0](laterals[0])+self.lateral3_convs[6](laterals[1])
        c13=self.down3_convs[1](c12)+self.lateral3_convs[7](laterals[2])
        c14=self.down3_convs[2](c13)+self.lateral3_convs[8](laterals[3])
        c23=self.down3_convs[3](laterals[1])+self.lateral3_convs[9](laterals[2])
        c24=self.down3_convs[4](c23)+self.lateral3_convs[10](laterals[3])
        c34=self.down3_convs[5](laterals[2])+self.lateral3_convs[11](laterals[3])
        f1=c41+c31+c21+laterals[0]
        f2=c42+c32+c12+laterals[1]
        f3=c43+c23+c13+laterals[2]
        f4=c14+c24+c34+laterals[3]
        laterals=[f1,f2,f3,f4]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            # x_softmax_weights = F.softmax(laterals[i - 1])*laterals[i - 1]
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + self.up2_convs[used_backbone_levels-1-i](laterals[i]) + self.up3_convs[i-1](temp[i])
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + self.up2_convs[used_backbone_levels-1-i](laterals[i]) + self.up3_convs[i-1](temp[i])
        # f2=self.up5_convs[0](laterals[1])
        # f32=self.up5_convs[1](laterals[2])
        # f3=self.up5_convs[2](f32)
        # f43=self.up5_convs[3](laterals[3])
        # f42=self.up5_convs[4](f43)
        # f4=self.up5_convs[5](f42)
        # laterals=[f1,f2,f3,f4]
        c43=self.up4_convs[0](laterals[-1])+self.lateral3_convs[0](laterals[2])
        c42=self.up4_convs[1](c43)+self.lateral3_convs[1](laterals[1])
        c41=self.up4_convs[2](c42)+self.lateral3_convs[2](laterals[0])
        c32=self.up4_convs[3](laterals[-2])+self.lateral3_convs[3](laterals[1])
        c31=self.up4_convs[4](c32)+self.lateral3_convs[4](laterals[0])
        c21=self.up4_convs[5](laterals[-3])+self.lateral3_convs[5](laterals[0])
        c12=self.down3_convs[0](laterals[0])+self.lateral3_convs[6](laterals[1])
        c13=self.down3_convs[1](c12)+self.lateral3_convs[7](laterals[2])
        c14=self.down3_convs[2](c13)+self.lateral3_convs[8](laterals[3])
        c23=self.down3_convs[3](laterals[1])+self.lateral3_convs[9](laterals[2])
        c24=self.down3_convs[4](c23)+self.lateral3_convs[10](laterals[3])
        c34=self.down3_convs[5](laterals[2])+self.lateral3_convs[11](laterals[3])
        f1=c41+c31+c21+laterals[0]
        f2=c42+c32+c12+laterals[1]
        f3=c43+c23+c13+laterals[2]
        f4=c14+c24+c34+laterals[3]
        laterals=[f1,f2,f3,f4]
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
        #     #  it cannot co-exist with `size` in `F.interpolate`.
        #     # x_softmax_weights = F.softmax(laterals[i - 1])*laterals[i - 1]
        #     if 'scale_factor' in self.upsample_cfg:
        #         laterals[i - 1] = laterals[i - 1] + self.up2_convs[used_backbone_levels-1-i](laterals[i]) + self.up3_convs[i-1](temp[i])
        #     else:
        #         prev_shape = laterals[i - 1].shape[2:]
        #         laterals[i - 1] = laterals[i - 1] + self.up2_convs[used_backbone_levels-1-i](laterals[i]) + self.up3_convs[i-1](temp[i])
        laterals =laterals + [self.down_convs[0](self.lateral4_convs[0](laterals[-1]))]
        # laterals = [
        #     lateral_conv(laterals[i])
        #     for i, lateral_conv in enumerate(self.lateral3_convs)
        # ]
        # build outputs
        # part 1: from original levels

        # laterals=[lateral_conv(laterals[i])
        #     for i, lateral_conv in enumerate(self.down_convs)
        # ]
        # laterals =[
        #     F.relu(conv(laterals[i]))
        #     for i, conv in enumerate(self.upl_convs)
        # ] + laterals
        # laterals =laterals + [
        #     F.relu(conv(laterals[-1]))
        #     for i, conv in enumerate(self.down_convs)
        # ]
        
        used_backbone_levels = len(laterals)
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
