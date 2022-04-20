# Modification 2022 AnChao
# Copyright 2018-2019 Open-MMLab.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import torch.nn as nn
import torch.nn.functional as F

from ..module.conv import ConvModule
from ..module.init_weights import xavier_init


class FPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        conv_cfg=None,
        norm_cfg=None,
        activation=None,
    ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.fp16_enabled = False

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
        self.lateral_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode="bilinear"
            )

        # build outputs
        outs = [
            # self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            laterals[i]
            for i in range(used_backbone_levels)
        ]
        return tuple(outs)

class DeconvLayer(nn.Module):

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 deconv_kernel: int,
                 deconv_stride: int = 2,
                 deconv_pad: int = 1,
                 deconv_out_pad: int = 0):
        super(DeconvLayer, self).__init__()
        self.dcn = nn.Conv2d(in_planes, out_planes,
                                kernel_size=3, stride=1,
                                padding=1, dilation=1, bias=False)
        self.dcn_bn = nn.BatchNorm2d(out_planes)
        self.up_sample = nn.ConvTranspose2d(in_channels=out_planes, out_channels=out_planes, kernel_size=deconv_kernel,
                                            stride=deconv_stride, padding=deconv_pad, output_padding=deconv_out_pad,
                                            bias=False)
        self._deconv_init()
        self.up_bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dcn(x)
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

class CenternetDeconv(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """

    def __init__(self,
                 deconv_channels = [512, 256, 128, 64],
                 deconv_kernel = [4, 4, 4],
                 modulate_deform: bool = True):
        super(CenternetDeconv, self).__init__()

        self.deconv1 = DeconvLayer(deconv_channels[0], deconv_channels[1], deconv_kernel=deconv_kernel[0])
        self.deconv2 = DeconvLayer(deconv_channels[1], deconv_channels[2], deconv_kernel=deconv_kernel[1])
        self.deconv3 = DeconvLayer(deconv_channels[2], deconv_channels[3], deconv_kernel=deconv_kernel[2])

    def forward(self, x, targets=None):
        x = self.deconv1(x[-1])
        x = self.deconv2(x)
        x = self.deconv3(x)
        return tuple([x])

# if __name__ == '__main__':
