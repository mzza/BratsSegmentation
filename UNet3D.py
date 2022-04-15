# the arch of unet3d

import pytorch_lightning as pl
import torch
from torch import nn

num_base_channel = 30
dropout_rate = 0.2


class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.encoder = self._help_build_encode_conv()
        self.downsample = self._help_build_downsample()
        self.upsample = self._help_build_upsample()
        self.decoder = self._help_build_decoder()
        self.seg_layer = self._help_build_seg_layer()
        self.seg_up=self._help_build_seg_up_layer()

    def _help_build_seg_up_layer(self):
        return nn.ModuleList([nn.Upsample(scale_factor=4, mode="trilinear"),
                              nn.Upsample(scale_factor=2, mode="trilinear")])

    def _help_build_seg_layer(self):
        return nn.ModuleList([nn.Conv3d(in_channels=60, out_channels=4, kernel_size=1, bias=False),
                            nn.Conv3d(in_channels=30, out_channels=4, kernel_size=1, bias=False),
                            nn.Conv3d(in_channels=30, out_channels=4, kernel_size=1, bias=False)])

    def _help_build_decoder(self):
        number_maps = [[item1, item2, item3] for item1, item2, item3 in
                       zip([560, 240, 120, 60], [240, 120, 60, 30], [120, 60, 30, 30])]
        return nn.ModuleList([DecodeConvUnit(channels=item) for item in number_maps])

    def _help_build_upsample(self):
        return nn.ModuleList([nn.Upsample(scale_factor=2, mode="trilinear") for i in range(4)])

    def _help_build_downsample(self):
        return nn.ModuleList([nn.MaxPool3d(kernel_size=2, stride=2) for i in range(4)])

    def _help_build_encode_conv(self):
        number_maps = [[item1, item2, item3] for item1, item2, item3 in
                       zip([4, 30, 60, 120, 240], [30, 60, 120, 240, 480], [30, 60, 120, 240, 320])]
        return nn.ModuleList([EncodeConvUnit(channels=item) for item in number_maps])

    def forward(self, input):
        encoder_level_output = []
        x = input
        # begin encoder
        for depth in range(4):
            x = self.encoder[depth](x)
            encoder_level_output.append(x)
            x = self.downsample[depth](x)
        x = self.encoder[4](x)
        # begin decode
        decode_output = []
        for depth in range(4):
            x = self.upsample[depth](x)
            x = torch.cat([x, encoder_level_output[-(depth + 1)]], dim=1)
            x = self.decoder[depth](x)
            if depth>=1:
                decode_output.append(x)
        ### get the output for deep surprviser
        seg_output=[]
        for depth in range(3):
            temp=self.seg_layer[depth](decode_output[depth])
            if depth!=2:
                temp=self.seg_up[depth](temp)
            seg_output.append(temp)
        return seg_output


class EncodeConvUnit(nn.Module):
    def __init__(self, channels):
        super(EncodeConvUnit, self).__init__()
        in_, mid_, out_ = channels
        self.layer1 = nn.Sequential(nn.Conv3d(in_channels=in_, out_channels=mid_, kernel_size=3, padding=1),
                                    nn.Dropout3d(p=dropout_rate, inplace=True),
                                    nn.InstanceNorm3d(num_features=mid_),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True))
        self.layer2 = nn.Sequential(nn.Conv3d(in_channels=mid_, out_channels=out_, kernel_size=3, padding=1),
                                    nn.Dropout3d(p=dropout_rate, inplace=True),
                                    nn.InstanceNorm3d(num_features=out_),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True))

    def forward(self, x):
        return self.layer2(self.layer1(x))


class DecodeConvUnit(nn.Module):
    def __init__(self, channels):
        super(DecodeConvUnit, self).__init__()
        in_, mid_, out_ = channels
        self.layer1 = nn.Sequential(nn.Conv3d(in_channels=in_, out_channels=mid_, kernel_size=3, padding=1),
                                    nn.Dropout3d(p=dropout_rate, inplace=True),
                                    nn.InstanceNorm3d(num_features=mid_),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True))
        self.layer2 = nn.Sequential(nn.Conv3d(in_channels=mid_, out_channels=out_, kernel_size=3, padding=1),
                                    nn.Dropout3d(p=dropout_rate, inplace=True),
                                    nn.InstanceNorm3d(num_features=out_),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True))

    def forward(self, input):
        return self.layer2(self.layer1(input))
