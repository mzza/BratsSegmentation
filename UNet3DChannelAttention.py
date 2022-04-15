# 1. only channel attention
# 2. intra-modality attention  //SE block
# 3. inter-modality attention  //modality  fusion
import torch
from torch import nn
dropout_rate = 0.2


class UNet3DChannelAttention(nn.Module):
    def __init__(self):
        print("using channel attention...")
        super(UNet3DChannelAttention, self).__init__()
        self.encoder = self._help_build_encode_conv()
        self.downsample = self._help_build_downsample()
        self.upsample = self._help_build_upsample()
        self.decoder = self._help_build_decoder()
        self.seg_layer = self._help_build_seg_layer()
        self.seg_up=self._help_build_seg_up_layer()
        self.se_block=self._help_build_channel_attention_se()
        self.modality_attention_block = self._help_build_GridAttentionWithSE()


    def _help_build_channel_attention_se(self):
        return nn.ModuleList([ChannelAttentionSE(num_encode_feature=320),
                              ChannelAttentionSE(num_encode_feature=144),
                              ChannelAttentionSE(num_encode_feature=68),
                              ChannelAttentionSE(num_encode_feature=32),
                              ChannelAttentionSE(num_encode_feature=32)])
    def _help_build_GridAttentionWithSE(self):
        return nn.ModuleList([GridAttentionWithSE(num_encode_feature=256, num_decode_feature=320),
                              GridAttentionWithSE(num_encode_feature=128, num_decode_feature=144),
                              GridAttentionWithSE(num_encode_feature=64, num_decode_feature=68),
                              GridAttentionWithSE(num_encode_feature=32, num_decode_feature=32)])

    def _help_build_seg_up_layer(self):
        return nn.ModuleList([nn.Upsample(scale_factor=4, mode="trilinear"),
                              nn.Upsample(scale_factor=2, mode="trilinear")])

    def _help_build_seg_layer(self):
        return nn.ModuleList([nn.Conv3d(in_channels=68, out_channels=4, kernel_size=1, bias=False),
                            nn.Conv3d(in_channels=32, out_channels=4, kernel_size=1, bias=False),
                            nn.Conv3d(in_channels=32, out_channels=4, kernel_size=1, bias=False)])

    def _help_build_decoder(self):
        number_maps = [[item1, item2, item3] for item1, item2, item3 in
                       zip([576, 272, 132, 64], [288, 136, 66, 32], [144, 68, 32, 32])]
        return nn.ModuleList([DecodeConvUnit(channels=item) for item in number_maps])

    def _help_build_upsample(self):
        return nn.ModuleList([nn.Upsample(scale_factor=2, mode="trilinear") for i in range(4)])

    def _help_build_downsample(self):
        return nn.ModuleList([nn.MaxPool3d(kernel_size=2, stride=2) for i in range(4)])

    def _help_build_encode_conv(self):
        number_maps = [[item1, item2, item3] for item1, item2, item3 in
                       zip([4, 32, 64, 128, 256], [32, 64, 128, 256, 512], [32, 64, 128, 256, 320])]
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

        x=self.se_block[0](x)
        # begin decode
        decode_output = []
        for depth in range(4):
            refined_encode_feature = self.modality_attention_block[depth](encoder_level_output[-(depth + 1)], x)
            x = self.upsample[depth](x)
            x = torch.cat([x, refined_encode_feature], dim=1)
            x = self.decoder[depth](x)
            x=self.se_block[depth + 1](x)
            if depth >= 1:
                decode_output.append(x)
        ### get the output for deep surprviser
        seg_output = []
        for depth in range(3):
            temp = self.seg_layer[depth](decode_output[depth])
            if depth != 2:
                temp = self.seg_up[depth](temp)
            seg_output.append(temp)
        return seg_output


class EncodeConvUnit(nn.Module):
    def __init__(self, channels):
        super(EncodeConvUnit, self).__init__()
        in_, mid_, out_ = channels
        num_group=4
        if out_==320:
            num_group=1
        self.layer1 = nn.Sequential(nn.Conv3d(in_channels=in_, out_channels=mid_, kernel_size=3, padding=1,groups=num_group),
                                    nn.Dropout3d(p=dropout_rate, inplace=True),
                                    nn.InstanceNorm3d(num_features=mid_),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True))
        self.layer2 = nn.Sequential(nn.Conv3d(in_channels=mid_, out_channels=out_, kernel_size=3, padding=1,groups=num_group),
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


class GridAttentionWithSE(nn.Module):
    def __init__(self, num_encode_feature, num_decode_feature):
        super(GridAttentionWithSE, self).__init__()

        num_attention_feature=int(num_encode_feature / 1)
        self.conv_modality = nn.Conv3d(in_channels=num_encode_feature, out_channels=num_attention_feature,
                                                        kernel_size=1, groups=4)
        self.conv_gating = nn.Sequential(nn.Upsample(scale_factor=2, mode="trilinear"),
                        nn.Conv3d(in_channels=num_decode_feature, out_channels=num_attention_feature, kernel_size=1))

        self.space_attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(num_attention_feature, 4, 1, groups=4),
            nn.Sigmoid())
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=1),
            nn.Conv3d(num_encode_feature,int(num_encode_feature/2),kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(num_encode_feature/2), num_encode_feature ,kernel_size=1),
            nn.Sigmoid())

    def forward(self, x, g):
        """
        x:[b,4c,h,w,d]
        g:[b,c,h,w,d]
        output:[b,4c,h,w,d]
        :return:
        """
        #
        origin_seperat_modality = torch.chunk(x, 4, dim=1)  # a list containing [b,c,h,w,d]

        modality_space = self.conv_modality(x)
        gating_space = self.conv_gating(g)  # [b,c,h,w,d]

        attention_map = self.space_attention(modality_space + gating_space)
        seperate_space_map = torch.chunk(attention_map, 4, dim=1)  # [b,4,h,w,d]

        s1_refined = seperate_space_map[0].expand_as(origin_seperat_modality[0]) * origin_seperat_modality[0]  # [b,c,h,w,d]
        s2_refined = seperate_space_map[1].expand_as(origin_seperat_modality[1]) * origin_seperat_modality[1]
        s3_refined = seperate_space_map[2].expand_as(origin_seperat_modality[2]) * origin_seperat_modality[2]
        s4_refined = seperate_space_map[3].expand_as(origin_seperat_modality[3]) * origin_seperat_modality[3]

        space_refined = torch.cat([s1_refined, s2_refined,
                                      s3_refined, s4_refined], dim=1)
        x= space_refined + x

        channel_weights = self.channel_attention(x)#[b,4c,1,1,1]

        channel_refined = channel_weights.expand_as(x) * x # [b,c,h,w,d]

        return channel_refined + x

class ChannelAttentionSE(nn.Module):
    def __init__(self, num_encode_feature):
        super(ChannelAttentionSE, self).__init__()

        self.SE=nn.Sequential(nn.AdaptiveAvgPool3d(output_size=1),
                                         nn.Conv3d(in_channels=num_encode_feature,out_channels=int(num_encode_feature/2),
                                                   kernel_size=1),
                                         nn.ReLU(inplace=True),
                                        nn.Conv3d(in_channels=int(num_encode_feature/2),out_channels=num_encode_feature,
                                                  kernel_size=1),
                                         nn.Sigmoid())


    def forward(self, x):
        """
        x:[b,4c,h,w,d]
        g:[b,c,h,w,d]
        output:[b,4c,h,w,d]
        :return:
        """
        #
        channel_weights=self.SE(x)#[b,c,1,1,1]
        modality_refine = channel_weights.expand_as(x) * x  # [b,c,h,w,d]

        return modality_refine + x