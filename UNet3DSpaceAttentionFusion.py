# use the modality fusion
import torch
from torch import nn
dropout_rate = 0.2


class UNet3DSpaceAttentionFusion(nn.Module):
    def __init__(self):
        print("using MMF")
        super(UNet3DSpaceAttentionFusion, self).__init__()
        self.encoder = self._help_build_encode_conv()
        self.downsample = self._help_build_downsample()
        self.upsample = self._help_build_upsample()
        self.decoder = self._help_build_decoder()
        self.seg_layer = self._help_build_seg_layer()
        self.seg_up=self._help_build_seg_up_layer()
        self.mmf=self._help_build_mmf()

    def _help_build_mmf(self):
        return nn.ModuleList([MMF(num_encode_feature=256, num_decode_feature=320),
                              MMF(num_encode_feature=128, num_decode_feature=144),
                              MMF(num_encode_feature=64, num_decode_feature=68),
                              MMF(num_encode_feature=32, num_decode_feature=33)])

    def _help_build_seg_up_layer(self):
        return nn.ModuleList([nn.Upsample(scale_factor=4, mode="trilinear"),
                              nn.Upsample(scale_factor=2, mode="trilinear")])

    def _help_build_seg_layer(self):
        return nn.ModuleList([nn.Conv3d(in_channels=68, out_channels=4, kernel_size=1, bias=False),
                            nn.Conv3d(in_channels=33, out_channels=4, kernel_size=1, bias=False),
                            nn.Conv3d(in_channels=32, out_channels=4, kernel_size=1, bias=False)])

    def _help_build_decoder(self):
        number_maps = [[item1, item2, item3] for item1, item2, item3 in
                       zip([576, 272, 132, 65], [288, 136, 66, 32], [144, 68, 33, 32])]
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
        # begin decode
        decode_output = []
        for depth in range(4):
            # num_encode_feature=encoder_level_output[-(depth + 1)].shape[1]
            # num_decode_feature=x.shape[1]
            # attention_module=Grid_Attention(num_encode_feature,num_decode_feature)
            refined_encode_feature=self.mmf[depth](encoder_level_output[-(depth + 1)], x)
            x = self.upsample[depth](x)
            x = torch.cat([x, refined_encode_feature], dim=1)
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
        num_group=4
        if out_==320:
            num_group=1
            # print("lastlayer not using group conv")
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


class MMF(nn.Module):
    def __init__(self, num_encode_feature, num_decode_feature):
        super(MMF, self).__init__()

        num_modality_feature=int(num_encode_feature / 4)
        self.conv_gating = nn.Sequential(nn.Upsample(scale_factor=2, mode="trilinear"),
                        nn.Conv3d(in_channels=num_decode_feature, out_channels=num_modality_feature, kernel_size=1))

        self.space_attention = nn.Sequential(
            nn.Conv3d(num_modality_feature*8,num_modality_feature*4,kernel_size=1,groups=4),
            nn.InstanceNorm3d(num_modality_feature*4),nn.LeakyReLU(),
            nn.Conv3d(num_modality_feature*4, num_modality_feature*4 ,kernel_size=3,padding=1,groups=4),
            nn.InstanceNorm3d(num_modality_feature*4), nn.LeakyReLU(),
        nn.Conv3d(num_modality_feature*4,4,kernel_size=3,padding=1,groups=4),
            nn.Sigmoid() )
        # self.refine = nn.Sequential(
        #     nn.Conv3d(int(num_encode_feature*8),num_modality_feature*4,kernel_size=1,groups=4),
        #     nn.InstanceNorm3d(num_modality_feature*4),nn.LeakyReLU(),
        #     nn.Conv3d(num_modality_feature*4, num_modality_feature*4 ,kernel_size=3,padding=1,groups=4),
        #     nn.InstanceNorm3d(num_modality_feature*4), nn.LeakyReLU(),
        # nn.Conv3d(num_modality_feature*4,4,kernel_size=3,padding=1,groups=4))

    def forward(self, x, g):
        """
        x:[b,4c,h,w,d]
        g:[b,c,h,w,d]
        output:[b,c,h,w,d]
        :return:
        """
        origin_separate_modality = torch.chunk(x, 4, dim=1)  # a list containing [b,c,h,w,d]
        gating_space = self.conv_gating(g)  # [b,c,h,w,d]

        m1=torch.cat([origin_separate_modality[0],gating_space],dim=1)
        m2 = torch.cat([origin_separate_modality[1], gating_space],dim=1)
        m3 = torch.cat([origin_separate_modality[2], gating_space],dim=1)
        m4 = torch.cat([origin_separate_modality[3], gating_space],dim=1)

        modality_space=torch.cat([m1,m2,m3,m4],dim=1)
        attention_map = self.space_attention(modality_space)
        separate_space_map = torch.chunk(attention_map, 4, dim=1) # [b,4,h,w,d]

        # modality1_refer = seperate_space_map[0].expand_as(gating_space) * gating_space  # [b,c,h,w,d]
        # modality2_refer = seperate_space_map[1].expand_as(gating_space) * gating_space
        # modality3_refer = seperate_space_map[2].expand_as(gating_space) * gating_space
        # modality4_refer = seperate_space_map[3].expand_as(gating_space) * gating_space
        #
        # mr1=torch.cat([modality1_refer,origin_seperate_modality[0]],dim=1)
        # mr2 = torch.cat([modality2_refer, origin_seperate_modality[1]],dim=1)
        # mr3 = torch.cat([modality3_refer, origin_seperate_modality[2]],dim=1)
        # mr4 = torch.cat([modality4_refer, origin_seperate_modality[3]],dim=1)
        #
        # refer_space=torch.cat([mr1,mr2,mr3,mr4])
        # modality_refined = self.refine(refer_space)

        modality1_refer = separate_space_map[0].expand_as(origin_separate_modality[0]) * origin_separate_modality[0]  # [b,c,h,w,d]
        modality2_refer = separate_space_map[1].expand_as(origin_separate_modality[1]) * origin_separate_modality[1]
        modality3_refer = separate_space_map[2].expand_as(origin_separate_modality[2]) * origin_separate_modality[2]
        modality4_refer = separate_space_map[3].expand_as(origin_separate_modality[3]) * origin_separate_modality[3]

        modality_refer=torch.cat([modality1_refer,modality2_refer,modality3_refer,modality4_refer],dim=1)

        return modality_refer + x



















