import torch
import numpy as np
from torch import nn
from utils import pad_nd_image, compute_steps_for_sliding_window, get_device, get_gaussian, to_cuda, to_tensor
from ParametersSetting import patch_size, step_size, num_class


class HookFeatureExtractorLastLayer(nn.Module):
    def __init__(self, network):
        super(HookFeatureExtractorLastLayer, self).__init__()
        print("get channel weights")
        torch.cuda.empty_cache()
        self.network = network
        # self.network.eval()
        self.fusion_weights = []
        self.se_weights = []

    def get_fusion_weights(self, m, i, o):
        self.fusion_weights.append(torch.squeeze(o.data.clone()).detach().cpu().numpy())
        # print('Output Array Size: ', temp.shape)

    def get_se_weights(self, m, i, o):
        self.se_weights.append(torch.squeeze(o.data.clone()).detach().cpu().numpy())
        # print('Output Array Size: ', temp.shape)

    def prepare_target_layer(self):
        fusion_layer=[]
        se_layer=[]
        for i in range(4):
            fusion_layer.append(self.network._modules["model"].channel_attention_fusion[i].channel_attention)
            fusion_layer[-1].register_forward_hook(self.get_fusion_weights)

        for i in range(5):
            se_layer.append(self.network._modules["model"].channel_attention_se[i].SE)
            se_layer[-1].register_forward_hook(self.get_se_weights)




    def forward(self, x):
        print("get step...")
        data, slicer = pad_nd_image(x, new_shape=patch_size, return_slicer=True)
        steps = compute_steps_for_sliding_window(patch_size, data.shape[1:], step_size)

        data = torch.unsqueeze(to_tensor(data), dim=0)
        if torch.cuda.is_available():
            data = to_cuda(data)
        se_weights=[np.zeros(shape=(num_channels,)) for num_channels in [320,256,128,64,32]]
        fusion_weights=[np.zeros(shape=(num_channels,))for num_channels in [256,128,64,32]]

        i=1
        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    torch.cuda.empty_cache()
                    lb_z = z
                    ub_z = z + patch_size[2]
                    self.network(data[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z])
                    print(self.se_weights[0].shape)
                    se_weights= [x+y for x,y in zip(se_weights,self.se_weights)]
                    fusion_weights = [x+y for x ,y in zip(fusion_weights,self.fusion_weights)]

                    print(i)
                    i=i+1
        se_weights=[(item/8).astype(np.float16)  for item in se_weights]
        fusion_weights=[(item/8).astype(np.float16) for item in fusion_weights]

        return se_weights, fusion_weights