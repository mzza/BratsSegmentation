import torch
import numpy as np
from torch import nn
from utils import pad_nd_image, compute_steps_for_sliding_window, get_device, get_gaussian, to_cuda, to_tensor
from ParametersSetting import patch_size, step_size, num_class


class HookNoWindow(nn.Module):
    def __init__(self, network):
        super(HookNoWindow, self).__init__()
        torch.cuda.empty_cache()
        self.network = network
        # self.network.eval()
        self.feature_in = None
        self.feature_out = None
        self.attention_map = None
        self.model="MMF"

    def get_feature_in_array(self, m, i, o):
        self.feature_in = torch.squeeze(i[0].data.clone()).detach().cpu().numpy()
        # print('Input Array Size: ', temp.shape)

    def get_feature_out_array(self, m, i, o):
        self.feature_out = torch.squeeze(o.data.clone()).detach().cpu().numpy()
        # print('Output Array Size: ', temp.shape)

    def get_attention_map_array(self, m, i, o):
        self.attention_map = torch.squeeze(o.data.clone()).detach().cpu().numpy()
        # print('Output Array Size: ', temp.shape)

    def prepare_target_layer(self,):
        if self.model=="Grid":
            attention_layer = self.network._modules["model"].grid_attention[3].space_attention
            feature_layer = self.network._modules["model"].grid_attention[3]
        else:
            print("using mmf...")
            attention_layer = self.network._modules["model"].mmf[3].space_attention
            feature_layer = self.network._modules["model"].mmf[3]

        # first get in and out feature
        feature_layer.register_forward_hook(self.get_feature_in_array)
        feature_layer.register_forward_hook(self.get_feature_out_array)

        # then get attention map
        attention_layer.register_forward_hook(self.get_attention_map_array)

    def forward(self, x):
        torch.cuda.empty_cache()
        x=x[:,0:128,0:128,0:128]
        data = torch.unsqueeze(to_tensor(x), dim=0)
        if torch.cuda.is_available():
            data = to_cuda(data)
        self.network(data)

        return self.feature_in,self.feature_out,self.attention_map
