import torch
import numpy as np
from torch import nn
from utils import pad_nd_image, compute_steps_for_sliding_window, get_device, get_gaussian, to_cuda, to_tensor
from ParametersSetting import patch_size, step_size, num_class

gaussian_importance_map = get_gaussian(patch_size, sigma_scale=1. / 8)
print("gaussian_importance_map", gaussian_importance_map.shape)
# gaussian_importance_map = to_tensor(gaussian_importance_map)
# if torch.cuda.is_available():
#     gaussian_importance_map = to_cuda(gaussian_importance_map)
add_for_normalize = gaussian_importance_map


class HookFeatureExtractorLastLayer(nn.Module):
    def __init__(self, network):
        super(HookFeatureExtractorLastLayer, self).__init__()
        print("using window last layer")
        torch.cuda.empty_cache()
        self.network = network
        # self.network.eval()
        self.feature_in = None
        self.feature_out = None
        self.attention_map = None
        self.model="Grid"

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
            attention_layer = self.network._modules["model"].mmf[3].space_attention
            feature_layer = self.network._modules["model"].mmf[3]

        # first get in and out feature
        feature_layer.register_forward_hook(self.get_feature_in_array)
        feature_layer.register_forward_hook(self.get_feature_out_array)

        # then get attention map
        attention_layer.register_forward_hook(self.get_attention_map_array)

    def forward(self, x):
        print("get step...")
        data, slicer = pad_nd_image(x, new_shape=patch_size, return_slicer=True)
        steps = compute_steps_for_sliding_window(patch_size, data.shape[1:], step_size)

        data = torch.unsqueeze(to_tensor(data), dim=0)
        if torch.cuda.is_available():
            data = to_cuda(data)
        channels = 32
        titling_stage = list(data.shape)[2:]
        aggregated_feature_in = np.zeros([channels] + titling_stage, dtype=np.float32)
        aggregated_feature_out = np.zeros([channels] + titling_stage, dtype=np.float32)
        aggregated_gaussian_for_feature = np.zeros([channels] + titling_stage, dtype=np.float32)

        aggregated_attention_map = np.zeros([num_class] + titling_stage, dtype=np.float32)
        aggregated_gaussian_for_attention_map = np.zeros([num_class] + titling_stage, dtype=np.float32)

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
                    aggregated_feature_in[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += self.feature_in*gaussian_importance_map
                    aggregated_feature_out[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += self.feature_out*gaussian_importance_map
                    aggregated_attention_map[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += self.attention_map*gaussian_importance_map
                    aggregated_gaussian_for_attention_map[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_normalize
                    aggregated_gaussian_for_feature[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_normalize

        slicer = tuple(
            [slice(0, aggregated_feature_in.shape[i]) for i in
             range(len(aggregated_feature_in.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_feature_in = aggregated_feature_in[slicer]
        aggregated_feature_out = aggregated_feature_out[slicer]
        aggregated_attention_map = aggregated_attention_map[slicer]
        aggregated_gaussian_for_attention_map = aggregated_gaussian_for_attention_map[slicer]
        aggregated_gaussian_for_feature = aggregated_gaussian_for_feature[slicer]

        aggregated_feature_in = (aggregated_feature_in / aggregated_gaussian_for_feature)
        aggregated_feature_out = (aggregated_feature_out / aggregated_gaussian_for_feature)
        aggregated_attention_map = (aggregated_attention_map / aggregated_gaussian_for_attention_map)

        return aggregated_feature_in.astype(np.float16), aggregated_feature_out.astype(np.float16), aggregated_attention_map.astype(np.float16)
