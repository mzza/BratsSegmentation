import torch
import numpy as np
from torch import nn
from utils import pad_nd_image, compute_steps_for_sliding_window, get_device, get_gaussian,to_cuda,to_tensor
from ParametersSetting import patch_size, step_size, num_class

class HookFeatureExtractor(nn.Module):
    def __init__(self, network):
        super(HookFeatureExtractor, self).__init__()
        torch.cuda.empty_cache()
        self.network = network
        self.network.eval()
        self.inputs = []
        self.outputs=[]

    def get_input_array(self, m, i, o):
        temp= torch.squeeze(i[0].data.clone()).detach().cpu().numpy()
        self.inputs.append(temp)
        #print('Input Array Size: ', temp.shape)


    def get_output_array(self, m, i, o):
        temp=torch.squeeze(o.data.clone()).detach().cpu().numpy()
        self.outputs.append(temp)
        #print('Output Array Size: ', temp.shape)

    def prepare_target_layer(self):
        for i in range(4):
            attention_map_layer = self.network._modules["model"].grid_attention[i].space_attention
            attention_feature_layer=self.network._modules["model"].grid_attention[i]

            # first get in and out feature
            attention_feature_layer.register_forward_hook(self.get_input_array)
            attention_feature_layer.register_forward_hook(self.get_output_array)

            # then get attention map
            attention_map_layer.register_forward_hook(self.get_output_array)

    def forward(self, x):
        print("get step...")
        data, slicer = pad_nd_image(x, new_shape=patch_size, return_slicer=True)
        steps = compute_steps_for_sliding_window(patch_size, data.shape[1:], step_size)

        data=torch.unsqueeze(to_tensor(data),dim=0)
        if torch.cuda.is_available():
            data = to_cuda(data)
        print("star forward....")
        boundary_list=[]# length of 8
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
                    boundary_list.append([lb_x,ub_x, lb_y,ub_y, lb_z,ub_z])

        print("forwad finish")
        # list 中的顺序从low level to high level
        feature_in=self.inputs# 32
        feature_out=self.outputs[1::2]#32
        attention_map=self.outputs[::2]#32
        data_shape=list(data.size())[2:]
        boundary_list = np.array(boundary_list)
        print("length of feature in",len(feature_in))
        print("length of feature out",len(feature_out))
        print("length of attention map", len(attention_map))
        print("data shape", data_shape)
        print("origin boundary shape",boundary_list)

        print("star tiling for all levels...")

        return get_all_level_titling(feature_in, feature_out, attention_map, data_shape,boundary_list,slicer)

def get_all_level_titling(feature_in_all_level, feature_out_all_level, attention_map_all_level,
                          data_shape,boundary_list, slicer):
    all_result=[]
    for i in range(4):
        print("start level",i)
        current_feature_in= feature_in_all_level[i::4]
        print("current feature in",current_feature_in[0].shape)
        assert current_feature_in[0].size==current_feature_in[2].size ,"not equal shape"
        current_feature_out = feature_out_all_level[i::4]
        print("current feature out", current_feature_out[0].shape)
        assert current_feature_out[0].size == current_feature_out[2].size, "not equal shape"
        current_attention_map = attention_map_all_level[i::4]
        print("current attention map", current_attention_map[0].shape)
        assert current_attention_map[0].size == current_attention_map[2].size, "not equal shape"
        current_boundary_list=(boundary_list/pow(2,3-i)).astype(np.uint8)
        print("current boundary_list", current_boundary_list)
        current_titling_shape=[int(item/pow(2,3-i)) for item in data_shape]
        print("current titling shape", current_titling_shape)

        current_slicer_flag=False if i<3 else True
        print("current slicer flag",current_slicer_flag)
        current_results=titling_patch_one_level(current_feature_in, current_feature_out, current_attention_map,current_boundary_list,
                                current_titling_shape,slicer,current_slicer_flag)
        print("current level finish")
        all_result.append(current_results)
    return all_result


def titling_patch_one_level(feature_in_one_level, feature_out_one_level, attention_map_one_level,
                            boundary_one_level, titling_stage, slicer, slicer_flag=False):
    patch_shape=feature_in_one_level[0].shape[1:]
    gaussian_importance_map = get_gaussian(patch_shape, sigma_scale=1. / 8)
    print("gaussian_importance_map", gaussian_importance_map.shape)
    # gaussian_importance_map = to_tensor(gaussian_importance_map)
    # if torch.cuda.is_available():
    #     gaussian_importance_map = to_cuda(gaussian_importance_map)
    add_for_normalize = gaussian_importance_map

    channels=feature_in_one_level[0].shape[0]
    # aggregated_feature_in = torch.zeros([channels] + titling_stage, dtype=torch.float32, device=get_device())
    # aggregated_feature_out = torch.zeros([channels] + titling_stage, dtype=torch.float32, device=get_device())
    # aggregated_attention_out = torch.zeros([num_class] + titling_stage, dtype=torch.float32, device=get_device())
    #
    # aggregated_gaussian_for_attention_map = torch.zeros([num_class] + titling_stage, dtype=torch.float32, device=get_device())
    # aggregated_gaussian_for_feature = torch.zeros([channels] + titling_stage, dtype=torch.float32, device=get_device())

    aggregated_feature_in = np.zeros([channels] + titling_stage, dtype=np.float32)
    aggregated_feature_out = np.zeros([channels] + titling_stage, dtype=np.float32)
    aggregated_attention_out = np.zeros([num_class] + titling_stage, dtype=np.float32)

    aggregated_gaussian_for_attention_map = np.zeros([num_class] + titling_stage, dtype=np.float32)
    aggregated_gaussian_for_feature = np.zeros([channels] + titling_stage, dtype=np.float32)

    for i in range(len(boundary_one_level)):
        lb_x,ub_x, lb_y,ub_y, lb_z,ub_z=boundary_one_level[i]
        aggregated_feature_in[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += feature_in_one_level[i]
        aggregated_feature_out[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += feature_out_one_level[i]
        aggregated_attention_out[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += attention_map_one_level[i]
        aggregated_gaussian_for_attention_map[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_normalize
        aggregated_gaussian_for_feature[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_normalize

    if slicer_flag:
        slicer = tuple(
            [slice(0, aggregated_feature_in.shape[i]) for i in
             range(len(aggregated_feature_in.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_feature_in = aggregated_feature_in[slicer]
        aggregated_feature_out = aggregated_feature_out[slicer]
        aggregated_attention_out = aggregated_attention_out[slicer]
        aggregated_gaussian_for_attention_map = aggregated_gaussian_for_attention_map[slicer]
        aggregated_gaussian_for_feature = aggregated_gaussian_for_feature[slicer]

    aggregated_feature_in = (aggregated_feature_in / aggregated_gaussian_for_feature)
    aggregated_feature_out = (aggregated_feature_out / aggregated_gaussian_for_feature)
    aggregated_attention_out = (aggregated_attention_out / aggregated_gaussian_for_attention_map)

    print("aggreating finish")

    return aggregated_feature_in, aggregated_feature_out,


