## visulize the attention map
import os
import pandas as pd
import torch
import numpy as np

from HookForSpaceAttention import HookFeatureExtractor
from HooKForSpaceAttentionLastLayer import HookFeatureExtractorLastLayer
from ParametersSetting import output_root_path, batch_size, patch_size, val_batch_size,vi_lastlayer_flag
from Training import BrainTumorSeg
from utils import get_vi_data_list


def run():
    space_attention_visualize_path = os.path.join(output_root_path, "vi_space_attention")
    if not os.path.exists(space_attention_visualize_path):
        os.makedirs(space_attention_visualize_path)
    all_test_path = get_vi_data_list(output_root_path)
    print("loading model...")
    hp = {"batch_size": batch_size,
          "patch_size": patch_size,
          "predict_root_path": output_root_path,
          "fold": 1,
          "val_batch_size": val_batch_size}
    best_model = os.path.join(output_root_path, "result_SpaceAttention", "epoch=599.ckpt")
    if torch.cuda.is_available():
        print("reload on gpu...")
        wm = lambda storage, loc: storage.cuda(0)
    else:
        print("reload on cpu")
        wm = lambda storage, loc: storage
    if torch.cuda.is_available():
        loaded_model = BrainTumorSeg.load_from_checkpoint(checkpoint_path=best_model, map_location=wm, hp=hp).cuda()
    else:
        loaded_model = BrainTumorSeg.load_from_checkpoint(checkpoint_path=best_model, map_location=wm, hp=hp)
    for single_test_path in all_test_path[:5]:
        if vi_lastlayer_flag:
            filename = single_test_path.split("/")[-1].split(".")[0]
            print("predicting", filename)
            hook = HookFeatureExtractorLastLayer(network=loaded_model)
            load_data = np.load(single_test_path, mmap_mode="r")
            sample = {"image": load_data[:4], "label": load_data[4]}  # data：[4,x,y,z] label:[x,y,z]
            input_data = sample["image"]
            hook.prepare_target_layer()
            out = hook(input_data)
            np.savez(os.path.join(space_attention_visualize_path, filename+"_lastlayer"),
                     attention_map_3=out[2],feature_in_3=out[0], feature_out_3=out[1])
            print("save sucessful")
        else:
            filename = single_test_path.split("/")[-1].split(".")[0]
            print("predicting", filename)
            hook = HookFeatureExtractor(network=loaded_model)
            load_data = np.load(single_test_path, mmap_mode="r")
            sample = {"image": load_data[:4], "label": load_data[4]}  # data：[4,x,y,z] label:[x,y,z]
            input_data = sample["image"][:, 0:80, 0:80, 0:80]
            hook.prepare_target_layer()
            out = hook(input_data)
            np.savez(os.path.join(space_attention_visualize_path, filename),
                     attention_map_0=out[0][2], feature_in_0=out[0][0], feature_out_0=out[0][1],
                     attention_map_1=out[1][2], feature_in_1=out[1][0], feature_out_1=out[1][1],
                     attention_map_2=out[2][2], feature_in_2=out[2][0], feature_out_2=out[2][1],
                     attention_map_3=out[3][2], feature_in_3=out[3][0], feature_out_3=out[3][1],
                     image=sample["image"], truth=sample["label"])
            print("save sucessful")

if __name__ == '__main__':
    run()

