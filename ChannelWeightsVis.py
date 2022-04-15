"""
get the channel weights:
1. feed all the training data
2. get intra and inter weights
"""
import os
import torch
import numpy as np

from HookForChannelAttention import HookFeatureExtractorLastLayer
from ParametersSetting import output_root_path, batch_size, patch_size, val_batch_size,train_npy_path,test_npy_path
from Training import BrainTumorSeg
from utils import get_folder_data_list


def run():
    all_test_path = get_folder_data_list(test_npy_path)
    print("loading model...")
    hp = {"batch_size": batch_size,
          "patch_size": patch_size,
          "predict_root_path": output_root_path,
          "fold": 1,
          "val_batch_size": val_batch_size}

    best_model = os.path.join(output_root_path, "ChannelAttention", "checkpoint","epoch=299.ckpt")
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

    save_path=os.path.join(output_root_path,"ChannelAttention","weights")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for item in all_test_path:
        filename = item[0].split("/")[-1].split(".")[0]
        print("predicting", filename)
        hook = HookFeatureExtractorLastLayer(network=loaded_model)
        load_data = np.load(item[0], mmap_mode="r")
        hook.prepare_target_layer()
        out = hook(load_data)
        np.savez(os.path.join(save_path, filename),
                     se1=out[0][0],se2=out[0][1],se3=out[0][2],se4=out[0][3],se5=out[0][4],
                 fusion1=out[1][0],fusion2=out[1][1],fusion3=out[1][2],fusion4=out[1][3])

        print("save sucess")


if __name__ == '__main__':
    run()

