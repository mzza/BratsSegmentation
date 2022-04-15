#### use this scrip if predict not wee done
import os

import pandas as pd
import numpy as np
from LossFunction import DiceForSample
from ParametersSetting import output_root_path
from utils import get_name_segnpz_pair_for_test, to_tensor

output_folder = os.path.join(output_root_path, "test_result_baseline")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("compute dice...")
name_segnpz_list = get_name_segnpz_pair_for_test(os.path.join(output_folder, "seg"))
all_dice = []
names = []
for name, path in name_segnpz_list.items():
    print(name)
    names.append(name)
    data = np.load(path)
    metrics = DiceForSample()
    metrics.forward_for_test(to_tensor(data["seg"]).unsqueeze(dim=0), to_tensor(data["truth"]).unsqueeze(dim=0))
    del data
    val_dice_perclass = metrics.get_dice_per_class().numpy().tolist()
    val_dice_region = metrics.get_dice_region().numpy().tolist()
    all_dice.append([val_dice_perclass[2], val_dice_perclass[1], val_dice_perclass[3],
                     val_dice_region[0], val_dice_region[1]])
all_dice = np.array(all_dice)
mean = all_dice.mean(axis=0, keepdims=True)
all_dice = np.concatenate([all_dice, mean], axis=0)

names.append("mean")

df = pd.DataFrame(data=all_dice, columns=["edema", "non_enhance", "enhance", "core", "whole"], index=names)
df.to_csv(os.path.join(output_folder, "GridAttention","dice_result.csv"))
