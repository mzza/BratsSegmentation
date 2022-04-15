# GET five fold for training
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ParametersSetting import output_root_path, train_npy_path
from utils import get_all_image_names_path_pair

information_csv = os.path.join(output_root_path, "data_information", "all_information.csv")
df = pd.read_csv(information_csv)
grade = [int(item == "HGG") for item in list(df["Grade"].values)]
df["label"] = grade
name_path_pair = get_all_image_names_path_pair(train_npy_path)

name_in_csv = list(df["Patient"].values)
path = [name_path_pair[name] for name in name_in_csv]
df["predict_root_path"] = path

kfold = StratifiedKFold(n_splits=5, random_state=12)
kfold_save_path = os.path.join(output_root_path, "KFold")
if not os.path.isdir(kfold_save_path):
    os.makedirs(kfold_save_path)

for num, (train_index, val_index) in enumerate(kfold.split(X=list(df.index), y=grade)):
    print('train -  {}   |   test -  {}'.format(train_index, val_index))
    train_excel = df.loc[train_index]
    val_excel = df.loc[val_index]
    excel_name = os.path.join(kfold_save_path, "fold_" + str(num + 1) + ".xlsx")
    with pd.ExcelWriter(excel_name) as writer:
        train_excel.to_excel(writer, sheet_name="train")
        val_excel.to_excel(writer, sheet_name="val")
print("Kfold finish...")
