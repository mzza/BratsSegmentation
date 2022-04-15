import json
import numpy as np
import os
import SimpleITK as sitk
from multiprocessing import Pool
import pandas as pd
from ParametersSetting import test_npy_path,raw_train_data_path,train_npy_path
from utils import reshape_by_padding_upper_coords, cut_off_values_upper_lower_percentile


def extract_brain_region(image, segmentation, outside_value=0):
    brain_voxels = np.where(segmentation != outside_value)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))

    # resize images
    resizer = (slice(minZidx, maxZidx), slice(minXidx, maxXidx), slice(minYidx, maxYidx))
    return image[resizer], [[minZidx, maxZidx], [minXidx, maxXidx], [minYidx, maxYidx]]


def run_star(args):
    return run(*args)


def run(folder, out_folder, name):
    print(name)
    if not os.path.isfile(os.path.join(folder, "%s_flair.nii.gz" % name)):
        return
    if not os.path.isfile(os.path.join(folder, "%s_t1.nii.gz" % name)):
        return
    if not os.path.isfile(os.path.join(folder, "%s_t1ce.nii.gz" % name)):
        return
    if not os.path.isfile(os.path.join(folder, "%s_t2.nii.gz" % name)):
        return

    t1_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, "%s_t1.nii.gz" % name))).astype(np.float32)
    t1_img_sitk = sitk.ReadImage(os.path.join(folder, "%s_t1.nii.gz" % name))
    t1c_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, "%s_t1ce.nii.gz" % name))).astype(np.float32)
    t2_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, "%s_t2.nii.gz" % name))).astype(np.float32)
    flair_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(folder, "%s_flair.nii.gz" % name))).astype(
        np.float32)

    original_shape = t1_img.shape

    brain_mask = (t1_img != t1_img[0, 0, 0]) & (t1c_img != t1c_img[0, 0, 0]) & (t2_img != t2_img[0, 0, 0]) & (
            flair_img != flair_img[0, 0, 0])

    # compute bbox of brain, This is now actually also returned when calling extract_brain_region, but was not at the
    # time this code was initially written. In order to not break anything we will keep it like it was
    brain_voxels = np.where(brain_mask != 0)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))
    with open(os.path.join(out_folder, name + ".json"), 'w') as f:
        dp = {"Patient": name,
              "orig_shp": original_shape,
              "current_shape": [maxXidx - minXidx, maxXidx - minYidx, maxZidx - minZidx],
              "bbox_z": [minZidx, maxZidx],
              "bbox_x": [minXidx, maxXidx],
              "bbox_y": [minYidx, maxYidx],
              "spacing": t1_img_sitk.GetSpacing(),
              "direction": t1_img_sitk.GetDirection(),
              "origin": t1_img_sitk.GetOrigin()},
        json.dump(dp, f, indent=4)

    t1_img, bbox = extract_brain_region(t1_img, brain_mask, 0)
    t1c_img, bbox = extract_brain_region(t1c_img, brain_mask, 0)
    t2_img, bbox = extract_brain_region(t2_img, brain_mask, 0)
    flair_img, bbox = extract_brain_region(flair_img, brain_mask, 0)


    # if mean_mode=="no_mean":
    #     print("mean mode", mean_mode)
    #     assert t1_img.shape == t1c_img.shape == t2_img.shape == flair_img.shape
    #     msk = t1_img != 0
    #     t1_img = cut_off_values_upper_lower_percentile(t1_img, msk, 2., 98.)
    #
    #     msk = t1c_img != 0
    #     t1c_img = cut_off_values_upper_lower_percentile(t1c_img, msk, 2., 98.)
    #
    #     msk = t2_img != 0
    #     t2_img = cut_off_values_upper_lower_percentile(t2_img, msk, 2., 98.)
    #
    #     msk = flair_img != 0
    #     flair_img = cut_off_values_upper_lower_percentile(flair_img, msk, 2., 98.)

    assert t1_img.shape == t1c_img.shape == t2_img.shape == flair_img.shape
    msk = t1_img != 0
    tmp = cut_off_values_upper_lower_percentile(t1_img, msk, 2., 98.)
    t1_img[msk] = (t1_img[msk] - tmp[msk].mean()) / tmp[msk].std()

    msk = t1c_img != 0
    tmp = cut_off_values_upper_lower_percentile(t1c_img, msk, 2., 98.)
    t1c_img[msk] = (t1c_img[msk] - tmp[msk].mean()) / tmp[msk].std()

    msk = t2_img != 0
    tmp = cut_off_values_upper_lower_percentile(t2_img, msk, 2., 98.)
    t2_img[msk] = (t2_img[msk] - tmp[msk].mean()) / tmp[msk].std()

    msk = flair_img != 0
    tmp = cut_off_values_upper_lower_percentile(flair_img, msk, 2., 98.)
    flair_img[msk] = (flair_img[msk] - tmp[msk].mean()) / tmp[msk].std()


# if mean_mode=="all_mean":
#     print("mean mode", mean_mode)
#     assert t1_img.shape == t1c_img.shape == t2_img.shape == flair_img.shape
#     msk = t1_img != 0
#     tmp = cut_off_values_upper_lower_percentile(t1_img, msk, 2., 98.)
#     t1_img= (t1_img - tmp.mean()) / tmp.std()
#
#     msk = t1c_img != 0
#     tmp = cut_off_values_upper_lower_percentile(t1c_img, msk, 2., 98.)
#     t1c_img = (t1c_img - tmp.mean()) / tmp.std()
#
#     msk = t2_img != 0
#     tmp = cut_off_values_upper_lower_percentile(t2_img, msk, 2., 98.)
#     t2_img= (t2_img - tmp.mean()) / tmp.std()
#
#     msk = flair_img != 0
#     tmp = cut_off_values_upper_lower_percentile(flair_img, msk, 2., 98.)
#     flair_img = (flair_img - tmp.mean()) / tmp.std()


    shp = t1_img.shape
    pad_size = np.max(np.vstack((np.array([128, 128, 128]), np.array(shp))), 0)
    t1_img = reshape_by_padding_upper_coords(t1_img, pad_size, 0)
    t1c_img = reshape_by_padding_upper_coords(t1c_img, pad_size, 0)
    t2_img = reshape_by_padding_upper_coords(t2_img, pad_size, 0)
    flair_img = reshape_by_padding_upper_coords(flair_img, pad_size, 0)

    all_data = np.zeros([4] + list(t1_img.shape), dtype=np.float32)
    all_data[0] = t1_img
    all_data[1] = t1c_img
    all_data[2] = t2_img
    all_data[3] = flair_img
    np.save(os.path.join(out_folder, name), all_data)

    return dp[0]


def run_preprocessing_BraTS2017_trainSet(base_folder, folder_out):
    all_information_list = []
    all_information_list = all_information_list + multi_process_each_patient(base_folder, folder_out)
    print("all the data has been proprocessed")

    index = range(len(all_information_list))
    all_information_dict = dict(zip(index, all_information_list))
    all_information_cvs = pd.DataFrame(all_information_dict)
    all_information_cvs = all_information_cvs.transpose()

    all_information_cvs.to_csv(os.path.join(folder_out, "all_information.csv"), index=0)
    print("csv saved")


def multi_process_each_patient(base_folder, folder_out):
    if not os.path.isdir(base_folder):
        print("erro fold not exist")
    patients = os.listdir(base_folder)
    patients.sort()
    fldrs = [os.path.join(base_folder, pt) for pt in patients]
    p = Pool()
    data_information_list = p.map(run_star, zip(fldrs, [folder_out] * len(patients), patients))
    p.close()
    p.join()

    return data_information_list


if __name__ == '__main__':
    print("start...")
    run_preprocessing_BraTS2017_trainSet(, test_npy_path)
    print("preprocess finished...")
