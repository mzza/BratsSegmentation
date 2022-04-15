import os
import pathlib
from multiprocessing import Queue, Process

import numpy as np
import pandas as pd
import torch
from scipy.ndimage.filters import gaussian_filter

def cut_off_values_upper_lower_percentile(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):
    if mask is None:
        mask = image != image[0, 0, 0]
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    return res

def reshape_by_padding_upper_coords(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
    if pad_value is None:
        if len(shape) == 2:
            pad_value = image[0, 0]
        elif len(shape) == 3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    if len(shape) == 2:
        res[0:0 + int(shape[0]), 0:0 + int(shape[1])] = image
    elif len(shape) == 3:
        res[0:0 + int(shape[0]), 0:0 + int(shape[1]), 0:0 + int(shape[2])] = image
    return res


def tensor_to_onehot(target, num_class):
    # target = target.to(torch.long)
    if not torch.is_tensor(target):
        target=to_tensor(target)

    shape =target.shape
    target = target.unsqueeze(dim=1).to(torch.long)
    # shape.insert(1,num_class)
    y_onehot = torch.zeros(shape[0], num_class, shape[1], shape[2], shape[3], dtype=torch.long)
    if target.device.type == "cuda":
        y_onehot = y_onehot.cuda(target.device.index)
    y_onehot.scatter_(1, target, 1)

    return y_onehot


def get_all_image_names_path_pair(image_path):
    data_root = pathlib.Path(image_path)
    all_image_paths = list(data_root.glob("*.npy"))
    all_image_paths = [str(path).replace("\\", "/") for path in all_image_paths]

    names_path_pair = dict()
    for item in all_image_paths:
        name = item.split(".")[-2].split("/")[-1]
        names_path_pair[name] = item
    return names_path_pair


def get_list_from_csv_string(item):
    item = item[1:-1]
    item = item.split(",")
    item = [int(x) for x in item]
    return item


def get_pad_shape(max_shape):
    shape = np.array([16 * i for i in range(1, 15)])
    pad_shape = []
    for item in max_shape:
        index = list((item <= shape)).index(1)
        pad_shape.append(shape[index])
    return pad_shape


def compute_steps_for_sliding_window(patch_size, image_size, step_size):
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 32 and step_size of 0.5, then we want to make 4 steps starting at coordinate 0, 27, 55, 78
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


def get_gaussian(patch_size, sigma_scale=1. / 8):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False,
                 shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit

    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array(
            [new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in
             range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]] * num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.contiguous().cuda(gpu_id, non_blocking=True)
    return data

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

def to_tensor(d):
    if isinstance(d, list):
        d = [to_tensor(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d

def get_name_json_pair_for_test(path):
    data_root = pathlib.Path(path)
    all_json_paths = list(data_root.glob("*.json"))
    all_json_paths = [str(path).replace("\\", "/") for path in all_json_paths]

    names_path_pair = dict()
    for item in all_json_paths:
        name = item.split(".")[-2].split("/")[-1]
        names_path_pair[name] = item
    return names_path_pair

def get_name_segnpz_pair_for_test(path):
    data_root = pathlib.Path(path)
    all_image_paths = list(data_root.glob("*.npz"))
    all_image_paths = [str(path).replace("\\", "/") for path in all_image_paths]

    names_path_pair = dict()
    for item in all_image_paths:
        name = item.split(".")[-2].split("/")[-1]
        names_path_pair[name] = item
    return names_path_pair

def convert_to_original_coord_system(seg_pred, json):
    orig_shape = json['orig_shp']
    # axis order is z x y
    brain_bbox_z = json['bbox_z']
    brain_bbox_x = json['bbox_x']
    brain_bbox_y = json['bbox_y']
    new_seg = np.zeros(orig_shape, dtype=np.uint8)
    tmp_z = np.min((orig_shape[0], brain_bbox_z[0] + seg_pred.shape[0]))
    tmp_x = np.min((orig_shape[1], brain_bbox_x[0] + seg_pred.shape[1]))
    tmp_y = np.min((orig_shape[2], brain_bbox_y[0] + seg_pred.shape[2]))
    new_seg[brain_bbox_z[0]:tmp_z, brain_bbox_x[0]:tmp_x, brain_bbox_y[0]:tmp_y] = seg_pred[:tmp_z -brain_bbox_z[0],
                                                                                   :tmp_x - brain_bbox_x[0],
                                                                                   :tmp_y - brain_bbox_y[0]]
    new_seg[new_seg==3]=4
    return new_seg

def get_test_data_list(experiment_result_root_path,include_train_flag=False):
    fold_excel_name = os.path.join(experiment_result_root_path, "KFold", "fold_" + str(1) + ".xlsx")

    with pd.ExcelFile(fold_excel_name) as reader:
        if include_train_flag:
            train_data_path_list = list(pd.read_excel(reader, sheet_name="train")["predict_root_path"].values)
            print("using train samples for test")
        else:
            train_data_path_list = []
        val_data_path_list = list(pd.read_excel(reader, sheet_name="val")["predict_root_path"].values)
        test_data_path_list=train_data_path_list+val_data_path_list
    print("num of sample of this test data: " + str(len(test_data_path_list)))
    return test_data_path_list

def get_all_data_list(experiment_result_root_path):
    """
    include train and val
    """
    fold_excel_name = os.path.join(experiment_result_root_path, "KFold", "fold_" + str(1) + ".xlsx")
    with pd.ExcelFile(fold_excel_name) as reader:

        val_data_path_list = list(pd.read_excel(reader, sheet_name="val")["predict_root_path"].values)
    print("num of sample of this test data: " + str(len(train_data_path_list+val_data_path_list)))
    return train_data_path_list+val_data_path_list

def get_vi_data_list(experiment_result_root_path,all_mean_flag=False):
    ## all mean kfold
    if all_mean_flag:
        fold_excel_name = os.path.join(experiment_result_root_path, "KFold_AllMean", "fold_" + str(1) + ".xlsx")
        print("using all mean data")
    else:
        fold_excel_name = os.path.join(experiment_result_root_path, "KFold", "fold_" + str(1) + ".xlsx")
    with pd.ExcelFile(fold_excel_name) as reader:
        test_data_path_list = list(pd.read_excel(reader, sheet_name="val")["predict_root_path"].values)

    print("num of sample of this test data: " + str(len(test_data_path_list)))
    return test_data_path_list

def get_folder_data_list(path):
    data_root = pathlib.Path(path)
    all_image_paths = list(data_root.glob("*.npy"))
    all_image_paths = [str(path).replace("\\", "/") for path in all_image_paths]
    all_image_paths.sort()

    all_json_paths = list(data_root.glob("*.json"))
    all_json_paths = [str(path).replace("\\", "/") for path in all_json_paths]
    all_json_paths.sort()

    return  zip(all_image_paths,all_json_paths)

def norm_image(image):
    min=image.min()
    max=image.max()

    return (image-min)/(max-min)

def get_npz_paths(path):
    data_root = pathlib.Path(path)
    all_image_paths = list(data_root.glob("*.npz"))
    all_image_paths = [str(path).replace("\\", "/") for path in all_image_paths]

    return all_image_paths

