"""
重新组织代码架构
1.因为不需要在train data上面的数据
2. 直接预测在val上面的结果
3. 直接保存为nii文件，不要npy文件
4. softmax文件还是保存，已被不时之需
"""
## predict the segmentation results
import json
import os
import torch

from utils import compute_steps_for_sliding_window, get_gaussian, pad_nd_image, \
    to_cuda, get_device, to_tensor, get_name_json_pair_for_test, convert_to_original_coord_system, get_test_data_list, \
    get_folder_data_list
from ParametersSetting import output_root_path,patch_size,num_class,step_size,test_version_name,test_npy_path,hp,train_npy_path
from Training import BrainTumorSeg
import numpy as np
import SimpleITK as sitk



def predict_on_testdata(test_generator):
    for item in test_generator:
        filename = item[0].split("/")[-1].split(".")[0]
        print("predicting", filename)
        load_data = np.load(item[0], mmap_mode="r")
        result=predict_single_sample(load_data)
        print("predict end")
        #
        # print("saving prob")
        # np.save(os.predict_root_path.join(softmax_dir, filename), result[0].astype(np.float16))
        #
        # print("save npy..")
        # np.save(os.predict_root_path.join(softmax_dir, filename), result[1].astype(np.uint8))

        print("saving nii...")
        with open(item[1], 'r')as fp:
            json_data = json.load(fp)[0]
        seg = convert_to_original_coord_system(seg_pred=result[1], json=json_data)
        sitk_img = sitk.GetImageFromArray(seg)
        sitk_img.SetSpacing(json_data['spacing'])
        sitk_img.SetDirection(json_data['direction'])
        sitk_img.SetOrigin(json_data['origin'])
        sitk.WriteImage(sitk_img, os.path.join(nii_dir, filename + ".nii.gz"))


        print("saving finish")





def predict_single_sample(image):
    torch.cuda.empty_cache()
    #print("get step...")
    data, slicer = pad_nd_image(image, new_shape=patch_size, return_slicer=True)
    steps = compute_steps_for_sliding_window(patch_size, data.shape[1:], step_size)
    aggregated_results = torch.zeros([num_class] + list(data.shape[1:]), dtype=torch.float32, device=get_device())
    aggregated_nb_of_predictions = torch.zeros([num_class] + list(data.shape[1:]), dtype=torch.float32,
                                               device=get_device())
    #print("star titling....")
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

                predicted_patch = patch_predict(
                    data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], (0, 1, 2))

                aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds
                #print("finishing one patch...")
    # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
    slicer = tuple(
        [slice(0, aggregated_results.shape[i]) for i in
         range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
    aggregated_results = aggregated_results[slicer]
    aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]
    # computing the class_probabilities by dividing the aggregated result with result_numsamples
    class_probabilities = (aggregated_results / aggregated_nb_of_predictions).detach().cpu().numpy()
    predicted_segmentation = class_probabilities.argmax(0)

    return class_probabilities,predicted_segmentation



def patch_predict(feed_data, mirror_axes):
    feed_data=to_tensor(feed_data)
    result_torch = torch.zeros([1, num_class] + list(feed_data.shape[2:]),
                               dtype=torch.float)
    with torch.no_grad():
        if torch.cuda.is_available():
            feed_data = to_cuda(feed_data)
            result_torch = result_torch.cuda(get_device(), non_blocking=True)

        mirror_idx = 8
        num_results = 2 ** len(mirror_axes)

        for m in range(mirror_idx):
            if m == 0:
                pred = torch.softmax(loaded_model(feed_data)[-1],dim=1)
                result_torch += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = torch.softmax(loaded_model(torch.flip(feed_data, (4,)))[-1],dim=1)
                result_torch += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = torch.softmax(loaded_model(torch.flip(feed_data, (3,)))[-1],dim=1)
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = torch.softmax(loaded_model(torch.flip(feed_data, (4, 3)))[-1],dim=1)
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = torch.softmax(loaded_model(torch.flip(feed_data, (2,)))[-1],dim=1)
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = torch.softmax(loaded_model(torch.flip(feed_data, (4, 2)))[-1],dim=1)
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = torch.softmax(loaded_model(torch.flip(feed_data, (3, 2)))[-1],dim=1)
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = torch.softmax(loaded_model(torch.flip(feed_data, (4, 3, 2)))[-1],dim=1)
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

            result_torch[:, :] *= gaussian_importance_map

    return torch.squeeze(result_torch)


def run():
    global  softmax_dir, nii_dir,seg_dir, loaded_model, gaussian_importance_map, add_for_nb_of_preds
    output_folder = os.path.join(output_root_path, test_version_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # make sure the test data is preprocessed
    test_data_generator = get_folder_data_list(test_npy_path)
    # softmax_dir = os.predict_root_path.join(output_folder, "softmax")
    # if not os.predict_root_path.exists(softmax_dir):
    #     os.makedirs(softmax_dir)
    nii_dir = os.path.join(output_folder, "nii")
    if not os.path.exists(nii_dir):
        os.makedirs(nii_dir)
    # seg_dir = os.predict_root_path.join(output_folder, "seg")
    # if not os.predict_root_path.exists(seg_dir):
    #     os.makedirs(seg_dir)
    print("loading model...")
    best_model = os.path.join(output_folder, "checkpoint", "epoch=276.ckpt")
    if torch.cuda.is_available():
        print("reload on gpu...")
        wm = lambda storage, loc: storage.cuda(0)
        loaded_model = BrainTumorSeg.load_from_checkpoint(checkpoint_path=best_model, map_location=wm, hp=hp).cuda()
    else:
        print("reload on cpu")
        wm = lambda storage, loc: storage
        loaded_model = BrainTumorSeg.load_from_checkpoint(checkpoint_path=best_model, map_location=wm, hp=hp)

    print("get gaussian...")
    gaussian_importance_map = get_gaussian(patch_size, sigma_scale=1. / 8)
    gaussian_importance_map = to_tensor(gaussian_importance_map)
    if torch.cuda.is_available():
        gaussian_importance_map = to_cuda(gaussian_importance_map)
    add_for_nb_of_preds = gaussian_importance_map
    print("start predict...")
    predict_on_testdata(test_data_generator)

if __name__ == '__main__':
    run()

