import torch
from batchgenerators.augmentations.spatial_transformations import augment_spatial_2, augment_mirroring
from batchgenerators.augmentations.color_augmentations import augment_brightness_additive, augment_contrast
from batchgenerators.augmentations.utils import resize_image_by_padding_batched, random_crop_3D_image_batched
import numpy as np


class SpacialTransform(object):
    def __init__(self, patch_size=(128, 128, 128)):
        self.patch_size = patch_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = np.expand_dims(image, axis=0)
        label = label[np.newaxis, np.newaxis]
        image, label = augment_spatial_2(data=image, seg=label,
                                         patch_size=self.patch_size,
                                         patch_center_dist_from_border=list(np.array(self.patch_size) // 2),
                                         do_elastic_deform=True, deformation_scale=(0, 0.2),
                                         do_rotation=True, angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                                         angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                                         angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                                         do_scale=True, scale=(0.9, 1.1),
                                         random_crop=True,
                                         p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1)
        sample["image"], sample["label"] = np.squeeze(image), np.squeeze(label)
        return sample


class MirrorTransform(object):
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        label = np.expand_dims(label, axis=0)
        image, label = augment_mirroring(sample_data=image, sample_seg=label)
        sample["image"], sample["label"] = image, np.squeeze(label)
        return sample


class BrightnessTransform(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        image = sample["image"]
        image = augment_brightness_additive(data_sample=image,
                                            mu=0, sigma=self.sigma,
                                            per_channel=True)
        sample["image"] = image

        return sample


class ContrastTransform(object):
    def __init__(self, range):
        self.range = range

    def __call__(self, sample):
        image = sample["image"]
        image = augment_contrast(data_sample=image,
                                 contrast_range=self.range,
                                 preserve_range=False, per_channel=True)
        sample["image"] = image

        return sample


class RandomCrop(object):
    def __init__(self, data, path_size):
        super(RandomCrop, self).__init__()
        self.data = np.expand_dims(data, axis=0)
        self.patch_size = path_size

    def __call__(self, *args, **kwargs):
        if np.any(np.array(self.data.shape[2:]) - np.array(self.patch_size) < 0):
            new_shape = np.max(np.vstack([np.array(self.data.shape[2:]), np.array(self.patch_size)]), 0)
            self.data = resize_image_by_padding_batched(self.data, new_shape, 0)
        crop_data = random_crop_3D_image_batched(self.data, self.patch_size)
        crop_data = np.squeeze(crop_data)

        return crop_data


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': torch.from_numpy(image).to(torch.float32),
                'label': torch.from_numpy(label).to(torch.long)}


class PaddingTransform(object):
    def __init__(self, new_shape, mode="constant"):
        self.mode = mode
        self.new_shape = new_shape

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        label = np.expand_dims(label, axis=0)
        data = np.concatenate([image, label], axis=0)
        kwargs = {'constant_values': 0}

        old_shape = np.array(data.shape[-len(self.new_shape):])

        num_axes_nopad = len(data.shape) - len(self.new_shape)

        new_shape = [max(self.new_shape[i], old_shape[i]) for i in range(len(self.new_shape))]

        if not isinstance(new_shape, np.ndarray):
            new_shape = np.array(new_shape)

        difference = new_shape - old_shape
        pad_below = difference // 2
        pad_above = difference // 2 + difference % 2
        pad_list = [[0, 0]] * num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

        res = np.pad(data, pad_list, self.mode, **kwargs)
        sample["image"] = res[:4]
        sample["label"] = res[4:]
        return sample
