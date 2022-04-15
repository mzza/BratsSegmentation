from torch.utils.data import Dataset
import numpy as np

class BrainTumorDataSet(Dataset):
    def __init__(self, image_path_list, transform=None):
        super(BrainTumorDataSet, self).__init__()
        self.image_path_list = image_path_list
        self.transform = transform

    def __getitem__(self, index):
        npy_name = self.image_path_list[index]
        data = np.load(npy_name, mmap_mode="r")
        sample = {"image": data[:4], "label": data[4]}  # data：[4,x,y,z] label:[x,y,z]

        if self.transform != None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.image_path_list)
