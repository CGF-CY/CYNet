from torch.utils.data import Dataset
import imgaug.augmenters as iaa
import os
import cv2
import torch
from torchvision import transforms

class Load_OULU(Dataset):
    def __init__(self,root_dir=None,use='Train',protocol='not protocol',transforms=None,only_true=False,UUID=-1):
        super(Load_OULU, self).__init__()
        self.root_dir=root_dir
        self.protocol=protocol
        self.image_paths=[]
        self.labels=[]
        self.transforms=transforms
        dir_path=os.path.join(self.root_dir,protocol,use)
        paths=self.list_all_files(dir_path)
        if protocol=='not protocol':
            if train:

        else:
            if only_true:
                for i in range(len(paths)):
                    label=paths[i].split('\\')[-2]
                    if label=='+1':
                        self.labels.append(1)
                        self.image_paths.append(paths[i])
            else:
                self.image_paths=self.list_all_files(os.path.join(self.root_dir,protocol,use))
                for i in range(len(self.image_paths)):
                    label=self.image_paths[i].split('\\')[-2]
                    if label=='+1':
                        self.labels.append(1)
                    else:
                        self.labels.append(0)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        real_h, real_w, c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)
    def list_all_files(self,directory):
        dir=[]
        for root, dirs, files in os.walk(directory):
            for file in files:
                dir.append(os.path.join(root, file))
        return dir




