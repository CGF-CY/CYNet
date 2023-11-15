from torch.utils.data import Dataset
import imgaug.augmenters as iaa
import os
import cv2
import torch
from torchvision import transforms

class Load_OULU(Dataset):
    def __init__(self,root_dir=None,train=True,protocol='not protocol',transforms=None,only_true=False,UUID=-1):
        super(Load_OULU, self).__init__()
        self.root_dir=os.path.join(root_dir,"OULU")
        self.protocol=protocol
        self.image_paths=[]
        self.labels=[]
        self.transforms=transforms
        self.UUID=UUID
        if protocol=='not protocol':
            if train:
                for root, dirs, files in os.walk(self.root_dir):
                    for file in files:
                        path=os.path.join(root, file)
                        if file.endswith('.png') and "train" in file and path not in self.image_paths:
                            self.image_paths.append(path)
                            if "+1" in path:
                                self.labels.append(1)
                            else:
                                self.labels.append(0)

        else:
            if train:
                dir_path=self.list_all_files(os.path.join(self.root_dir,protocol,'train'))
            else:
                dir_path = self.list_all_files(os.path.join(self.root_dir, protocol, 'test'))

            for root,dirs,files in os.walk(dir_path):
                for file in files:
                    path=os.path.join(root,file)
                    if(file.endswith(".png")):
                        self.image_paths.append(path)
                        if "+1" in path:
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
        return image, label,self.UUID

    def __len__(self):
        return len(self.image_paths)
    def list_all_files(self,directory):
        dir=[]
        for root, dirs, files in os.walk(directory):
            for file in files:
                dir.append(os.path.join(root, file))
        return dir




