import torch
import os
import cv2
from torch.utils.data import Dataset
from  torchvision.transforms import transforms
class Load_MSU(Dataset):
    def __init__(self,root_dir=None,train=True,transforms=None,UUID=-1):
        super(Load_MSU, self).__init__()
        self.root_dir=os.path.join(root_dir,"MSU-MFSD")
        self.transforms=transforms
        self.UUID=-1
        self.labels=[]
        self.img_paths=[]
        self.UUID=UUID
        train_object = ["002", "003", "005", "006", "007", "008", "009", "011", "012", "021", "034", "053", "054",
                        "055"]
        for i in range(len(train_object)):
            train_object[i] = "client" + train_object[i]
        test_object = ["001", "013", "014", "023", "024", "026", "028", "029", "030", "032", "033", "035", "036", "037",
                       "039", "042", "048", "049", "050", "051"]
        for i in range(len(test_object)):
            test_object[i] = "client" + test_object[i]
        if train:
            object=train_object
        else:
            object=test_object
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('jpg'):
                    if file.split('_')[1] in object:
                        self.img_paths.append(os.path.join(root,file))
                        if "real" in file:
                            self.labels.append(1)
                        else:
                            self.labels.append(0)





    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        real_h, real_w, c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label, self.UUID

    def __len__(self):
        return len(self.img_paths)


# train_transform=transforms.Compose([transforms.ToTensor(),
#                                     transforms.Resize((224,224)),
#                                     transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
#                                     transforms.RandomVerticalFlip(),
#                                     transforms.RandomHorizontalFlip()
#                                     ])
#
# test=Load_MSU(root_dir="F:\\FAS",train=True,transforms=train_transform)
# image,label,UUID=test[1]
# print(image)
# print(label)
# print(UUID)