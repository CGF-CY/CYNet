from torch.utils.data import Dataset
import os
import cv2


class Load_CASIA(Dataset):
    def __init__(self,root_dir,train=True,transforms=None,target_size=(224,224),UUID=-1):
        super(Load_CASIA, self).__init__()
        self.root_dir=root_dir
        self.train=train
        self.CASIA_path=os.path.join(self.root_dir,'CASIA-SURF')
        self.img_paths=[]
        self.labels=[]
        self.target_size=target_size
        self.UUID=UUID
        self.transforms=transforms
        if train:
            with open(os.path.join(self.CASIA_path,'color_train_list.txt'), 'r', encoding='utf-8') as file:
                for line in file:
                    self.img_paths.append(os.path.join(self.CASIA_path,line.strip().split(' ')[0]))
                    self.labels.append(int(line.strip().split(' ')[1]))


        else:
            with open(os.path.join(self.CASIA_path,'color_test_list.txt'), 'r', encoding='utf-8') as file:
                for line in file:
                    self.img_paths.append(os.path.join(self.CASIA_path,line.strip().split(' ')[0]))
                    self.labels.append(int(line.strip().split(' ')[1]))



    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label,self.UUID

    def __len__(self):
        return len(self.image_paths)


