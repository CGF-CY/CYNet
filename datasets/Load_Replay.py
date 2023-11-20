import os
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms
class Load_Replay(Dataset):
    def __init__(self,root_dir,train=True,transforms=None,target_size=(224,224),UUID=-1):
        super(Load_Replay, self).__init__()
        self.root_dir = root_dir
        self.train = train
        self.Replay_path = os.path.join(self.root_dir, 'Replay')
        self.img_paths = []
        self.labels = []
        self.target_size = target_size
        self.UUID = UUID
        self.transforms = transforms
        if train:
            self.root_dir=os.path.join(self.Replay_path,'train')
        else:
            self.root_dir = os.path.join(self.Replay_path, 'test')
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.png'):
                    if 'attack' in root:
                        self.labels.append(0)
                    else:
                        self.labels.append(1)
                    self.img_paths.append(os.path.join(root, file))
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
        return len(self.img_paths)


