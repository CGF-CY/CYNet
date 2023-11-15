import torch
import timm
import torch.nn as nn
from configs import parse
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.Load_CelebA import CelebAData
from utils.performance import performance
class Swin_T(nn.Module):
    def __init__(self):
        super(Swin_T, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x=self.model(x)
        x=self.fc(x)
        return x

train_transforms=transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
)

args=parse().parse_args()
train_data=CelebAData(root_dir=args.data_dir,train=True,transforms=train_transforms)
test_data=CelebAData(root_dir=args.data_dir,train=False,transforms=train_transforms)
train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True)
test_loader=DataLoader(test_data,batch_size=args.batch_size,shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_ids=args.gpu_ids
model=Swin_T()
if device=="cuda" and len(args.gpu_ids) >= 2:
    model.to(device)
    model = nn.DataParallel(model, device_ids=gpu_ids)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
bce_loss=nn.BCELoss()
loss_bce_count_list = []
steps=20
performance=performance()
for step in range(steps):
    loss_bce_count = 0
    num=0
    ctr=0
    print(f'step:{step}')
    for data in train_loader:
        if ctr==5:
            break
        imgs,labels=data
        labels = labels.float()
        predictes=model(imgs)
        predictes=predictes.squeeze()

        loss_bce = bce_loss(predictes, labels)
        optimizer.zero_grad()
        loss_bce.backward()
        optimizer.step()
        loss_bce_count += loss_bce
        num+=1

    loss_bce_count = (loss_bce_count / num).detach().numpy()
    loss_bce_count_list.append(loss_bce_count)

performance.loss_pictrue(protocol="swin_t", loss_bec_count_list=loss_bce_count_list)
model.eval()
predictes_test = []
labels_test = []

for data in test_loader:
    imgs,labels=data
    predictes=model(imgs)
    predictes_test.append(predictes.detach().cpu())
    labels_test.append(labels.detach().cpu())
predictes_test=torch.cat(predictes_test,dim=0)
predictes_test=predictes_test.squeeze()
labels_test=torch.cat(labels_test,dim=0)
# get apcer and bpcer
best_threshold = performance.get_threshold(predictes_test, labels_test)
APCER,BPCER,ACER=performance.get_acper_bcper(predictes_test,labels_test,best_threshold)
print(f'Swin_T:\n        APCER:{APCER}    BPCER:{BPCER}   ACER:{ACER}  ',flush=True)
