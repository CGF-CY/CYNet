from configs import parse
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.Load_CelebA import CelebAData
from utils.performance import performance
import torch
import torch.nn as nn
from networks.S_T_model import CYmodel
import torch.optim as optim
from  loss import cosine_similarity_loss
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
true_data=CelebAData(root_dir=args.data_dir,train=True,transforms=train_transforms,only_true=True)
train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True)
test_loader=DataLoader(test_data,batch_size=args.batch_size,shuffle=True)
performace=performance()
gpu_ids = [0, 1]
device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
model_ST = CYmodel(true_data)
model_ST.to(device)
step = 0
optimizer = optim.SGD(model_ST.parameters(), lr=args.base_lr)
#begin train
bce_loss = nn.BCELoss()
loss_bec_count_list = []
loss_text_count_list = []
loss_difference_count_list = []
loss_all_count_list=[]
for step in range(5):
    loss_bec_count = 0
    loss_text_count = 0
    loss_difference_count = 0
    loss_all_count = 0
    num = 0

    for data in train_loader:
        if num==5:
            break
        imgs, labels = data
        labels = labels.float()
        teacher_feature, study_style_eature, study_cutpaste_feature, predictes = model_ST(imgs)
        predictes = predictes.squeeze(1)
        loss_bec = bce_loss(predictes, labels)
        loss_text = cosine_similarity_loss(teacher_feature, study_style_eature)
        loss_difference = cosine_similarity_loss(teacher_feature, study_cutpaste_feature)

        loss_all = 0.2 * loss_text + 0.2 * loss_difference + 0.6 * loss_bec

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        loss_bec_count += loss_bec
        loss_text_count += loss_text
        loss_difference_count += loss_difference
        loss_all_count += loss_all
    #loss
    loss_bec_count = (loss_bec_count / num).detach().numpy()
    loss_bec_count_list.append(loss_bec_count)
    loss_text_count = (loss_text_count / num).detach().numpy()
    loss_text_count_list.append(loss_text_count)
    loss_difference_count = (loss_difference_count / num).detach().numpy()
    loss_difference_count_list.append(loss_difference_count)
    loss_all_count = (loss_all_count / num).detach().numpy()
    loss_all_count_list.append(loss_all_count)
    performace.loss_pictrue(protocol="CelebA_re",loss_bec_count_list=loss_bec_count_list,
                 loss_text_count_list=loss_text_count_list,
                 loss_difference_count_list=loss_difference_count_list,
                 loss_all_count_list=loss_all_count_list)

# end train
model_ST.eval()



#eval
predictes_test = []
labels_test = []
i=0
for data in test_loader:
    if i==5:
        break
    i+=1
    imgs,labels=data
    _,_,_,predictes=model_ST(imgs)
    predictes_test.append(predictes.detach().cpu())
    labels_test.append(labels.detach().cpu())
predictes_test=torch.cat(predictes_test,dim=0)
predictes_test=predictes_test.squeeze()
labels_test=torch.cat(labels_test,dim=0)

#get apcer and bpcer
best_threshold=performace.get_threshold(predictes_test,labels_test)


APCER,BPCER,ACER=performace.get_acper_bcper(predictes_test,labels_test,best_threshold)
print(f'CelebA_re:\n        APCER:{APCER}    BPCER:{BPCER}   ACER:{ACER}  ')
performace.test_roc(predictes_test,labels_test)