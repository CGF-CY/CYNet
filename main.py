import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets.Load_CelebA import CelebAData
from datasets.Load_OULU import Load_OULU
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse
from configs import parse
from networks.S_T_model import CYmodel
from  loss import cosine_similarity_loss
from utils.performance import performance
def train_model(protocol='Protocol_1',train_loader=None,dev_loader=None,test_loader=None):
    performace=performance()
    gpu_ids = [0, 1]
    device = torch.device(f'cuda:{gpu_ids[0]} ' if torch.cuda.is_available() else 'cpu')
    model_ST = CYmodel()
    model_ST.to(device)
    step = 0
    optimizer = optim.SGD(model_ST.parameters(), lr=args.base_lr)
    #begin train
    bce_loss = nn.BCELoss()
    loss_bec_count_list = []
    loss_text_count_list = []
    loss_difference_count_list = []
    loss_all_count_list=[]
    for step in range(20):
        loss_bec_count = 0
        loss_text_count = 0
        loss_difference_count = 0
        loss_all_count = 0
        num = 0

        for data in train_loader:
            imgs, labels = data
            labels = labels.float()
            teacher_feature, study_style_eature, study_cutpaste_feature, predictes = model_ST(imgs)
            print(predictes.shape)
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
        performace.loss_pictrue(protocol=protocol,loss_bec_count_list=loss_bec_count_list,
                     loss_text_count_list=loss_text_count_list,
                     loss_difference_count_list=loss_difference_count_list,
                     loss_all_count_list=loss_all_count_list)

    # end train
    model_ST.eval()



    #eval
    predictes_dev = []
    labels_dev = []
    for data in dev_loader:
        imgs,labels=data
        _,_,_,predictes=model_ST(imgs)
        predictes_dev.append(predictes.detach().cpu())
        labels_dev.append(labels.detach().cpu())
    predictes_dev=torch.cat(predictes_dev,dim=0)
    predictes_dev=predictes_dev.squeeze()
    labels_dev=torch.cat(labels_dev,dim=0)

    #get apcer and bpcer
    best_threshold=performace.get_threshold(predictes_dev,labels_dev)
    predictes_test = []
    labels_test = []
    for data in dev_loader:
        imgs,labels=data
        _,_,_,predictes=model_ST(imgs)
        predictes_test.append(predictes.detach().cpu())
        labels_test.append(labels.detach().cpu())

    predictes_test=torch.cat(predictes_test,dim=0)
    predictes_test=predictes_test.squeeze()
    labels_test=torch.cat(labels_dev,dim=0)
    APCER,BPCER,ACER=performace.get_acper_bcper(predictes_test,labels_test,best_threshold)
    print(f'{protocol}:\n        APCER:{APCER}    BPCER:{BPCER}   ACER:{ACER}  ')
    performace.test_roc(predictes_test,labels_test)





def main(args):
    print(1)
    #data
    train_transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip()
        ]

    )
    print(1)
    test_transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip()
        ]

    )
    print(1)
    train_CelebA_data=CelebAData(root_dir=args.data_dir,train=True,transforms=train_transform)
    train_CelebA_data_true=CelebAData(root_dir=args.data_dir,train=True,transforms=train_transform,only_true=True)
    #protocol_1
    # train_OULU_protocol_data = Load_OULU(root_dir=args.data_dir,protocol='Protocol_1' ,use='Train',transforms=train_transform)
    # dev_OULU_protocol_data = Load_OULU(root_dir=args.data_dir, protocol='Protocol_1', use='Dev',transforms=train_transform)
    # test_OULU_protocol_data = Load_OULU(root_dir=args.data_dir,protocol='Protocol_1', use='Test', transforms=train_transform)
    # train_OULU_protocol_loader = DataLoader(train_OULU_protocol_data, batch_size=args.batch_size, shuffle=True)
    # dev_OULU_protocol_loader = DataLoader(dev_OULU_protocol_data, batch_size=args.batch_size, shuffle=True)
    # test_OULU_protocol_loader=DataLoader(test_OULU_protocol_data,batch_size=args.batch_size,shuffle=True)
    # protocol='Protocol_1'
    # train_model(protocol=protocol, train_loader=train_OULU_protocol_loader, dev_loader=dev_OULU_protocol_loader)
    # #protocol_2
    # train_OULU_protocol_data = Load_OULU(root_dir=args.data_dir, protocol='Protocol_2', use='Train',
    #                                      transforms=train_transform)
    # dev_OULU_protocol_data = Load_OULU(root_dir=args.data_dir, protocol='Protocol_2', use='Dev',
    #                                    transforms=train_transform)
    # test_OULU_protocol_data = Load_OULU(root_dir=args.data_dir, protocol='Protocol_2', use='Test',
    #                                     transforms=train_transform)
    # train_OULU_protocol_loader = DataLoader(train_OULU_protocol_data, batch_size=args.batch_size, shuffle=True)
    # dev_OULU_protocol_loader = DataLoader(dev_OULU_protocol_data, batch_size=args.batch_size, shuffle=True)
    # test_OULU_protocol_loader = DataLoader(test_OULU_protocol_data, batch_size=args.batch_size, shuffle=True)
    # protocol = 'Protocol_2'
    # train_model(protocol=protocol, train_loader=train_OULU_protocol_loader, dev_loader=dev_OULU_protocol_loader)
    protocols = ['Protocol_3','Protocol_4']
    Trains=['Train_1','Train_2','Train_3','Train_4','Train_5','Train_6']
    Devs=['Dev_1','Dev_2','Dev_3','Dev_4','Dev_5','Dev_6']
    Tests=['Test_1','Test_2','Test_3','Test_4','Test_5','Test_6']
    for protocol in protocols:
        for i in range(7):
            print(f'-------------{protocol}_{i}-------------------')
            train_OULU_protocol_data = Load_OULU(root_dir=args.data_dir, protocol=protocol, use=Trains[i],
                                                 transforms=train_transform)
            dev_OULU_protocol_data = Load_OULU(root_dir=args.data_dir, protocol=protocol, use=Devs[i],
                                               transforms=train_transform)
            test_OULU_protocol_data = Load_OULU(root_dir=args.data_dir, protocol=protocol, use=Tests[i],
                                                transforms=train_transform)
            print(f'train length {len(train_OULU_protocol_data)}')
            print(f'dev length {len(dev_OULU_protocol_data)}')
            print(f'test length {len(test_OULU_protocol_data)}')
            # train_OULU_protocol_loader = DataLoader(train_OULU_protocol_data, batch_size=args.batch_size, shuffle=True)
            # dev_OULU_protocol_loader = DataLoader(dev_OULU_protocol_data, batch_size=args.batch_size, shuffle=True)
            # test_OULU_protocol_loader = DataLoader(test_OULU_protocol_data, batch_size=args.batch_size, shuffle=True)
            #train_model(protocol=protocol, train_loader=train_OULU_protocol_loader, dev_loader=dev_OULU_protocol_loader)

















if __name__=="__main__":
    print(1)
    args = parse().parse_args()
    main(args)
    print("OKkkk!!!!")









