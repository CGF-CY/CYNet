import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms
import random
from datasets.Load_CelebA import CelebAData
from configs import parse
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from networks.model_utils import model_utils
from datasets.Load_OULU import Load_OULU
from networks.Transformer import Transformer
import os


class GlobalSpatialAttention(nn.Module):
    def __init__(self, channel):
        super(GlobalSpatialAttention, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # squeeze: BxCxHxW => Bx1xHxW
        out = self.squeeze(x)
        # sigmoid activation
        out = self.sigmoid(out)
        # apply attention
        out = x * out
        return out


class CYmodel(nn.Module):
    def __init__(self, data):
        super(CYmodel, self).__init__()
        self.args = parse().parse_args()
        gpu_ids = [0, 1]
        self.device_1 = torch.device(f'cuda:{gpu_ids[1]}' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
        self.transformer = Transformer()
        self.transformer.load_state_dict(
            torch.load(os.path.join("./pretrained_model", self.args.style + '_net_G_float.pth')))
        self.transformer.eval()
        self.transformer = self.transformer.to(self.device_1)
        # Remove the fcin model
        self.teacher = nn.Sequential(*(list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]))
        # keep the structure of the teach network and the student network the same
        self.student = nn.Sequential(*(list(models.resnet50().children())[:-1]))
        # freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        # get the true iamge
        self.Data = data
        self.model_utils = model_utils(self.Data)
        # self.bn=nn.BatchNorm2d()
        self.attention_1 = GlobalSpatialAttention(256)
        self.attention_2 = GlobalSpatialAttention(512)
        self.attention_3 = GlobalSpatialAttention(1024)
        self.attention_4 = GlobalSpatialAttention(2048)
        self.bn = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100352, 14336),
            nn.Linear(14336, 1024),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)

    def cartoonlize_images(self, images):
        images = images.to(self.device_1)
        self.transformer = self.transformer.to(self.device_1)
        outputs = self.transformer(images)
        return outputs

    def forward(self, x):
        # stylize images
        x = x.to(self.device)
        style_x = self.cartoonlize_images(x)
        style_x = style_x.to(self.device)
        # cut and paste
        cut_x = self.model_utils.cut_paste(x)
        cut_x = cut_x.to(self.device)
        teacher_feature = self.get_feature_output(self.teacher, x)
        study_style_feature = self.get_feature_output(self.student, style_x)
        study_cutpaste_feature = self.get_feature_output(self.student, cut_x)
        study_feature = self.get_feature_output(self.student, x)
        a_1 = self.attention_1(study_feature[0])
        a_2 = self.attention_2(study_feature[1])
        a_3 = self.attention_3(study_feature[2])
        a_4 = self.attention_4(study_feature[3])
        c_1 = self.conv_1(a_1)
        c_2 = self.conv_2(a_2 + c_1)
        c_3 = self.conv_3(a_3 + c_2)
        c_4 = self.bn(c_3 + a_4)
        label = self.fc(c_4)

        return teacher_feature, study_style_feature, study_cutpaste_feature, label

    def get_feature_output(self, model, input):
        layer_head = model[0:4]
        layer1 = model[4]
        layer2 = model[5]
        layer3 = model[6]
        layer4 = model[7]
        input_haed = layer_head(input)
        layer4 = layer4.to(self.device)
        layer3 = layer3.to(self.device)
        layer2 = layer2.to(self.device)
        layer1 = layer1.to(self.device)
        input_layer1 = layer1(input_haed)
        input_layer2 = layer2(input_layer1)
        input_layer3 = layer3(input_layer2)
        input_layer4 = layer4(input_layer3)
        return (input_layer1, input_layer2, input_layer3, input_layer4)



