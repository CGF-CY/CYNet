from torchvision import models
from torch import nn
import timm
from model_utils import GRL
import torch
from torchsummary import summary
class Discriminator(nn.Module):
    def __init__(self, max_iter):
        super(Discriminator, self).__init__()
        self.ad_net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(4, 4)
        )
        self.grl_layer = GRL(max_iter)
        self.fc = nn.Linear(512, 3)

    def forward(self, feature):
        adversarial_out = self.grl_layer(feature)
        adversarial_out = self.ad_net(adversarial_out).reshape(adversarial_out.shape[0], -1)
        adversarial_out = self.fc(adversarial_out)
        return adversarial_out
class Swin_T(nn.Module):
    def __init__(self,max_iter=4000):
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
        self.dis = Discriminator(max_iter)


    def forward(self,x):
        x=self.model(x)
        x=self.fc(x)
        return x


model = timm.create_model('swin_base_patch4_window7_224', pretrained=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(model[1])

# 创建一个随机输入张量，用于获取模型的输入大小
input_size = (3, 224, 224)  # 根据模型期望的输入尺寸设置
input_tensor = torch.randn(1, *input_size).to(device)

# 使用 torchsummary 打印模型信息
summary(model, input_size=input_size)

