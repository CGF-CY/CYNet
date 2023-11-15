import torch
import random
from torch.nn import functional as F
from configs import parse
from networks.Transformer import Transformer
import os
import torch.nn as nn
import numpy as np
from torchvision import transforms
from datasets.Load_CelebA import CelebAData
from torch.utils.tensorboard import SummaryWriter
class model_utils():
    def __init__(self,data):
        super(model_utils, self).__init__()
        self.args=parse().parse_args()
        train_transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.Data = data
    def cartoonlize_images(self,images):
        model = Transformer()
        model.load_state_dict(torch.load(os.path.join("./pretrained_model", self.args.style + '_net_G_float.pth')))
        model.eval()
        if self.args.gpu > -1:
            model.cuda()

        else:
            model.float()
        outputs = model(images)
        return outputs

    def cut_paste(self,images):
        cut_images = []
        for image in images:
            cut_image = self.cut_paste_single(image)
            cut_images.append(cut_image)
        return torch.stack(cut_images)
    def cut_paste_single(self, image1, max_patch_size=(100, 100)):
        # cut and paste image


        random_index = random.randint(0, len(self.Data) - 1)
        image_true, label = self.Data[random_index]
        image2 = image_true
        assert image1.shape == image2.shape
        _, height, width = image1.shape

        patch_height = random.randint(1, min(height, max_patch_size[0]) + 1)
        patch_width = random.randint(1, min(width, max_patch_size[1]) + 1)

        top = random.randint(0, height - patch_height + 1)
        left = random.randint(0, width - patch_width + 1)
        patch = image2[:, top:top + patch_height, left:left + patch_width]
        result = image1.clone()
        result[:, top:top + patch_height, left:left + patch_width] = patch
        return result
class GRL(nn.Module):

    def __init__(self, max_iter):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput











