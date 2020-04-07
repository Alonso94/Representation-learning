import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("GPU is working:",torch.cuda.is_available())

class DON_torchvision(nn.Module):
    def __init__(self,descriptor_dimensions):
        super().__init__()
        self.resnet=models.resnet34(pretrained=True)
        self.conv1=self.resnet.conv1
        self.bn1=self.resnet.bn1
        self.relu=self.resnet.relu
        self.maxpool=self.resnet.maxpool
        self.layer1=self.resnet.layer1
        self.layer2=self.resnet.layer2
        self.layer3=self.resnet.layer3
        self.layer4=self.resnet.layer4
        self.fc=nn.Linear(512,descriptor_dimensions)

    def forward(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.fc(x)
        return x

don=DON_torchvision(256).to(device)
print(don)
