import numpy as np
import math
from building_dataset import TripletDataSet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print("GPU is working:",torch.cuda.is_available())

class DON_torchvision(nn.Module):
    def __init__(self,descriptor_dims):
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
        self.fc=nn.Conv2d(self.resnet.inplanes, descriptor_dims, 1)
        self.fc1 = nn.Linear(descriptor_dims * 160 * 320, 32)

    def forward(self,x):
        input_dims=x.size()[2:]
        # 3 x 160 x 320
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        # 64 x 40 x 80
        x=self.layer1(x)
        # 64 x 40 x 80
        x=self.layer2(x)
        # 128 x 20 x 40
        x=self.layer3(x)
        # 256 x 10 x 20
        x=self.layer4(x)
        # 512 x 5 x 10
        x=self.fc(x)
        # 256 x 5 x 10
        x=F.interpolate(x,size=input_dims,mode='bilinear',align_corners=True)
        # 256 x 160 x 320
        y=x.view(x.size()[0],-1)
        y = self.fc1(y)
        return x,y

# model=DON_torchvision(256).to(device)
# dataset=TripletDataSet()
# dataloader = torch.utils.data.DataLoader(dataset, 8, shuffle=True, pin_memory=device)
# for minibatch in dataloader:
#     frames = torch.autograd.Variable(minibatch)
#     frames = frames.to(device).float()
#     # inputs
#     anchor = frames[:, 0, :, :, :]
#     pos = frames[:, 1, :, :, :]
#     neg = frames[:, 2, :, :, :]
#     # outputs
#     anchor_output = model(anchor)
#     pos_output = model(pos)
#     neg_output = model(neg)
