import numpy as np
import math
import cv2
from building_dataset import TripletDataSet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_URL={'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'}

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print("GPU is working:",torch.cuda.is_available())

# Based on https://github.com/warmspringwinds/vision/blob/eb6c13d3972662c55e752ce7a376ab26a1546fb5/torchvision/models/resnet.py

def conv3x3(in_planes,out_planes,stride=1,dilation=1):
    kernel_size=np.asarray((3,3))
    upsampled_kernel_size=(kernel_size-1)*(dilation-1)+kernel_size
    full_padding=(upsampled_kernel_size-1)//2
    # Conv2d doesn't accept numpy arrays
    full_padding,kernel_size=tuple(full_padding),tuple(kernel_size)
    conv=nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,stride=stride,
                   padding=full_padding,bias=False,dilation=dilation)
    return conv

class Block(nn.Module):
    def __init__(self,in_planes,planes,stride=1,downsample=None,dilation=1):
        super().__init__()
        self.conv1=conv3x3(in_planes,planes,stride=stride,dilation=dilation)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(planes,planes,dilation=dilation)
        self.bn2=nn.BatchNorm2d(planes)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x
        y=self.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        if self.downsample is not None:
            residual=self.downsample(x)
        y+=residual
        y=self.relu(y)
        return y

class dialated_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[3,4,6,3]
        self.multigrid=(1,1,1)
        self.in_planes=64
        self.current_stride=4
        self.current_dilation=1
        self.output_stride=8
        # pretrained Resnet34 - pretrained on ImageNet
        self.num_classes=1000
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self.layer(64,layers[0])
        self.layer2=self.layer(128,layers[1],stride=2)
        self.layer3 = self.layer(256, layers[2], stride=2)
        self.layer4 = self.layer(512, layers[3], stride=2)
        self.fc=nn.Linear(512,self.num_classes)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def layer(self,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or self.in_planes!=planes:
            if self.current_stride==self.output_stride:
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:
                self.current_stride = self.current_stride * stride
            downsample=nn.Sequential(
                nn.Conv2d(self.in_planes,planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes)
            )
        layers=[]
        # first layer with downsample
        dilation= self.current_dilation
        layers.append(Block(self.in_planes,planes,stride,downsample=downsample,dilation=dilation))
        self.in_planes=planes
        for i in range(1,blocks):
            dilation = self.current_dilation
            layers.append(Block(self.in_planes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self,x,adjust=False):
        # 3 x 160 x 320
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        # 64 x 40 x 80
        x=self.layer1(x)
        # 64 x 40 x 80
        x=self.layer2(x)
        # 128 x 20 x 40
        x=self.layer3(x)
        # 256 x 20 x 40
        x=self.layer4(x)
        # 512 x 20 x 40
        x=self.fc(x)
        # 256 x 20 x 40
        return x

class DON(nn.Module):
    def __init__(self,descriptor_dims):
        super().__init__()
        self.resnet=dialated_resnet()
        self.resnet.load_state_dict(model_zoo.load_url(model_URL['resnet34']))

        self.resnet.fc=nn.Conv2d(self.resnet.in_planes, descriptor_dims, 1)
        self.resnet.fc.weight.data.normal_(0, 0.01)
        self.resnet.fc.bias.data.zero_()

    def forward(self,x):
        input_dims = x.size()[2:]
        x=self.resnet(x)
        # 256 x 20 x 40
        x=F.interpolate(x,size=input_dims,mode='bilinear',align_corners=True)
        # 256 x 160 x 320
        return x

    def process_output(self,x):
        # [N,descriptor_dims,h,w] -> [N,w*h,descriptor_dim]
        N,dim,w,h=x.size()
        x=x.view(N,dim,w*h)
        x=x.permute(0,2,1)
        return x

# model=DON(256).to(device)
# dataset=TripletDataSet()
# dataloader = torch.utils.data.DataLoader(dataset, 4, shuffle=True, pin_memory=device)
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