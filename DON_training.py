from DON_model import DON
from DON_model_torchvision import DON_torchvision
from building_dataset import TripletDataSet
from tqdm import trange
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("GPU is working:",torch.cuda.is_available())

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin=0.5

    def distance(self, x, y):
        diff = torch.abs(x - y)
        diff=torch.pow(diff, 2).sum(-1)
        return diff

    def forward(self,anchor,pos,neg):
        pos_distance=self.distance(anchor,pos)
        neg_distance=self.distance(anchor,neg)
        loss=torch.clamp(self.margin+pos_distance-neg_distance,min=0.0).mean()
        return loss

class DON_trainer:
    def __init__(self,load=True):
        self.writer=SummaryWriter('runs/DON_1')
        self.descriptor_dims=3
        self.model=DON_torchvision(self.descriptor_dims).to(device)
        self.criterion=TripletLoss()
        self.load_from = "/home/ali/Representation-learning/models/DON.pth"
        self.save_to = "/home/ali/Representation-learning/models/DON.pth"
        if load:
            self.model.load_state_dict(torch.load(self.load_from, map_location=device))
            self.model.eval()
        self.optimizer=optim.Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4)
        self.pixelwise_contrastive_loss=None
        self.max_iter=3500
        self.dataset=TripletDataSet()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, 4, shuffle=True, pin_memory=device)
        self.images = self.dataset.frames[0].float()
        self.embedding=torch.empty((len(self.images),32))

    def run_training(self,iterations=20):
        global i
        running_loss=0.0

        for epoch in trange(iterations):
            for i,data in enumerate(self.dataloader):
                step=i+len(self.dataloader)*epoch
                frames = Variable(data)
                frames = frames.to(device).float()
                # inputs
                anchor = frames[:, 0, :, :, :]
                pos = frames[:, 1, :, :, :]
                neg = frames[:, 2, :, :, :]
                # outputs
                anchor_out,anchor_embedding=self.model(anchor)
                pos_out,pos_embedding=self.model(pos)
                neg_out,neg_embedding=self.model(neg)
                self.writer.add_image("original vs. prediction", torchvision.utils.make_grid([anchor[0],anchor_out[0]]),step)
                # loss computation
                loss=self.criterion(anchor_embedding,pos_embedding,neg_embedding)
                # optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # plot loss
                running_loss+=loss.item()
                self.writer.add_scalar("training_loss",running_loss,step)
            for i in range(len(self.images)):
                img=self.images[i].unsqueeze(0).to(device)
                with torch.no_grad():
                    output,self.embedding[i]=self.model(img)
            self.writer.add_embedding(self.embedding,global_step=epoch)
            torch.save(self.model.state_dict(), self.save_to)

don=DON_trainer(load=True)
don.run_training()