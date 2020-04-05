import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing
from torchvision import models

import numpy as np
import imageio
import os
import functools
from PIL import Image
from tqdm import trange
import matplotlib.pyplot as plt
import time
import visvis as vv
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class TCN(nn.Module):
    def __init__(self):
        print("building the model")
        super().__init__()
        self.inception=models.inception_v3(pretrained=True,progress=True)
        self.conv1=self.inception.Conv2d_1a_3x3
        self.conv2_1=self.inception.Conv2d_2a_3x3
        self.conv2_2=self.inception.Conv2d_2b_3x3
        self.conv3=self.inception.Conv2d_3b_1x1
        self.conv4=self.inception.Conv2d_4a_3x3
        self.mix5_1=self.inception.Mixed_5b
        self.mix5_2=self.inception.Mixed_5c
        self.mix5_3=self.inception.Mixed_5d
        self.conv6_1=nn.Conv2d(288,100,kernel_size=3,stride=1)
        self.batch_norm_1=nn.BatchNorm2d(100,eps=1e-3)
        self.conv6_2=nn.Conv2d(100,20,kernel_size=3,stride=1)
        self.batch_norm_2=nn.BatchNorm2d(20,eps=1e-3)
        # softmax2d is an activation in each channel
        # spatial softmax s_{ij}=\frac{exp(a_{ij})}{\sum_{i',j'} exp(a_{i'j'})}
        self.spatial_softmax=nn.Softmax2d()
        self.fc7=nn.Linear(6*6*32,32)
        self.alpha=10.0

    def normalize(self,x):
        norm_const=torch.pow(x,2).sum(1).add(1e-10)
        norm_const=norm_const.sqrt()
        output=torch.div(x,norm_const.view(-1,1).expand_as(x))
        return output

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2_1(x)
        x=self.conv2_2(x)
        x=F.max_pool2d(x,kernel_size=3,stride=2)
        x=self.conv3(x)
        x=self.conv4(x)
        x=F.max_pool2d(x,kernel_size=3,stride=2)
        x=self.mix5_1(x)
        x=self.mix5_2(x)
        x=self.mix5_3(x)
        x=self.conv6_1(x)
        x=self.batch_norm_1(x)
        x=self.conv6_2(x)
        x=self.batch_norm_2(x)
        x=self.spatial_softmax(x)
        x=self.fc7(x.view(x.size()[0],-1))
        return self.normalize(x)*self.alpha

class TCN_trainer:
    def __init__(self):
        print("TCN trainer started")
        self.num_epochs=10000
        self.minibatch_size=250
        self.learning_rate=0.01
        self.triplets_from_video=5
        self.iterate_over_triplets=5
        self.margin=10

        self.model=TCN().to(device)
        self.triplate_builder=TripletBuilder()
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=self.learning_rate)
        self.lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[100,200,1000],gamma=0.1)

        self.queue=multiprocessing.Queue(1)
        self.dataset_builder_process=multiprocessing.Process(target=self.build_set,args=(self.queue,))#,daemon=True)
        self.dataset_builder_process.start()

    def build_set(self,queue=None):
        print("\n start process")
        while True:
            datasets=[]
            for i in range(self.triplets_from_video):
                print(i, "build set")
                dataset=self.triplate_builder.collect_triplets()
                datasets.append(dataset)
            dataset=torch.utils.Data.ConcatDataset(datasets)
            print('Created {0} triplets'.format(len(dataset)))
            queue.put(dataset)

    def distance(self,x,y):
        diff=torch.abs(x-y)
        return torch.pow(diff,2).sum(1)

    def run(self):
        epochs_progress=trange(self.num_epochs)
        xx, yy=[], []
        for epoch in epochs_progress:
            print("enter the loop")
            dataset=self.queue.get()
            print("dataset ...")
            dataloader=torch.utils.Data.DataLoader(dataset,self.minibatch_size,shuffle=True,pin_memory=device)
            print("dataloaded..")
            x=0
            iteration_range=trange(self.iterate_over_triplets)
            for _ in iteration_range:
                losses=[]
                # no labels
                for minibatch,_ in dataloader:
                    x+=1
                    frames=torch.autograd.Variable(minibatch)
                    frames=frames.to(device)
                    # inputs
                    anchor=frames[:,0,:,:,:]
                    pos=frames[:,1,:,:,:]
                    neg=frames[:,2,:,:,:]
                    # outputs
                    anchor_output=self.model(anchor)
                    pos_output=self.model(pos)
                    neg_output=self.model(neg)
                    # loss computation
                    d_pos=self.distance(anchor_output,pos_output)
                    d_neg=self.distance(anchor_output,neg_output)
                    loss=torch.clamp(self.margin+d_pos-d_neg,min=0.0).mean()
                    losses.append(loss)
                    # optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                self.lr_scheduler.step()
                # plot
                xx.append(x)
                yy.append(np.mean(losses))
                plt.xlabel("timestep")
                plt.ylabel("loss")
                plt.plot(xx,yy)
                plt.show()

tcn=TCN_trainer()
# tcn.build_set()
# time.sleep(100)
tcn.run()