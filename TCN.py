import torch
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

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception=models.inception_v3(pretrained=True)
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

class TripletBuilder:
    def __init__(self):
        self.positive_margin=10
        self.negative_margin=30
        self.video_index=0

        # read video directory
        self.path="/home/ali/Representation-learning/video"
        filenames=[p for p in os.listdir(self.path) if p[0]!='.']
        self.video_paths=[os.path.join(self.path,f) for f in filenames]
        self.video_count=len(self.video_paths)

        # count frames
        self.frame_lengths=np.array([len(imageio.read(p)) for p in self.video_paths])
        self.cum_length=np.cumsum(self.frame_lengths)

        self.frame_size=(640,480)

    # Decorator to wrap a function with a memorizing callable that saves up to the maxsize most recent calls.
    # It can save time when an expensive or I/O bound function is periodically called with the same arguments.
    @functools.lru_cache(maxsize=1)
    def get_video(self,index):
        video_path=self.video_paths[index]
        video=imageio.read(video_path)
        frames=np.zeros((len(video),3,*self.frame_size))
        i=0
        for frame in video:
            image=Image.fromarray(frame)
            image=image.resize(self.frame_size)
            scaled=np.array(image,dtype=np.float32)/255
            tr_img=np.transpose(scaled,[2,0,1])
            frames[i,:,:,:]=tr_img
            i+=1
        return frames

    def sample_anchor(self):
        return np.random.choice(np.arange(0,self.frame_lengths[self.video_index]))

    def sample_positive(self,anchor):
        min_=max(0,anchor-self.positive_margin)
        max_=min(self.frame_lengths[self.video_index]-1,anchor+self.positive_margin)
        return np.random.choice(np.arange(min_,max_))

    def sample_negative(self,anchor):
        end1=max(0,anchor-self.negative_margin)
        range1=np.arange(0,end1)
        start2=min(self.frame_lengths[self.video_index]-1,anchor+self.negative_margin)
        range2=np.arange(start2,self.frame_lengths[self.video_index])
        range=np.concatenate([range1,range2])
        return np.random.choice(range)

    def sample_triplet(self,frames):
        anchor=frames[self.sample_anchor()]
        pos=frames[self.sample_positive(anchor)]
        neg=frames[self.sample_negative(anchor)]
        return anchor,pos,neg

    def build_set(self,sample_size=200):
        triplets=torch.Tensor(sample_size,3,3,*(self.frame_size))
        for i in range(sample_size):
            frames=self.get_video(self.video_index)
            anchor,pos,neg=self.sample_triplet(frames)
            triplets[i,0,:,:,:]=anchor
            triplets[i,1,:,:,:]=pos
            triplets[i,2,:,:,:]=neg
        self.video_index=(self.video_index+1)%self.video_count
        return triplets

class TCP_trainer:
    def __init__(self):
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
        self.dataset_builder_process=multiprocessing.Process(target=)
        self.dataset_builder_process.start()

    def build_set(self):
        while 1:
            datasets=[]
            for i in range(self.triplets_from_video):
                dataset=self.triplate_builder.build_set()
                datasets.append(dataset)
            datasets=torch.utils.Data.ConcatDataset(datasets)
            self.queue.put(datasets)

    def distance(self,x,y):
        diff=torch.abs(x-y)
        return torch.pow(diff,2).sum(1)

    def run(self):
        epochs_progress=trange(self.num_epochs)
        for epoch in epochs_progress:
            self.lr_scheduler.step()
            dataset=self.queue.get()
            dataloader=torch.utils.Data.DataLoader(dataset,self.minibatch_size,shuffle=True,pin_memory=device)
            for _ in range(self.iterate_over_triplets):
                losses=[]
                # no labels
                for minibatch,_ in dataloader:
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

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()