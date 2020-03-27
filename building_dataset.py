import os
import numpy as np
import pandas as pd
import imageio
from skimage import io, transform
import cv2
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
from tqdm import trange

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("GPU is working:",torch.cuda.is_available())

class TripletDataSet(Dataset):
    def __init__(self,iterations=100):
        print("Triplet builder started")
        self.ref_length = 100
        self.positive_margin = 10
        self.negative_margin = 20
        self.video_index = 0

        self.num_channels=3
        self.height=640
        self.width=320

        # read video directory
        self.path = "/home/ali/Representation-learning/videos"
        filenames = [p for p in os.listdir(self.path) if p[0] != '.']
        self.video_paths = [os.path.join(self.path, f) for f in filenames]
        self.video_count = len(self.video_paths)
        # count frames
        self.frames=[self.read_video(p) for p in self.video_paths]
        self.framers=torch.stack(self.frames,dim=0)
        # logging
        print("The number fo the videos:", self.video_count)
        print(" video pathes:")
        for i in range(self.video_count):
            print("%d. %s - %d frames" % (i, self.video_paths[i], self.frames[i].shape[0]))

        self.triplets=self.collect_triplets()

    def __len__(self):
        return self.triplets.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx=np.mod(idx,self.triplets.shape[0])
        sample=self.triplets[idx]
        sample={'anchor':self.triplets[idx][0],
                'positive':self.triplets[idx][1],
                'negative':self.triplets[idx][2]}
        return sample

    def show_sample(self,idx):
        sample=self.__getitem__(idx)
        anchor = sample['anchor'].permute(1, 2, 0).numpy()
        pos = sample['positive'].permute(1, 2, 0).numpy()
        neg = sample['negative'].permute(1, 2, 0).numpy()
        image=np.hstack([anchor,pos,neg])
        cv2.imshow("anchor, pos, neg", image)
        cv2.waitKey(0)

    def read_video(self,video_path, show_video=False):
        cap=cv2.VideoCapture(video_path)
        length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        factor=self.ref_length/length
        frames = torch.empty((self.ref_length, self.num_channels, self.height, self.width),dtype=torch.uint8)
        for i in range(length):
            ret,frame=cap.read()
            if ret:
                frame=torch.from_numpy(frame)
                frame=frame.permute(2,0,1)
                index = int(i * factor)
                frames[index,...]=frame
        cap.release()
        if show_video:
            for i in range(self.ref_length):
                image=frames[i].permute(1,2,0).numpy()
                cv2.imshow("Video after",image)
                cv2.waitKey(30)
        return frames

    def sample_anchor(self):
        return np.random.choice(np.arange(self.ref_length))

    def sample_positive(self, anchor):
        min_ = max(0, anchor - self.positive_margin)
        max_ = min(self.ref_length - 1, anchor + self.positive_margin)
        return np.random.choice(np.arange(min_, max_))

    def sample_negative(self, anchor):
        end1 = max(0, anchor - self.negative_margin)
        range1 = np.arange(0, end1)
        start2 = min(self.ref_length - 1, anchor + self.negative_margin)
        range2 = np.arange(start2, self.ref_length)
        range = np.concatenate([range1, range2])
        return np.random.choice(range)

    def collect_triplets(self,iterations=30):
        # multi-video
        anchor_index = self.sample_anchor()
        pos_index=self.sample_positive(anchor_index)
        neg_index=self.sample_negative(anchor_index)
        num_triplets=(self.video_count**2)*(2*self.video_count-1)
        triplets=torch.empty(num_triplets*iterations,3,self.num_channels,self.height,self.width,dtype=torch.uint8)
        index=0
        for _ in trange(iterations):
            for i in range(self.video_count):
                for j in range(self.video_count):
                    for k in range(self.video_count):
                        anchor = self.frames[i][anchor_index]
                        pos = self.frames[j][pos_index]
                        neg = self.frames[k][neg_index]
                        triplets[index]=torch.stack([anchor,pos,neg])
                        index += 1
                        if j!=i:
                            pos = self.frames[j][anchor_index]
                            neg = self.frames[k][neg_index]
                            triplets[index] = torch.stack([anchor, pos, neg])
                            index += 1
        return triplets

a=TripletDataSet()
for i in range(10):
    idx=np.random.randint(0,30)
    a.show_sample(idx)
