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

class TripletDataSet(Dataset):
    def __init__(self):
        print("Triplet builder started")
        self.positive_margin = 10
        self.negative_margin = 30
        self.video_index = 0

        # read video directory
        self.path = "/home/ali/Representation-learning/videos"
        filenames = [p for p in os.listdir(self.path) if p[0] != '.']
        self.video_paths = [os.path.join(self.path, f) for f in filenames]
        self.video_count = len(self.video_paths)
        # count frames
        self.frames=[self.read_video(p) for p in self.video_paths]
        # self.frame_lengths = np.array([len(imageio.mimread(p)) for p in self.video_paths])
        # self.frame_lengths=[]
        # self.cum_length = np.cumsum(self.frame_lengths)
        # logging
        print("The number fo the videos:", self.video_count)
        print(" video pathes:")
        for i in range(self.video_count):
            print("%d. %s - %d frames" % (i, self.video_paths[i], self.frames[i].shape[0]))
        # self.frame_size = (640, 320)

    def read_video(self,video_path):
        cap=cv2.VideoCapture(video_path)
        length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        factor=100/length
        frames = torch.empty((100, 3, h, w),dtype=torch.uint8)
        for i in range(length):
            ret,frame=cap.read()
            if ret:
                frame=torch.from_numpy(frame)
                frame=frame.permute(2,0,1)
                index = int(i * factor)
                frames[index,...]=frame
        for i in range(100):
            image=frames[i].permute(1,2,0).numpy()
            cv2.imshow("1",image)
            cv2.waitKey(30)
        return frames

    def sample_anchor(self):
        return np.random.choice(np.arange(self.frame_lengths[self.video_index]))

    def sample_positive(self, anchor):
        min_ = max(0, anchor - self.positive_margin)
        max_ = min(100 - 1, anchor + self.positive_margin)
        return np.random.choice(np.arange(min_, max_))

    def sample_negative(self, anchor):
        end1 = max(0, anchor - self.negative_margin)
        range1 = np.arange(0, end1)
        start2 = min(100 - 1, anchor + self.negative_margin)
        range2 = np.arange(start2, 100)
        range = np.concatenate([range1, range2])
        return np.random.choice(range)

    def sample_triplet(self, frames):
        anchor_index = self.sample_anchor()
        # print("anchor index",anchor_index)
        anchor = frames[anchor_index]
        pos = frames[self.sample_positive(anchor_index)]
        neg = frames[self.sample_negative(anchor_index)]
        # print("took samples")
        anchor = torch.from_numpy(anchor).to(device)
        # print("anchor tensor")
        pos = torch.from_numpy(pos).to(device)
        # print("pos tensor")
        neg = torch.from_numpy(neg).to(device)
        # print("neg tensor")
        return (anchor, pos, neg)

    def collect_triplets(self, sample_size=200):
        triplets = torch.FloatTensor(sample_size, 3, 3, *(self.frame_size)).to(device)
        frames = self.get_video(self.video_index)
        for i in range(sample_size):
            # print(i, "collect triplets")
            anchor, pos, neg = self.sample_triplet(frames)
            # print("after sample triplet",anchor.shape)
            triplets.data[i, 0, :, :, :] = anchor.data
            triplets.data[i, 1, :, :, :] = pos.data
            triplets.data[i, 2, :, :, :] = neg.data
            # print("iteration %d " % i,triplets.shape)
        self.video_index = (self.video_index + 1) % self.video_count
        # print(triplets.shape, "collect triplets")
        return torch.utils.data.TensorDataset(triplets, torch.zeros(triplets.size()[0]))

a=TripletDataSet()