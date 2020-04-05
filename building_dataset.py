import os
import numpy as np
import pandas as pd
import imageio
from skimage import io, transform
import cv2
import functools
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
from tqdm import trange

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print("GPU is working:",torch.cuda.is_available())

class TripletDataSet(Dataset):
    def __init__(self,load=True,to_file=True,iterations=100):
        print("Triplet builder started")
        self.load=load
        self.to_file=to_file
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
        # logging
        print("The number fo the videos:", self.video_count)
        print(" video pathes:")
        for i in range(self.video_count):
            print("%d. %s" % (i, self.video_paths[i]))

        # collect triplets
        self.frames = [self.read_video(p) for p in self.video_paths]
        self.framers = torch.stack(self.frames, dim=0)
        self.index = 0
        if self.to_file:
            self.triples_path = "/home/ali/Representation-learning/triplets/"
            if not load:
                shutil.rmtree("/home/ali/Representation-learning/triplets/")
                os.makedirs("/home/ali/Representation-learning/triplets/")
                self.collect_triplets_to_folder(iterations)
            else:
                self.index=len([p for p in os.listdir(self.triples_path) if p[0]!='.']) // 3
        else:
            self.triplets=self.collect_triplets_single_view(iterations)


    def __len__(self):
        return self.index

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx=np.mod(idx,self.__len__())
        if self.to_file:
            anchor,pos,neg=self.read_triplet(idx)
            # sample={'anchor':anchor,
            #         'positive':pos,
            #         'negative':neg}
            sample=torch.stack([anchor,pos,neg])
        else:
            # sample = {'anchor': self.triplets[idx][0],
            #           'positive': self.triplets[idx][1],
            #           'negative': self.triplets[idx][2]}
            sample=self.triplets[idx]
        return sample

    def show_sample(self,idx):
        sample=self.__getitem__(idx)
        anchor = sample[0].permute(1, 2, 0).numpy()
        pos = sample[0].permute(1, 2, 0).numpy()
        neg = sample[0].permute(1, 2, 0).numpy()
        image=np.hstack([anchor,pos,neg])
        cv2.imshow("anchor, pos, neg", image)
        cv2.waitKey(0)

    def read_video(self,video_path, show_video=False):
        cap=cv2.VideoCapture(video_path)
        length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        factor=self.ref_length/length
        frames = torch.empty((self.ref_length, self.num_channels, self.height//2, self.width//2),dtype=torch.uint8)
        for i in range(length):
            ret,frame=cap.read()
            frame=cv2.resize(frame,(self.width//2,self.height//2))
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

    def collect_triplets(self,iterations=100):
        # multi-video
        num_triplets=(self.video_count**2)*(2*self.video_count-1)
        triplets=torch.empty(num_triplets*iterations,3,self.num_channels,self.height//2,self.width//2,dtype=torch.uint8)
        for _ in trange(iterations):
            anchor_index = self.sample_anchor()
            pos_index = self.sample_positive(anchor_index)
            neg_index = self.sample_negative(anchor_index)
            for i in range(self.video_count):
                for j in range(self.video_count):
                    for k in range(self.video_count):
                        anchor = self.frames[i][anchor_index]
                        pos = self.frames[j][pos_index]
                        neg = self.frames[k][neg_index]
                        triplets[self.index]=torch.stack([anchor,pos,neg])
                        self.index += 1
                        if j!=i:
                            pos = self.frames[j][anchor_index]
                            neg = self.frames[k][neg_index]
                            triplets[self.index] = torch.stack([anchor, pos, neg])
                            self.index += 1
        return triplets

    def write_triplet_to_files(self,i,anchor,pos,neg):
        anchor_path=self.triples_path+"%d_%s"%(self.index,"anchor.png")
        anchor=anchor.permute(1, 2, 0).numpy()
        cv2.imwrite(anchor_path,anchor)
        pos_path = self.triples_path + "%d_%s" % (self.index, "pos.png")
        pos=pos.permute(1, 2, 0).numpy()
        cv2.imwrite(pos_path,pos)
        neg_path = self.triples_path + "%d_%s" % (self.index, "neg.png")
        neg=neg.permute(1, 2, 0).numpy()
        cv2.imwrite(neg_path,neg)

    def read_triplet(self,i):
        anchor_path = self.triples_path + "%d_%s" % (i, "anchor.png")
        anchor=cv2.imread(anchor_path,1)
        anchor=torch.from_numpy(anchor).permute(2,0,1)
        pos_path = self.triples_path + "%d_%s" % (i, "pos.png")
        pos=cv2.imread(pos_path,1)
        pos=torch.from_numpy(pos).permute(2,0,1)
        neg_path = self.triples_path + "%d_%s" % (i, "neg.png")
        neg=cv2.imread(neg_path,1)
        neg=torch.from_numpy(neg).permute(2,0,1)
        return anchor,pos,neg

    def collect_triplets_to_folder(self,iterations=100):
        # multi-video
        for _ in trange(iterations):
            anchor_index = self.sample_anchor()
            pos_index = self.sample_positive(anchor_index)
            neg_index = self.sample_negative(anchor_index)
            for i in range(self.video_count):
                for j in range(self.video_count):
                    for k in range(self.video_count):
                        anchor = self.frames[i][anchor_index]
                        pos = self.frames[j][pos_index]
                        neg = self.frames[k][neg_index]
                        self.write_triplet_to_files(i,anchor,pos,neg)
                        self.index += 1
                        if j != i:
                            pos = self.frames[j][anchor_index]
                            neg = self.frames[k][neg_index]
                            self.write_triplet_to_files(i,anchor,pos,neg)
                            self.index += 1

    def collect_triplets_single_view(self,iterations=500):
        triplets = torch.empty(self.video_count * iterations, 3, self.num_channels, self.height, self.width,dtype=torch.uint8)
        index=0
        for _ in trange(iterations):
            anchor_index = self.sample_anchor()
            pos_index = self.sample_positive(anchor_index)
            neg_index = self.sample_negative(anchor_index)
            for k in range(self.video_count):
                anchor = self.frames[k][anchor_index]
                pos = self.frames[k][pos_index]
                neg = self.frames[k][neg_index]
                triplets[index]=torch.stack([anchor,pos,neg])
                index += 1
        return triplets

# a=TripletDataSet(load=False)
# print(len(a))
# for i in range(10):
#     idx=np.random.randint(0,len(a))
#     a.show_sample(idx)
