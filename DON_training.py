from DON_model import DON
from building_dataset import TripletDataSet
from tqdm import trange

import torch
import torch.optim as optim
from torch.autograd import Variable

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("GPU is working:",torch.cuda.is_available())

class DON_trainer:
    def __init__(self,load=True):
        self.descriptor_dims=256
        self.model=DON(self.descriptor_dims).to(device)
        self.load_from = "/home/ali/Representation-learning/models/DON.pth"
        self.save_to = "/home/ali/Representation-learning/models/DON.pth"
        if load:
            self.model.load_state_dict(torch.load(self.load_from, map_location=device))
            self.model.eval()
        self.optimizer=optim.Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4)
        self.pixelwise_contrastive_loss=None
        self.max_iter=3500
        self.dataset=TripletDataSet()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, 8, shuffle=True, pin_memory=device)
    
    def get_loss(self,anchor,pos,neg):
        loss=torch.tensor(0.0)
        return loss

    def run_training(self):
        for epoch in trange(50):
            for i,data in enumerate(self.dataloader):
                anchor,pos,neg= data

                anchor_out=self.model(anchor)
                pos_out=self.model(pos)
                neg_out=self.model(neg)

                self.optimizer.zero_grad()
                loss=self.get_loss(anchor_out,pos_out,neg_out)
                loss.backward()
                self.optimizer.step()
            torch.save(self.model.state_dict(), self.save_to)