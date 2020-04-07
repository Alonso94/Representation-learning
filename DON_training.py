from DON_model import DON
from building_dataset import TripletDataSet
import torch

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("GPU is working:",torch.cuda.is_available())

class DON_trainer:
    def __init__(self):
        pass