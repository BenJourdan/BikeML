import torch
import torch.nn as nn

from EmbeddingNetwork import EmbeddingNetwork

import os,sys
sys.path.append("/scratch/GIT/BikeML")
from baseline.baseline import baseline

class BaselineModel_1b(nn.Module):
    def __init__(self, input_shape):
        super(BaselineModel_1b, self).__init__()
        
        self.input_shape = input_shape
        self.components = nn.ModuleDict() 

        backbone = EmbeddingNetwork(self.input_shape, mlp_layers=3, embedding_dimension=128)
        self.components["backbone"] = backbone

    @staticmethod
    def train_batch(data, criterion, device, model):
        images_x, images_y, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        outputs = model(images_x, images_y)
        loss = criterion(torch.squeeze(outputs[0]), torch.squeeze(outputs[1]), torch.squeeze(labels))
        return loss

    @staticmethod
    def evaluate_batch(data, criterion, device, model):
        images_x, images_y, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        outputs = model(images_x, images_y)
        loss = criterion(input=torch.squeeze(outputs), target=torch.squeeze(labels))  # compute loss
        accuracy = None
        return loss.cpu().data.numpy(), accuracy, outputs

    @staticmethod
    def visualize(data,i):
        pass

    def forward(self,image_a,image_b):
        out_x = self.components["backbone"](image_a)
        out_y = self.components["backbone"](image_b)
        
        return out_x, out_y

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.components.values():
            try:
                item.reset_parameters()
            except:
                pass

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BaselineModel_1b((200,3,512,512),3)
    
    model.to(device)

    for value in model.components.values():
        print(value)
    input()