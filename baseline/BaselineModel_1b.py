import torch
import torch.nn as nn
import numpy as np
import wandb
from torchvision.models.resnet import ResNet, BasicBlock
from EmbeddingNetwork import EmbeddingNetwork

import os,sys
sys.path.append("/scratch/GIT/BikeML")
from baseline.Scaffold import Scaffold


class BaselineModel_1b(Scaffold):
    def __init__(self, input_shape,mlp_layers=3,embedding_dimension=128,**kwargs):
        super(BaselineModel_1b, self).__init__()
        
        self.input_shape = input_shape
        self.components = nn.ModuleDict() 

        self.components["embedding"] = EmbeddingNetwork(self.input_shape, mlp_layers=mlp_layers, embedding_dimension=embedding_dimension,**kwargs)

    @staticmethod
    def train_batch(data, criterion, device, model):

        images_x, images_y, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        outputs = model(images_x, images_y)

        vector_1 = outputs[0]
        vector_2 = outputs[1]

        loss = criterion(torch.squeeze(vector_1), torch.squeeze(vector_2), target=torch.squeeze(labels))
        return loss, [vector_1,vector_2], labels
    
    @staticmethod
    def evaluate_batch(data, criterion, device, model):
        images_x, images_y, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        outputs = model(images_x, images_y)
        vector_1 = outputs[0].cpu()
        vector_2 = outputs[1].cpu()
        loss = criterion(torch.squeeze(outputs[0]), torch.squeeze(outputs[1]), target=torch.squeeze(labels))  # compute loss
        accuracy = 0.0
        return loss.cpu().numpy(), accuracy, [vector_1,vector_2]

    # @staticmethod
    # def track_extra_metrics(outputs, epoch, split):
    #     labels = outputs[1].view(-1)
    #     embd_a = outputs[0][0]
    #     embd_b = outputs[0][1]

    #     embd_pos_a = embd_a[labels==1.0,:]
    #     embd_neg_a = embd_a[labels==0.0,:]
    #     embd_pos_b = embd_b[labels==1.0,:]
    #     embd_neg_b = embd_b[labels==0.0,:]

    #     pos_sq_dist = (embd_pos_a - embd_pos_b).pow(2).sum(1).mean()
    #     neg_sq_dist = (embd_neg_a - embd_neg_b).pow(2).sum(1).mean()

    #     wandb.log({"{}_avg_intra_ad_dist".format(split): pos_sq_dist,"global_step":epoch}) # intra Ad distance
    #     wandb.log({"{}_avg_inter_ad_dist".format(split): neg_sq_dist,"global_step":epoch}) # inter Ad distance
    #     wandb.log({"{}_inter/intra_ad_dist_ratio".format(split): neg_sq_dist/pos_sq_dist,"global_step":epoch})

    @staticmethod
    def visualize(data, outputs, epoch, number_of_figures, unNormalizer):
        pass

    @staticmethod
    def train_compute_acc(outputs):
        return 0

    def forward(self,image_a,image_b):
        out_x = self.components["embedding"](image_a)
        out_y = self.components["embedding"](image_b)
        
        return out_x, out_y

    @staticmethod
    def track_metrics(outputs,epoch,step,criterion,loss=None,split="train"):
        
        labels = outputs[1].view(-1)
        embd_a = outputs[0][0]
        embd_b = outputs[0][1]

        embd_pos_a = embd_a[labels==1.0,:]
        embd_neg_a = embd_a[labels==0.0,:]
        embd_pos_b = embd_b[labels==1.0,:]
        embd_neg_b = embd_b[labels==0.0,:]
        
        #works for euclidean distance
        # pos_sq_dist = (embd_pos_a - embd_pos_b).pow(2).sum(1).mean()
        # neg_sq_dist = (embd_neg_a - embd_neg_b).pow(2).sum(1).mean()
        pos_targets = torch.ones((embd_pos_a.shape[0]))
        neg_targets = torch.zeros((embd_neg_a.shape[0]))
        pos_sq_dist = criterion.embedding_dist(embd_pos_a,embd_pos_b)
        neg_sq_dist = criterion.embedding_dist(embd_neg_a,embd_neg_b)
        
        if split=="train":
            wandb.log({"epoch": epoch, "loss": float(loss)}, step=step)
            wandb.log({"{}_avg_intra_ad_dist".format(split): pos_sq_dist},step=step) # intra Ad distance
            wandb.log({"{}_avg_inter_ad_dist".format(split): neg_sq_dist},step=step) # inter Ad distance
            wandb.log({"{}_inter/intra_ad_dist_ratio".format(split): neg_sq_dist/pos_sq_dist},step=step)
        else:
            wandb.log({"epoch": epoch, "loss": float(loss),"global_step":epoch})
            wandb.log({"{}_avg_intra_ad_dist".format(split): pos_sq_dist,"global_step":epoch}) # intra Ad distance
            wandb.log({"{}_avg_inter_ad_dist".format(split): neg_sq_dist,"global_step":epoch}) # inter Ad distance
            wandb.log({"{}_inter/intra_ad_dist_ratio".format(split): neg_sq_dist/pos_sq_dist,"global_step":epoch})

        
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
