import torch
import torch.nn as nn
import torchvision.models as models
from math import floor
import matplotlib.pyplot as plt
from torch.nn import functional as F

import os,sys
sys.path.append("/scratch/GIT/BikeML")
from baseline.Scaffold import Scaffold
import wandb
import scikitplot as skplt
import numpy as np
import matplotlib.pyplot as plt

class ModelSnipper(nn.Module):
    def __init__(self, original_model,snip=1):
        super(ModelSnipper, self).__init__()
        self.features = torch.nn.Sequential(*list(original_model.children())[:-snip])

    def forward(self,x):
        x = self.features(x)
        return x

class BaselineModel_1a(Scaffold):
    def __init__(self, input_shape=None, mlp_layers=None, **kwargs):
        super(BaselineModel_1a, self).__init__()
        
        self.input_shape = input_shape
        self.mlp_layers = mlp_layers
        self.components = nn.ModuleDict() 

        self.build_network()


    def build_network(self):
        x = torch.zeros((self.input_shape))
        y = torch.zeros((self.input_shape))
        # Setup ResNet feature extractor and loss
        resnet = models.resnet18(pretrained=True)
        backbone  = ModelSnipper(resnet,snip=1)
        if self.train_backbone == False:
            for param in backbone.parameters():
                param.requires_grad = False
        self.components["embedding"] = backbone
        #should be "embedding"

        # Dims: batch no.(1) x image dims(3)
        out_x = backbone(x)
        out_y = backbone(y)

        xy = torch.cat((torch.flatten(out_x, start_dim=1, end_dim=-1),torch.flatten(out_y, start_dim=1, end_dim=-1)),dim=1)

        for i in range(self.mlp_layers):
            # print(xy.shape[1],xy.shape[1]/2)
            self.components['fcc_{}'.format(i)] = nn.Linear(in_features=xy.shape[1],
                       out_features=floor(xy.shape[1]/2))
            xy = self.components['fcc_{}'.format(i)](xy)

            self.components['bn_{}'.format(i)] = nn.BatchNorm1d(xy.shape[1])
            xy = self.components['bn_{}'.format(i)](xy)
            #relu goes here

        self.components["final_layer"] = nn.Linear(in_features=xy.shape[1],out_features=1)
        # print(xy.shape[1],1)

    def forward(self,image_a,image_b):
        out_x = self.components["embedding"](image_a)
        out_y = self.components["embedding"](image_b)
        
        out_xy = torch.cat((torch.flatten(out_x, start_dim=1, end_dim=-1),torch.flatten(out_y, start_dim=1, end_dim=-1)),dim=1)
        
        for i in range(self.mlp_layers):
            out_xy = self.components['fcc_{}'.format(i)](out_xy)
            out_xy = self.components['bn_{}'.format(i)](out_xy)
            out_xy = F.leaky_relu(out_xy, negative_slope=0.01)

        out_xy = torch.sigmoid(self.components["final_layer"](out_xy))
        
        return out_xy

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.components.values():
            try:
                item.reset_parameters()
            except:
                pass

    def track_metrics(self,outputs,epoch,step,criterion,loss,split="train"):
        
        if split=="train":
            accuracy = ((outputs[0].round() == outputs[1]).sum()/outputs[1].shape[0])*100
        else:
            accuracy = ((outputs[0][0].round() == outputs[1]).sum()/outputs[1].shape[0])*100
        wandb.log({"epoch": epoch, "{}_loss".format(split): float(loss),"global_step":epoch})
        wandb.log({"epoch": epoch, "{}_accuracy".format(split): float(accuracy),"global_step":epoch})




    def track_extra_metrics(self,list_of_outputs, epoch,split):
        


        predictions = list_of_outputs[0][0].numpy()
        labels = list_of_outputs[1].numpy()
        
        fig,ax = plt.subplots(1,1,figsize=(7,7))

        skplt.metrics.plot_roc_curve(labels.flatten(),np.hstack((1-predictions,predictions)),ax=ax,curves=("each_class"),text_fontsize="large")
        ax.set_title("Test Set ROC Curve")
        leg = ax.legend()
        leg.remove()
        ax.grid(axis="y",which="major")
        wandb.log({f"ROC curve on {split} set":fig})
        fig.savefig(f"roc_{split}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BaselineModel_1a((200,3,512,512),3)
    
    model.to(device)

    for value in model.components.values():
        print(value)
    input()











