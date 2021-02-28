from typing import OrderedDict
import torch
import numpy as np
from torchvision import utils
import torchvision.models as models
from math import floor
import torch.nn as nn
import torch.nn.functional as F

class ModelSnipper(nn.Module):
    def __init__(self, original_model, snip=2):
        super(ModelSnipper, self).__init__()
        self.features = torch.nn.Sequential(OrderedDict(list(original_model.named_children())[:-snip]))
    def forward(self, x):
        x = self.features(x)
        return x

class ModelSnipperViz(nn.Module):
    def __init__(self, original_model, snip=2):
        super(ModelSnipperViz, self).__init__()
        self.features = torch.nn.Sequential(OrderedDict(list(original_model.named_children())[:-snip]))

    def forward(self, x):
        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        g0 = self.features.layer1(x)
        g1 = self.features.layer2(g0)
        g2 = self.features.layer3(g1)
        g3 = self.features.layer4(g2)
        
        return [g.pow(2).mean(1) for g in (g0, g1, g2, g3)]

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_shape, mlp_layers, embedding_dimension,train_backbone=False,**kwargs):
            super(EmbeddingNetwork, self).__init__()
            
            self.input_shape = input_shape
            self.mlp_layers = mlp_layers
            self.embedding_dimension = embedding_dimension
            self.components = nn.ModuleDict() 
            self.train_backbone = train_backbone

            self.build_network()
        
    def build_network(self):
        x = torch.zeros((self.input_shape))
        # Setup ResNet feature extractor and loss
        resnet = models.resnet18(pretrained=True)
        # print(resnet)
        backbone = ModelSnipper(resnet, snip=1)

        if self.train_backbone== False:
            for param in backbone.parameters():
                param.requires_grad = False

        self.components["backbone"] = backbone

        x = backbone(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        shape_ratio = x.shape[1]/self.embedding_dimension
        layer_ratio = shape_ratio**(1/self.mlp_layers)
        print(shape_ratio)
        print(layer_ratio)
        for i in range(self.mlp_layers):
            print(i)
            print(x.shape)
            
            #enforce the final layer to have the correct embedding dimension
            if i == (self.mlp_layers - 1):
                self.components['fcc_{}'.format(i)] = nn.Linear(in_features=x.shape[1],
                    out_features=self.embedding_dimension)
            else:
                self.components['fcc_{}'.format(i)] = nn.Linear(in_features=x.shape[1],
                        out_features=floor(x.shape[1]/layer_ratio))
            
            x = self.components['fcc_{}'.format(i)](x)

            self.components['bn_{}'.format(i)] = nn.BatchNorm1d(x.shape[1])
            x = self.components['bn_{}'.format(i)](x)
        
    def forward(self,x):
        x = self.components["backbone"](x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        for i in range(self.mlp_layers):
            x = self.components['fcc_{}'.format(i)](x)
            x = self.components['bn_{}'.format(i)](x)
            x = F.leaky_relu(x, negative_slope=0.01)
        
        return x

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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmbeddingNetwork((200,3,512,512),3, 128)
    print(model)

    for value in model.components.values():
        print(value)
    input()
