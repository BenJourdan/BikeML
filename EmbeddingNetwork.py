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
        self.features = torch.nn.Sequential(*list(original_model.children())[:-snip])

    def forward(self, x):
        x = self.features(x)
        return x

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_shape, mlp_layers, embedding_dimension):
            super(EmbeddingNetwork, self).__init__()
            
            self.input_shape = input_shape
            self.mlp_layers = mlp_layers
            self.embedding_dimension = embedding_dimension
            self.components = nn.ModuleDict() 

            self.build_network()
        
    def build_network(self):
        x = torch.zeros((self.input_shape))
        # Setup ResNet feature extractor and loss
        resnet = models.resnet18(pretrained=True)
        backbone  = ModelSnipper(resnet, snip=2)
        print(backbone)
        for param in backbone.parameters():
            param.requires_grad = False
        self.components["backbone"] = backbone

        x = backbone(x)
        print(x.shape)

        shape_ratio = x.shape[1]/self.embedding_dimension
        layer_ratio = shape_ratio**(1/self.mlp_layers)
        for i in range(self.mlp_layers):
            self.components['fcc_{}'.format(i)] = nn.Linear(in_features=x.shape[1],
                       out_features=floor(x.shape[1]/layer_ratio))
            x = self.components['fcc_{}'.format(i)](x)

            self.components['bn_{}'.format(i)] = nn.BatchNorm1d(x.shape[1])
            x = self.components['bn_{}'.format(i)](x)
        
    def forward(self,x):
        x = self.components["backbone"](x)
                
        for i in range(self.mlp_layers):
            x = self.components['fcc_{}'.format(i)](x)
            x = self.components['bn_{}'.format(i)](x)
            x = F.leaky_relu(out_xy, negative_slope=0.01)
        
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmbeddingNetwork((200,3,512,512),3, 128)
    
    model.to(device)

    for value in model.components.values():
        print(value)
    input()