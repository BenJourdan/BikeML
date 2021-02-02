import torch
import torch.nn as nn

from EmbeddingNetwork import EmbeddingNetwork

class BaselineModel_1c(nn.Module):
    def __init__(self, input_shape):
        super(BaselineModel_1c, self).__init__()
        
        self.input_shape = input_shape
        self.components = nn.ModuleDict() 

        backbone = EmbeddingNetwork(self.input_shape, mlp_layers=3, embedding_dimension=128)
        self.components["backbone"] = backbone

    @staticmethod
    def evaluate_batch(data, criterion, device, model):
        images_a, images_p, images_n, labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
        outputs = model(images_a, images_p, images_n)
        loss = criterion(input=torch.squeeze(outputs), target=torch.squeeze(labels))  # compute loss
        accuracy = None
        return loss.cpu().data.numpy(), accuracy, outputs

    @staticmethod
    def visualize(data,i):
        pass

    """
    image_a : anchor image
    image_p : positive image
    image_n : negative image
    """
    def forward(self,image_a,image_p,image_n):
        out_a = self.components["backbone"](image_a)
        out_p = self.components["backbone"](image_p)
        out_n = self.components["backbone"](image_n)
        
        return out_a, out_p, out_n

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
    
    model = BaselineModel_1c((200,3,512,512),3)
    
    model.to(device)

    for value in model.components.values():
        print(value)
    input()