import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt


class Baseline(nn.Module):

    def __init__(self,**kwargs):
        super(Baseline,self).__init__()
        
    @staticmethod
    def train_compute_acc(outputs):
        predicted = torch.squeeze(torch.round(outputs[0]))
        accuracy = predicted.eq(torch.squeeze(outputs[1])).float().mean() 
        return accuracy

    @staticmethod
    def train_batch(data, criterion, device, model):
        images_x, images_y, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        outputs = model(images_x, images_y)
        loss = criterion(torch.squeeze(outputs), torch.squeeze(labels))
        return loss

    @staticmethod
    def evaluate_batch(data, criterion, device, model):
        images_x, images_y, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        outputs = model(images_x, images_y)
        loss = criterion(torch.squeeze(outputs[0]), torch.squeeze(outputs[1]), labels)  # compute loss
        predicted = torch.squeeze(torch.round(outputs))
        accuracy = predicted.eq(torch.squeeze(labels).data).float().mean().cpu()
        return loss.cpu().data.numpy(), accuracy, outputs

    @staticmethod
    def visualize(data, outputs, epoch, unNormalizer, figures_to_store):
        fig,ax = plt.subplots(4,3,figsize=(20,20))
        for i in range(figures_to_store):
            img_a = unNormalizer(data[0][i]).T
            img_b = unNormalizer(data[1][i]).T
            label = data[2][i]

            ax[i][0].imshow(img_a)
            ax[i][1].imshow(img_b)
            ax[i][2].annotate("Same Ad" if label[i]==1.0 else "Different Ad",(0,0.5))
            ax[i][2].annotate(u"\U0001F604",(0.5,0.5),size=20) if abs(label[i]-outputs[i])<=0.5 else ax[i][2].annotate(u"\U0001F62D",(0.5,0.5),size=20)
            
            ax[i][0].set_axis_off()
            ax[i][1].set_axis_off()
            ax[i][2].set_axis_off()
        fig.suptitle(f'Performance at epoch {epoch}', fontsize=20)
        plt.tight_layout()
        return plt