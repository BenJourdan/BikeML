import torch
import wandb
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt


class Scaffold(nn.Module):

    def __init__(self,**kwargs):
        super(Scaffold,self).__init__()
        
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
        return loss, outputs, labels


    @staticmethod
    def evaluate_batch(data, criterion, device, model):
        images_x, images_y, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        outputs = model(images_x, images_y)
        loss = criterion(torch.squeeze(outputs), torch.squeeze(labels))  # compute loss
        predicted = torch.squeeze(torch.round(outputs))
        accuracy = predicted.eq(torch.squeeze(labels).data).float().mean().cpu()
        return loss.cpu().data.numpy(), accuracy, outputs

    @staticmethod
    def visualize(data, outputs, epoch, number_of_figures, unNormalizer):
        pairs_per_figure = 4
        for j in range(number_of_figures):
            fig,ax = plt.subplots(4,3,figsize=(20,20))
            for i in range(pairs_per_figure):
                img_a = unNormalizer(data[0][pairs_per_figure*j+i]).T.cpu()
                img_b = unNormalizer(data[1][pairs_per_figure*j+i]).T.cpu()
                label = data[2][pairs_per_figure*j+i]
                output = outputs[pairs_per_figure*j+i].cpu()

                ax[i][0].imshow((img_a*255).type(torch.uint8))
                ax[i][1].imshow((img_b*255).type(torch.uint8))
                ax[i][2].annotate("Same Ad" if label==1.0 else "Different Ad",(0,0.5))
                ax[i][2].annotate(u"\U0001F604",(0.5,0.5),size=20) if abs(label-output)<=0.5 else ax[i][2].annotate(u"\U0001F62D",(0.5,0.5),size=20)
                
                ax[i][0].set_axis_off()
                ax[i][1].set_axis_off()
                ax[i][2].set_axis_off()
            fig.suptitle(f'Performance at epoch {epoch}: figure {j}', fontsize=20)
            # plt.tight_layout()
            wandb.log({f"examples_{j}":fig})
            plt.close(fig)
                    