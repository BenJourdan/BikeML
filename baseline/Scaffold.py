import torch
import wandb
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.models.resnet import ResNet, BasicBlock,Bottleneck
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np

from EmbeddingNetwork import ModelSnipperViz

# Helper class for constructing visualisations of resnet outputs.
class ResNetAT(ResNet):
    """Attention maps of ResNet.
    
    Overloaded ResNet model to return attention maps.
    """
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        g0 = self.layer1(x)
        g1 = self.layer2(g0)
        g2 = self.layer3(g1)
        g3 = self.layer4(g2)
        
        return [g.pow(2).mean(1) for g in (g0, g1, g2, g3)]

class Scaffold(nn.Module):

    def __init__(self,train_backbone = False,**kwargs):
        super(Scaffold, self).__init__()
        self.train_backbone = train_backbone

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
        predicted = torch.squeeze(torch.round(outputs.detach().cpu()))
        accuracy = predicted.eq(torch.squeeze(labels).data).float().mean().cpu()
        return loss.cpu().data.numpy(), accuracy, outputs

    @staticmethod
    def track_metrics(outputs, epoch, split):
        pass

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
                    
    def am_viz(self, image_batch, filepaths):

        resnet = self.components["embedding"].components["backbone"]
        features = self.components["embedding"].components["backbone"].features
        blocks_per_layer = [len(features.layer1), len(features.layer2), len(features.layer3), len(features.layer4)]

        model = ResNetAT(Bottleneck if any([isinstance(m, Bottleneck) for m in features.modules()]) else BasicBlock, blocks_per_layer)
        model = ModelSnipperViz(model,snip=2)
        model.load_state_dict(resnet.state_dict())

        model.eval()
        with torch.no_grad():
            outputs = []
            for input_image in image_batch:
                outputs.append(model(input_image.unsqueeze(0)))

            for j,(attention_maps,filepath) in enumerate(zip(outputs,filepaths)):
                fig = plt.figure(constrained_layout=True,figsize=(20,20))
                spec = gridspec.GridSpec(ncols=4, nrows=6, figure=fig)

                ax_original = fig.add_subplot(spec[0:2, 0:2])
                ax_original.set_title("Original")
                ax_output_1 = fig.add_subplot(spec[0, 2])
                ax_output_2 = fig.add_subplot(spec[0, 3])
                ax_output_3 = fig.add_subplot(spec[1, 2])
                ax_output_4 = fig.add_subplot(spec[1, 3])
                ax_combined_1 = fig.add_subplot(spec[2:4,0:2])
                ax_combined_2 = fig.add_subplot(spec[2:4, 2:4])
                ax_combined_3 = fig.add_subplot(spec[4:,0:2])
                ax_combined_4 = fig.add_subplot(spec[4:,2:4])
                
                output_axes = [ax_output_1,ax_output_2,ax_output_3,ax_output_4]
                combined_axes = [ax_combined_1,ax_combined_2,ax_combined_3,ax_combined_4]

                #load image from disk
                im = np.array(Image.open(filepath).convert("RGB"))
                ax_original.imshow(im)

                for ax in output_axes +combined_axes + [ax_original]:
                    ax.axis("off")
                for i, attention_map in enumerate(attention_maps):
                    shape = im.shape
                    activations = np.array(Image.fromarray((attention_map[0].numpy()-attention_map[0].numpy().min())*255*(1/(attention_map[0].numpy().max()-attention_map[0].numpy().min()))).resize((shape[:2][1], shape[:2][0])).convert("L"))
                    output_axes[i].imshow(activations)
                    output_axes[i].set_title('Layer '+str(i))
                    combined_axes[i].imshow(im)
                    combined_axes[i].imshow(activations,alpha=0.75)
                    combined_axes[i].set_title('Layer '+str(i))
                wandb.log({f"examples_{j}":fig})
                plt.close(fig)