from PIL import Image
import requests
import numpy as np
from io import BytesIO
import torch
from torch import nn
from torchvision.models import resnet18,resnet34,resnet50,resnet101
from torchvision.models.resnet import ResNet, BasicBlock,Bottleneck
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


if __name__ == "__main__":
    base_resnet = resnet101(pretrained=True)

    print(len(base_resnet.layer1))
    print(len(base_resnet.layer2))
    print(len(base_resnet.layer3))
    print(len(base_resnet.layer4))

    blocks_per_layer = [len(base_resnet.layer1),len(base_resnet.layer2), len(base_resnet.layer3), len(base_resnet.layer4)]

    class ResNetAT(ResNet):
        """Attention maps of ResNet-34.
        
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
    
    print(blocks_per_layer)

    model = ResNetAT(Bottleneck if any([isinstance(m, Bottleneck) for m in base_resnet.modules()]) else BasicBlock, blocks_per_layer)
    model.load_state_dict(base_resnet.state_dict())


    def load(url):
        response = requests.get(url)
        return np.ascontiguousarray(Image.open(BytesIO(response.content)), dtype=np.uint8)

    im = load('http://www.zooclub.ru/attach/26000/26132.jpg')

    tr_center_crop = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    model.eval()
    with torch.no_grad():
        transformed_image = tr_center_crop(im)
        x = transformed_image.unsqueeze(0)
        print(x.shape)
        gs = model(x)

    def scale(im, nR, nC):
        nR0 = len(im)     # source number of rows 
        nC0 = len(im[0])  # source number of columns 
        return [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]  
                    for c in range(nC)] for r in range(nR)]

    
    fig2 = plt.figure(constrained_layout=True,figsize=(20,20))
    spec2 = gridspec.GridSpec(ncols=4, nrows=6, figure=fig2)

    ax_original = fig2.add_subplot(spec2[0:2, 0:2])
    ax_original.set_title("Original")
    ax_output_1 = fig2.add_subplot(spec2[0, 2])
    ax_output_2 = fig2.add_subplot(spec2[0, 3])
    ax_output_3 = fig2.add_subplot(spec2[1, 2])
    ax_output_4 = fig2.add_subplot(spec2[1, 3])
    ax_combined_1 = fig2.add_subplot(spec2[2:4,0:2])
    ax_combined_2 = fig2.add_subplot(spec2[2:4, 2:4])
    ax_combined_3 = fig2.add_subplot(spec2[4:,0:2])
    ax_combined_4 = fig2.add_subplot(spec2[4:,2:4])
    
    output_axes = [ax_output_1,ax_output_2,ax_output_3,ax_output_4]
    combined_axes = [ax_combined_1,ax_combined_2,ax_combined_3,ax_combined_4]

    ax_original.imshow(im)

    for ax in output_axes +combined_axes + [ax_original]:
        ax.axis("off")
    for i, g in enumerate(gs):

        shape = im.shape
        activations = np.array(Image.fromarray((g[0].numpy()-g[0].numpy().min())*255*(1/(g[0].numpy().max()-g[0].numpy().min()))).resize((shape[:2][1], shape[:2][0])).convert("L"))
        output_axes[i].imshow(activations)
        output_axes[i].set_title('Layer '+str(i))
        combined_axes[i].imshow(im)
        combined_axes[i].imshow(activations,alpha=0.75)
        combined_axes[i].set_title('Layer '+str(i))

    
    plt.show()
