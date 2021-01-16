from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pl_bolts.models.autoencoders import VAE
import sys
sys.path.append('/home/c-abbott/BikeML/dataloaders')
from dataloader import *

def imshow(image, ax=None, title=None, normalize=True, input=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = np.squeeze(image)
    image = image.numpy().transpose(1,2,0)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.show()
    if input:
        plt.savefig('./test_input.png')
    else:
        plt.savefig('./test_output.png')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataloader = BikeDataLoader(normalize=True,prefetch_factor=3,batch_size=1,num_workers=16, cache_dir="/home/c-abbott/BikeML/dataloaders/cache")
image = next(iter(dataloader))

image.to(device)
#display the image
imshow(image)
#create the model
#VAE(input_height, enc_type='resnet18', first_conv=False, maxpool1=False, enc_out_dim=512, kl_coeff=0.1, latent_dim=256, lr=0.0001, **kwargs)
model = VAE(input_height=256, enc_type='resnet18').from_pretrained('cifar10-resnet18')
model.freeze()
#print(VAE.pretrained_weights_available())
#vae = vae.from_pretrained('cifar10-resnet18')

# print(model)
num_params = sum([param.nelement() for param in model.parameters()])
print('Num Params: {}'.format(num_params))

#pass the image through the model
model.to(device)
image = image.to(device)
print(image.device)
output = model(image) 
output = output - torch.min(output)
output = output / torch.max(output) 
# print(output)
print("hereasdf")
# #display the output
# imshow(output, input=False)

# model = CPCV2(encoder='resnet18', pretrained='imagenet128')
# resnet18_unsupervised = model.encoder.freeze()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# num_ftrs = model_ft.fc.in_features
# # Here the size of each output sample is set to 2.
# # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# model_ft.fc = nn.Linear(num_ftrs, 2)

# model_ft = model_ft.to(device)

# criterion = nn.CrossEntropyLoss()

# # Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)