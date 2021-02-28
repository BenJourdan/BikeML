from loss_functions import SupervisedContrastiveLoss,SupervisedCosineContrastiveLoss
import os,sys
from os.path import join
sys.path.append("/scratch/GIT/BikeML")
from baseline.BaselineModel_1a import BaselineModel_1a
from baseline.BaselineModel_1b import BaselineModel_1b
from dataloaders.dataloader import AddGaussianNoise, AddGaussianNoise, BikeDataLoader
from dataloaders.dataloader import SquarePad,Resize,SquareCrop
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomHorizontalFlip,ColorJitter, RandomGrayscale

base_config = dict(
    epochs = 20,
    lr = 0.01,
    weight_decay = 0.000001,
    image_dim = 224,
    starting_epoch = 9,
    number_of_figures = 16,
    half_precision = False,
    train_backbone = True,
)

dataloader_params = dict(
    data_set_size = 500000,
    data_splits = {"train":55/60,"val": 5/60,"test":5/60},
    normalize = True,
    balance = 0.5,
    num_workers = 20,
    prefetch_factor=1,
    batch_size = 200,
    transforms = torchvision.transforms.Compose([
                                                SquarePad(),
                                                Resize((base_config["image_dim"],base_config["image_dim"])),
                                                ToTensor(),

                                                    ]),

# def get_color_distortion(s=1.0):
#     # s is the strength of color distortion.
#     color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
#     rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
#     rnd_gray = transforms.RandomGrayscale(p=0.2)
#     color_distort = transforms.Compose([
#     rnd_color_jitter,
#     rnd_gray])
# return color_distort


    root = "/scratch/datasets/raw/",
    shuffle=True,
    memory=True,
    half=False,
    pin_memory=True
)

#Baseline_1a
Baseline_1a_config = dict(
    model = BaselineModel_1a,
    dataloader = BikeDataLoader,
    criterion = nn.BCELoss,
    project_path = "./baseline_1a",
    input_shape = (dataloader_params["batch_size"], 3, base_config["image_dim"], base_config["image_dim"]),
    mlp_layers = 4,
)


#Baseline_1b
Baseline_1b_config = dict(
    exp_name = 'messing with visualization 1b callums is a poo',
    model = BaselineModel_1b,
    dataloader = BikeDataLoader,
    criterion = SupervisedCosineContrastiveLoss,
    project_path = "./baseline_1b",
    input_shape = (dataloader_params["batch_size"], 3, base_config["image_dim"], base_config["image_dim"]),
    mlp_layers = 4,
    embedding_dimension = 128,
    margin = 0.9,
    transforms = torchvision.transforms.Compose([
                                            SquareCrop((base_config["image_dim"],base_config["image_dim"])),
                                            RandomHorizontalFlip(p=0.5),
                                            ColorJitter(0.8, 0.8, 0.8, 0.2),
                                            RandomGrayscale(p=0.2),
                                            ToTensor()
                                                ])
)

hyperparameters = base_config
hyperparameters["dataloader_params"] = dataloader_params
hyperparameters.update(Baseline_1b_config)
