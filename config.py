from loss_functions import SupervisedContrastiveLoss
import os,sys
from os.path import join
sys.path.append("/scratch/GIT/BikeML")
from baseline.model_1a import BaselineModel_1a
from baseline.model_1b import BaselineModel_1b
from dataloaders.dataloader import BikeDataLoader
from dataloaders.dataloader import SquarePad
import torch
from torch import nn
import torchvision
from torchvision.transforms import Normalize,CenterCrop,ToTensor,ColorJitter,Pad,RandomAffine,Resize,RandomCrop

base_config = dict(
    epochs = 100,
    lr = 0.001,
    weight_decay = 0.000001,
    image_dim = 512,
    starting_epoch = 7,
    half_precision = False
)

dataloader_params = dict(
    data_set_size = 500000,
    data_splits = {"train":50/60,"val": 5/60,"test":5/60},
    normalize = True,
    balance = 0.5,
    num_workers = 32,
    prefetch_factor=1,
    batch_size = 256,
    transforms = torchvision.transforms.Compose([
                                                SquarePad(),
                                                Resize((base_config["image_dim"],base_config["image_dim"])),
                                                ToTensor(),
                                                    ]),
    root = "/scratch/datasets/raw/",
    shuffle=True
)

#Baseline_1a
Baseline_1a_config = dict(
    model = BaselineModel_1a,
    dataloader = BikeDataLoader,
    criterion = nn.BCELoss,
    project_path = "./baseline_1a",
    input_shape = (dataloader_params["batch_size"], 3, base_config["image_dim"], base_config["image_dim"]),
    mlp_layers = 4
)


#Baseline_1b
Baseline_1b_config = dict(
    model = BaselineModel_1b,
    dataloader = BikeDataLoader,
    criterion = SupervisedContrastiveLoss,
    project_path = "./baseline_1b",
    input_shape = (dataloader_params["batch_size"], 3, base_config["image_dim"], base_config["image_dim"]),
    mlp_layers = 3,
    embedding_dimension = 128,
)


hyperparameters = base_config
hyperparameters["dataloader_params"] = dataloader_params
hyperparameters.update(Baseline_1a_config)

# train_data_loader = BikeDataLoader(data_set_type='train',data_set_size=data_set_size*(50/60),balance=0.5,normalize=True,prefetch_factor=1,batch_size=batch_size,num_workers=30)
#     val_data_loader = BikeDataLoader(data_set_type='val',data_set_size=data_set_size*(5/60),balance=0.5,normalize=True,prefetch_factor=1,batch_size=batch_size,num_workers=30)
#     test_data_loader = BikeDataLoader(data_set_type='test',data_set_size=data_set_size*(5/60),balance=0.5,normalize=True,prefetch_factor=1,batch_size=batch_size,num_workers=30)

