import numpy as np
import torch
import torch.nn as nn
import os,sys
sys.path.append("/scratch/GIT/BikeML")
from baseline.BaselineModel_1b import BaselineModel_1b
from config import hyperparameters
from baseline.experiment_building import ExperimentBuilder
from baseline.analysis import UnNormalize
from baseline.BaselineModel_1a import BaselineModel_1a
from os.path import join
from dataloaders.dataloader import BikeDataLoader
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from torch.optim import Adam
from itertools import chain
import numpy as np
import torch
import matplotlib.pyplot as plt

import os,sys
sys.path.append("/scratch/GIT/BikeML")
from baseline.experiment_building import ExperimentBuilder
from baseline.BaselineModel_1a import BaselineModel_1a

from dataloaders.dataloader import  BikeDataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


from matplotlib.font_manager import FontProperties
from pytorch_memlab import MemReporter
import inspect
from baseline.gpu_mem_track import  MemTracker


class TrainAndEvaluate:

    def __init__(self,hyperparameters,seed=0,**kwargs) -> None:
        print(hyperparameters)
        #setup matplotlib fonts
        self.prop = FontProperties(fname="NotoColorEmoji.tff")
        plt.rcParams['font.family'] = self.prop.get_family()

        #setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #setup memory reporter
        self.reporter = MemReporter()


        #setup random seed:
        self.rng = np.random.RandomState(seed=seed)

        self.hyperparameters = hyperparameters
    
    def run(self):
        with wandb.init(project="bike-1b",config=self.hyperparameters, name=self.hyperparameters["exp_name"],save_code=True):
            # access all HPs through wandb.config, so logging matches execution!
            self.config = wandb.config
            # make the model, data, and optimization problem


            self.make()
            # and use them to train the model
            torch.cuda.empty_cache()
            self.reporter.report()
            self.train()
            # and test its final performance
            # self.evaluate(dataset='test')
            return self.model 

    def make(self):
        # Make the data
        self.train_loader = self.hyperparameters["dataloader"](data_set_type='train',**self.hyperparameters["dataloader_params"])
        self.test_loader = self.hyperparameters["dataloader"](data_set_type='test',**self.hyperparameters["dataloader_params"])
        self.val_loader = self.hyperparameters["dataloader"](data_set_type="val",**self.hyperparameters["dataloader_params"])
        self.tiny_val_loader = self.hyperparameters["dataloader"](data_set_type="val",data_set_size = 4,
                                                                    normalize = True,
                                                                    balance = 0.5,
                                                                    num_workers = 20,
                                                                    data_splits = {"val":1.0 },
                                                                    prefetch_factor=1,
                                                                    batch_size = 4,
                                                                    transforms = self.hyperparameters["transforms"],
                                                                    shuffle=False)
        self.tiny_filepaths = self.tiny_val_loader.dataset.same_ad_filenames + self.tiny_val_loader.dataset.diff_ad_filenames
        
        # self.tiny_filepaths = list(sum(self.tiny_filepaths, ()))
        # Flatten list of tuples into list
        # self.tiny_filepaths = [a for b in self.tiny_filepaths for b in a]
        self.tiny_filepaths = list(chain.from_iterable(self.tiny_filepaths))
        tiny_image_as, tiny_image_bs, _ = next(iter(self.tiny_val_loader)) 

        # Flatten batch of image pairs to batch of single images
        image_list = [torch.unsqueeze(x,0) for x in chain.from_iterable(zip(tiny_image_as,tiny_image_bs))]
        self.tiny_batch = torch.cat(image_list)

        # Make the model

        self.model = self.hyperparameters["model"](**self.config)

        # Make the loss and optimizer
        self.criterion = self.hyperparameters["criterion"](**self.hyperparameters)
        self.base_optimizer =  Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
       
        # load weights and optimizer state if continuing:
        if self.config.starting_epoch>0:
            path = self.config.project_path
            checkpoint = torch.load(join(path,"models",f"model_{self.config.starting_epoch}.tar"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.base_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for g in self.base_optimizer.param_groups:
            g['lr'] = self.config.lr
            g["weight_decay"] = self.config.weight_decay

        # make weights half precision if told to
        if self.config.half_precision:
            self.model.half()  # convert to half precision
            #make sure bn layers are floats for stability
            for layer in self.model.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()

        self.model.to(self.device)
        self.move_base_optimizer_to_device()

        self.optimizer = CosineAnnealingLR(self.base_optimizer,last_epoch=-1, T_max = self.hyperparameters["epochs"], eta_min=0.00002)
        for _ in range(self.hyperparameters["starting_epoch"]):
            self.optimizer.step()
    def train(self):
        wandb.watch(self.model,self.criterion,log="all",log_freq=10)

        # Run training and track with wandb
        example_seen = 0  # number of examples seen
        batch_seen = 0

        for epoch in range(self.config.starting_epoch,self.config.epochs):
            self.model.train()
            self.current_epoch = epoch
            with tqdm(total=len(self.train_loader),ncols=120) as pbar_train:
                for data in self.train_loader:
                    torch.cuda.empty_cache()
                    print(data[0].shape)
                    print(data[1].shape)
                    print(data[2].shape)
                    self.image_as, self.image_bs,labels = data[0].to(self.device),data[1].to(self.device),data[2].to(self.device)

                    loss,outputs = self.train_batch([self.image_as,self.image_bs,labels])
                    example_seen +=  data[0].shape[0]
                    batch_seen += 1
                    # Report metrics every 10 batches
                    if batch_seen % 10 == 0:
                        self.model.track_metrics(outputs,epoch,step=example_seen,criterion=self.criterion,loss=loss,split="train")

                    pbar_train.update(1)
                    pbar_train.set_description(f" Epoch: {epoch} loss: {loss:.4f}")
            #validate
            torch.cuda.empty_cache() 
            # reporter.report()
            self.evaluate(dataset='val',epoch=epoch)

    def train_batch(self,data):
        loss,outputs,labels = self.model.train_batch(data,self.criterion,self.device,self.model)

        #backward pass:
        self.base_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(epoch=self.current_epoch)
        self.base_optimizer.step()
        return loss.detach().item(),[[outputs[0].detach().cpu(),outputs[1].detach().cpu()],labels.detach().cpu()]

    def evaluate(self,dataset="val",epoch=None):

        path=self.config.project_path
        
        #put model in evaluation mode:
        accuracies = []
        losses = []

        viz_flag =  True
        list_of_outputs = None
        list_of_image_a_outputs = None
        list_of_image_b_outputs = None
        list_of_labels = None

        #Visualise attention maps of the model
        self.model.am_viz(self.tiny_batch, self.tiny_filepaths)

        loader = self.val_loader if dataset=="val" else self.test_loader
        with torch.no_grad():
            for data in loader:
                torch.cuda.empty_cache() 
                # reporter.report()
                self.image_as, self.image_bs,labels = data[0].to(self.device),data[1].to(self.device),data[2].to(self.device)
                loss, accuracy, outputs = self.model.evaluate_batch([self.image_as,self.image_bs,labels],self.criterion,self.device,self.model)
                if viz_flag:
                    list_of_image_a_outputs = outputs[0].cpu()
                    list_of_image_b_outputs = outputs[1].cpu()
                    list_of_labels = data[2].cpu()
                    self.model.visualize(data,
                                            outputs,
                                            epoch,
                                            number_of_figures=self.hyperparameters["number_of_figures"],
                                            unNormalizer = UnNormalize(loader.means,loader.stds))
                    viz_flag =False
                else:
                    list_of_image_a_outputs = torch.cat((list_of_image_a_outputs, outputs[0].cpu()), 0)
                    list_of_image_b_outputs = torch.cat((list_of_image_b_outputs, outputs[0].cpu()), 0)
                    list_of_labels = torch.cat((list_of_labels,data[2].cpu()),0)

                losses.append(loss)
                accuracies.append(accuracy)
        list_of_outputs = [[list_of_image_a_outputs, list_of_image_b_outputs], list_of_labels]
        if dataset == "val":
            self.model.track_metrics(list_of_outputs,epoch,step=epoch,criterion=self.criterion,loss=np.mean(losses),split="val")
            # wandb.log({"{}_accuracy".format(dataset): np.mean(accuracies),"global_step":epoch})
            # wandb.log({"{}_loss".format(dataset): np.mean(losses),"global_step":epoch})

            # Save the model
            actual_path = join(path,"models")
            if not os.path.exists(actual_path):
                os.makedirs(actual_path)
            #save weights and optimizer
            torch.save({
                "epoch":epoch,
                "model_state_dict":self.model.state_dict(),
                "optimizer_state_dict":self.base_optimizer.state_dict()
            },join(path,"models",f"model_{epoch}.tar"))

        if dataset == "test":
            self.model.track_extra_metrics(list_of_outputs, epoch,split="test")

    def move_base_optimizer_to_device(self):
        for param in self.base_optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(self.device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(self.device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(self.device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(self.device)

if __name__ == "__main__":
    wandb.login()
    
    experiment = TrainAndEvaluate(hyperparameters)

    model = experiment.run()