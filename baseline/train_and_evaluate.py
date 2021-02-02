
import numpy as np
import torch
import torch.nn as nn
import os,sys
sys.path.append("/scratch/GIT/BikeML")
from baseline.model_1b import BaselineModel_1b
from config import hyperparameters
from baseline.experiment_building import ExperimentBuilder
from baseline.analysis import UnNormalize
from baseline.model_1a import BaselineModel_1a
from os.path import join
from dataloaders.dataloader import BikeDataLoader
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from torch.optim import Adam

prop = FontProperties(fname="NotoColorEmoji.tff")
plt.rcParams['font.family'] = prop.get_family()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def model_pipeline(hyperparameters):
    seed = 0
    rng = np.random.RandomState(seed=seed)
    torch.manual_seed(seed=0)

    # tell wandb to get started
    with wandb.init(config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        # make the model, data, and optimization problem
        model, train_loader, val_loader, test_loader, criterion, optimizer = make(config,hyperparameters)
        # and use them to train the model
        train(model, train_loader, val_loader, criterion, optimizer, config)

        # and test its final performance
        evaluate(model,optimizer,test_loader,criterion,dataset='test')
        return model

def make(config,hyperparameters):
    # config.project_name
    # Make the data
    train_loader = hyperparameters["dataloader"](data_set_type='train',**hyperparameters["dataloader_params"])
    test_loader = hyperparameters["dataloader"](data_set_type='test',**hyperparameters["dataloader_params"])
    val_loader = hyperparameters["dataloader"](data_set_type="val",**hyperparameters["dataloader_params"])

    # Make the model
    model = hyperparameters["model"](**config)

    # Make the loss and optimizer
    criterion = hyperparameters["criterion"]()
    optimizer = Adam(model.parameters(),lr=config.lr, weight_decay=config.weight_decay)

    #load weights and optimizer state if continuing:
    if config.starting_epoch>0:
        path = config.project_path
        checkpoint = torch.load(join(path,"models",f"model_{config.starting_epoch}.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for g in optimizer.param_groups:
        g['lr'] = config.lr
        g["weight_decay"] = config.weight_decay

    #make weights half precision if told to
    if config.half_precision:
        model.half()  # convert to half precision
        #make sure bn layers are floats for stability
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    model.to(device)
    optimizer_to(optimizer,device)

    
    return model, train_loader, val_loader, test_loader, criterion, optimizer


def train(model, train_loader, val_loader, criterion, optimizer, config):
    # tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    batch_ct = 0

    model_type = type(model)

    for epoch in range(config.starting_epoch,config.epochs):
        model.train()
        with tqdm(total=len(train_loader),ncols=160) as pbar_train:
            for data in train_loader:
                loss,outputs = train_batch(data, model, optimizer, criterion, model_type)
                example_ct +=  data[0].shape[0]
                batch_ct += 1
                # Report metrics every batch
                #TODO: Change this so that its not every batch
                accuracy = model.train_compute_accuracy(outputs)
                train_log(loss, accuracy, example_ct, epoch)
                pbar_train.update(1)
                pbar_train.set_description(f" Epoch: {epoch} loss: {loss:.4f}, accuracy: {accuracy:.4f}")
        #validate
        evaluate(model,optimizer,val_loader,criterion,dataset='val',path=config.project_path,epoch=epoch)


def train_batch(data, model, optimizer, criterion, model_type):
    loss = model.train_batch(data, criterion, device, model)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()
    # Step with optimizer
    optimizer.step()
    return loss, [outputs, labels]

def train_log(loss,accuracy, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": float(loss),"accuracy":accuracy}, step=example_ct)

def evaluate(model,optimizer, loader, criterion, dataset,path=None,epoch=None):
    model.eval() 
    model_type = type(model)
    # Run the model on some test examples
    accuracies = []
    losses = []


    viz_flag =  True
    with torch.no_grad():
        
        for data in loader:
            loss, accuracy, outputs = model.evaluate_batch(data, criterion, model, device)
            if viz_flag:
                plot = model.visualize(data,outputs,epoch,unNormalizer = UnNormalize(loader.means,loader.stds),figures_to_store=4)
                viz_flag = False
                if plot != False:
                    wandb.log({"examples":plot})
            losses.append(loss)
            accuracies.append(accuracy)

    if dataset == "val":
        wandb.log({"{}_accuracy".format(dataset): np.mean(accuracies),"global_step":epoch})
        wandb.log({"{}_loss".format(dataset): np.mean(losses),"global_step":epoch})

        # Save the model
        actual_path = join(path,"models")
        if not os.path.exists(actual_path):
            os.makedirs(actual_path)
        #save weights and optimizer
        torch.save({
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict()
        },join(path,"models",f"model_{epoch}.tar"))




if __name__ == "__main__":
    # login to wandb:
    wandb.login()
    model_pipeline(hyperparameters)
    print("we're done :)")
