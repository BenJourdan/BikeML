
import numpy as np
import torch

import os,sys
sys.path.append("/scratch/GIT/BikeML")
from baseline.experiment_building import ExperimentBuilder
from baseline.model_1a import BaselineModel_1a

from dataloaders.dataloader import  BikeDataLoader

if __name__ == "__main__":
    #TODO: write a get_args script to pull the seed from the command line
    seed = 0
    rng = np.random.RandomState(seed=seed)
    torch.manual_seed(seed=0)

    batch_size = 256

    data_set_size = 500000
    train_data_loader = BikeDataLoader(data_set_type='train',data_set_size=data_set_size*(50/60),balance=0.5,normalize=True,prefetch_factor=1,batch_size=batch_size,num_workers=30)
    val_data_loader = BikeDataLoader(data_set_type='val',data_set_size=data_set_size*(5/60),balance=0.5,normalize=True,prefetch_factor=1,batch_size=batch_size,num_workers=30)
    test_data_loader = BikeDataLoader(data_set_type='test',data_set_size=data_set_size*(5/60),balance=0.5,normalize=True,prefetch_factor=1,batch_size=batch_size,num_workers=30)


    model = BaselineModel_1a(input_shape=(batch_size,3,512,512), mlp_layers=3)

    experiment = ExperimentBuilder(network_model=model,experiment_name='first_run',num_epochs=200,use_gpu=True,
    train_data=train_data_loader, val_data=val_data_loader, test_data=test_data_loader,learning_rate=0.01,weight_decay_coefficient=0)

    experiment_metrics, test_metrics = experiment.run_experiment()

