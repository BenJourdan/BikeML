
import numpy as np
import torch
import matplotlib.pyplot as plt

import os,sys
sys.path.append("/scratch/GIT/BikeML")
from baseline.experiment_building import ExperimentBuilder
from baseline.BaselineModel_1a import BaselineModel_1a

from dataloaders.dataloader import  BikeDataLoader

from matplotlib.font_manager import FontProperties

prop = FontProperties(fname="NotoColorEmoji.tff")
plt.rcParams['font.family'] = prop.get_family()

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


 

if __name__ == "__main__":
    #TODO: write a get_args script to pull the seed from the command line
    seed = 0
    rng = np.random.RandomState(seed=seed)
    torch.manual_seed(seed=0)

    batch_size = 4

    data_set_size = 500000
    # train_data_loader = BikeDataLoader(data_set_type='train',data_set_size=data_set_size*(50/60),balance=0.5,normalize=True,prefetch_factor=1,batch_size=batch_size,num_workers=30)
    # val_data_loader = BikeDataLoader(data_set_type='val',data_set_size=data_set_size*(5/60),balance=0.5,normalize=True,prefetch_factor=1,batch_size=batch_size,num_workers=30)
    test_data_loader = BikeDataLoader(data_set_type='test',data_set_size=data_set_size*(5/60),balance=0.5,normalize=True,prefetch_factor=1,batch_size=batch_size,num_workers=30)


    data_loader = test_data_loader

    model = BaselineModel_1a(input_shape=(batch_size,3,512,512), mlp_layers=3)



    unNormalizer = UnNormalize(data_loader.means,data_loader.stds)

    for image_a,image_b,label in data_loader:
        print(image_a.shape)

        outputs = model(image_a,image_b)
        fig,ax = plt.subplots(batch_size,3,figsize=(20,20))

        # print(f"predicted: {output.item()}\t actual: {label.item()}")
        for i in range(batch_size):
            img_a = unNormalizer(image_a[i]).T
            img_b = unNormalizer(image_b[i]).T

            print(img_a)

            ax[i][0].imshow(img_a)
            ax[i][1].imshow(img_b)
            ax[i][2].annotate("Same Ad" if label[i]==1.0 else "Different Ad",(0,0.5))
            ax[i][2].annotate(u"\U0001F604",(0.5,0.5),size=20) if abs(label[i]-outputs[i])<=0.5 else ax[i][2].annotate(u"\U0001F62D",(0.5,0.5),size=20)
            print(outputs[i],label[i])
            
            ax[i][0].set_axis_off()
            ax[i][1].set_axis_off()
            ax[i][2].set_axis_off()

        plt.tight_layout()
        plt.show()

    # experiment = ExperimentBuilder(network_model=model,experiment_name='first_run',num_epochs=200,use_gpu=True,
    # train_data=train_data_loader, val_data=val_data_loader, test_data=test_data_loader,learning_rate=0.01,weight_decay_coefficient=0)

    # experiment_metrics, test_metrics = experiment.run_experiment()
