import os
from os.path import join
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import utils
from torchvision.transforms import Normalize,CenterCrop,ToTensor,ColorJitter,Pad,RandomAffine,Resize,RandomCrop
import pickle
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import numpy as np
from math import comb, floor
from collections import OrderedDict
from itertools import combinations as combs


def load_image(file):
    '''
        Loads an image into numpy array and converts to
        RGB format.
    '''
    return np.array(Image.open(file).convert("RGB"))
 
class BikeDataset(Dataset):
    def __init__(self,root,data_set_size,balance=0.5,transforms=[],cache_dir="./cache"):
        # Number of images from same ad (labelled 1)
        self.num_same_ad = floor(data_set_size * balance)
        # Number of images from diff ads (labelled 0)
        self.num_diff_ad = floor(data_set_size * (1 - balance))
        self.root = root # (root is /scratch/datasets/raw)

        self.same_ad_filenames = []
        # Example
        # same_ad_filenames[0] = ('root///img1.jpg', 'root///img2.jpg')
        self.diff_ad_filenames = []

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

        if os.path.exists(join(cache_dir,"same_ad_filenames.txt")):
                with open(join(cache_dir,"same_ad_filenames.txt"),"r") as f:
                    self.same_ad_filenames = f.read().split("\n")
        else:
            self.populate_same_ad_filename_list()
            with open(join(cache_dir,"same_ad_filenames.txt"),"w") as f:
                f.write("\n".join(self.same_ad_filenames)+"\n")
                
        if os.path.exists(join(cache_dir,"diff_ad_filenames.txt")):
            with open(join(cache_dir,"diff_ad_filenames.txt"),"r") as f:
                    self.diff_ad_filenames = f.read().split("\n")
        else:
            self.populate_diff_ad_filename_list()
            with open(join(cache_dir,"diff_ad_filenames.txt"),"w") as f:
                f.write("\n".join(self.diff_ad_filenames)+"\n")


        #work out total number of valid "same" ad. Aka ads with 2 or more images



        
        pass

    def __len__(self):
        pass

    def __getitem__(self,idx):
        pass
    

    def populate_same_ad_filename_list(self):
        # self.num_same_ad
        # Key: Ad filepath 
        # Value: List of image names
        ad_to_img_pairs = OrderedDict()

        # Looping over ad filenames
        for date in os.listdir(self.root):
            for hour in os.listdir(join(self.root,date)):
                for ad_filepath in os.listdir(join(self.root,date,hour)):
                    imgs = self.imgs_from_ad(join(self.root,date,hour,ad_filepath))
                    # Check if valid ad
                    if imgs:
                        # Add to dict list of image pairs
                        ad_to_img_pairs[ad_filepath] = list(map(lambda x: (join(self.root,date,hour,ad_filepath,x[0]),join(self.root,date,hour,ad_filepath,x[1])) ,list(combs(imgs,2))))

        # Num of img pairs per ad
        combinations = [len(ad_imgs_combs) for ad_imgs_combs in ad_to_img_pairs.values()]
        cdf = np.cumsum(np.array(combinations))
        n = cdf.tolist()[-1]

        # Sampling num_same_ad img pairs from n possible img pairs
        all_ad_img_pairs = list(ad_to_img_pairs.values())
        for i in range(self.num_same_ad):
            
            # Grab random pair index
            global_pair_idx = np.random.randint(0, n)
            ad_idx = cdf[cdf<=global_pair_idx].shape[0]-1
            
            # Computing img pair index from comb_idx
            if ad_idx >= 0:
                local_pair_idx = global_pair_idx - cdf[ad_idx]
            else:
                local_pair_idx = global_pair_idx
            # Grabbing imgs pairs from random ad
            ad_img_pairs = all_ad_img_pairs[ad_idx+1]
            # Store the single image pair
            self.same_ad_filenames.append(ad_img_pairs[local_pair_idx])
    
    def populate_diff_ad_filename_list(self):

        pass

    def imgs_from_ad(self,filepath):
        """
        Input: filepath of an ad
        Output: the number of distinct bike images in that ad
        """
        print(filepath)
        images = [x for x in os.listdir(filepath) if x[-3:]=="jpg"]
        num_images = len(images)
        return False if num_images < 2 else images

class BikeDataLoader(DataLoader):

    def __init__(self,root = "/scratch/datasets/raw/",memory=False,batch_size=100,shuffle=True,num_workers=24,prefetch_factor=3,
                    transforms = torchvision.transforms.Compose([
                                                        RandomCrop((256,256),pad_if_needed=True),
                                                        ToTensor()
                                                    ]),
                    normalize=False, cache_dir="./cache", **kwargs):
        
        if normalize:
            self.load_normalization_constants(cache_dir)
            self.dataset = BikeDataset(root,cache_dir,memory=memory,transforms=torchvision.transforms.Compose([transforms,Normalize(self.means,self.stds)]))
        else:
            self.dataset = BikeDataset(root,cache_dir,memory=memory,transforms=transforms)
        super().__init__(self.dataset,batch_size=batch_size,shuffle=shuffle,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                            **kwargs)

    def compute_normalization_constants(self):
        means = []
        stds = []
        for batch in tqdm(self):
            numpy_img = batch.numpy()

            batch_mean = numpy_img.mean(axis=(0,2,3))
            batch_std = numpy_img.std(axis=(0,2,3))

            means.append(batch_mean)
            stds.append(batch_std)

        means = [str(x) for x in np.ndarray.tolist(np.array(means).mean(axis=0))]
        stds  = [str(x) for x in np.ndarray.tolist(np.array(stds).mean(axis=0))]

        with open(join(self.dataset.cache_dir,"normalization.txt"),"w") as f:
            f.write(",".join(means)+"\n"+",".join(stds)+"\n")
    
    def load_normalization_constants(self,cache_dir="./cache"):
        with open(join(cache_dir,"normalization.txt"),"r") as f:
            data = [[float(x) for x in line.strip("\n").split(",")] for line in f.readlines()]
            self.means = data[0]
            self.stds = data[1]



if __name__ == "__main__":
    torch.manual_seed(0)

    dataset = BikeDataset(root="/scratch/datasets/raw",data_set_size=10000,balance=0.5,transforms=[],cache_dir="./cache")

    splat = dataset.same_ad_filenames[0]
    
    print(splat)
    # dataloader = BikeDataLoader(normalize=True,prefetch_factor=3,batch_size=256,num_workers=16)

    # for i,batch in enumerate(tqdm(dataloader)):
    #     for img in batch:
    #         pass
    #         # fig,ax = plt.subplots(1,1)
    #         # ax.imshow(img.numpy().T)
    #         # plt.show()


# raw normalization constants for randomcrop(256,256,pad_if_needed=True) on SeptOct dataset (batchsize 4096)
# 0.47847774624824524,0.45420822501182556,0.4112544357776642
# 0.26963523030281067,0.25664404034614563,0.2600036859512329