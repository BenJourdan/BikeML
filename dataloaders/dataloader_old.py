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



def load_image(file):
    return np.array(Image.open(file).convert("RGB"))

class BikeDataset(Dataset):
    def __init__(self,root,cache_dir,memory=True,transforms=[]):
        
        self.memory = memory
        self.images = []
        self.files  = None

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.cache_dir = cache_dir
        if self.memory:
            if os.path.exists(join(cache_dir,"images.p")):
                self.images = pickle.loads(open(join(cache_dir,"images.p"),"rb"))
            else:
                files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(root)) for f in fn if "jpg" in f]

                with Pool(16) as pool:
                    for i,file in enumerate(files):
                        print(i)
                        self.images.append(load_image(file))
                    # self.images = pool.map(load_image,files)
                pickle.dumps(self.images,open(join(cache_dir,"images.p"),"wb"))

        else:
            if os.path.exists(join(cache_dir,"filenames.txt")):
                with open(join(cache_dir,"filenames.txt"),"r") as f:
                    self.files = f.read().split("\n")
            else:
                self.files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(root)) for f in fn if "jpg" in f]
                with open(join(cache_dir,"filenames.txt"),"w") as f:
                    f.write("\n".join(self.files)+"\n")
            self.files = [x for x in self.files if x!=""]
        
        self.transforms = transforms

        

    def __len__(self):
        if self.memory:
            return len(self.images)
        else:
            return len(self.files)

    def __getitem__(self,idx):
        try:
            if self.memory:
                return self.transforms(self.images[idx])
            else:
                return self.transforms(Image.open(self.files[idx]).convert("RGB"))
        except Exception as e:
            print(idx)
            print(e)
            raise Exception(f"opps {idx},{self.files[idx]}")


class BikeDataLoader(DataLoader):

    def __init__(self,root = "/scratch/datasets/raw",memory=False,batch_size=100,shuffle=True,num_workers=24,prefetch_factor=3,
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
    dataloader = BikeDataLoader(normalize=True,prefetch_factor=3,batch_size=256,num_workers=16)

    for i,batch in enumerate(tqdm(dataloader)):
        for img in batch:
            pass
            # fig,ax = plt.subplots(1,1)
            # ax.imshow(img.numpy().T)
            # plt.show()


# raw normalization constants for randomcrop(256,256,pad_if_needed=True) on SeptOct dataset (batchsize 4096)
# 0.47847774624824524,0.45420822501182556,0.4112544357776642
# 0.26963523030281067,0.25664404034614563,0.2600036859512329