import os
from os.path import join
from typing import Protocol
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import utils
from torchvision.transforms import Normalize,CenterCrop,ToTensor,ColorJitter,Pad,RandomAffine,Resize,RandomCrop
import torchvision.transforms.functional as F
import pickle
import json
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import numpy as np
from math import comb, floor
from collections import OrderedDict
from itertools import combinations as combs
from time import time
import _pickle as cPickle

import torch.multiprocessing

def load_image(file):
    '''
        Loads an image into numpy array and converts to
        RGB format.
    '''
    return np.array(Image.open(file).convert("RGB"))

# Defines a SquarePad class to centre and pad the images
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')
 
class BikeDataset(Dataset):
    def __init__(self,root,data_set_type,data_set_size,balance=0.5,transforms=None,cache_dir="./cache",half=True,memory=True,image_dim=512,memory_dump_path="/data_raid/memory_dump",**kwargs):
        # Number of images from same ad (labelled 1)
        self.num_same_ad = floor(data_set_size * balance)
        # Number of images from diff ads (labelled 0)
        self.num_diff_ad = floor(data_set_size * (1 - balance))
        self.root = root # (root is /scratch/datasets/raw)

        self.same_ad_filenames = []
        # Example
        # same_ad_filenames[0] = ('root///img1.jpg', 'root///img2.jpg')
        self.diff_ad_filenames = []


        self.ad_to_img = OrderedDict()
        self.ad_to_img_pairs = OrderedDict()
        self.half = half

        self.memory = memory
        # check if cache folder exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir


        #check if ordered dicts from Ad filepaths to Image filepaths is cached. If not then create it and cache it
        if os.path.exists(join(cache_dir,"ad_to_imgs_dict_pairs.json")):
            with open(join(cache_dir,"ad_to_imgs_dict_pairs.json"), "r") as file:
                self.ad_to_img_pairs = json.load(file)
        else:
            #populate dicts
            self.populate_ad_to_img_dicts()
            #cache the dicts
            with open(join(cache_dir,'ad_to_imgs_dict_pairs.json'), 'w') as file:
                json.dump(self.ad_to_img_pairs, file)

        
        if os.path.exists(join(cache_dir,"ad_to_imgs_dict_unique.json")):
            with open(join(cache_dir,"ad_to_imgs_dict_unique.json")) as file:
                self.ad_to_img = json.load(file)

        else:
            if len(self.ad_to_img) == 0:
                self.populate_ad_to_img_dicts()

            with open(join(cache_dir,'ad_to_imgs_dict_unique.json'), 'w') as file:
                json.dump(self.ad_to_img, file)


        if os.path.exists(join(cache_dir,"same_ad_filenames.json")):
                with open(join(cache_dir,"same_ad_filenames.json"),"r") as file:
                    self.same_ad_filenames = json.load(file)
        else:
            self.populate_same_ad_filename_list()
            with open(join(cache_dir,"same_ad_filenames.json"),"w") as file:
                json.dump(self.same_ad_filenames, file)
                
        if os.path.exists(join(cache_dir,"diff_ad_filenames.json")):
            with open(join(cache_dir,"diff_ad_filenames.json"),"r") as file:
                    self.diff_ad_filenames = json.load(file)
        else:
            self.populate_diff_ad_filename_list()
            with open(join(cache_dir,"diff_ad_filenames.json"),"w") as file:
                json.dump(self.diff_ad_filenames,file)

        if self.num_same_ad != len(self.same_ad_filenames):
            print('Change in data size, recalculating same_ad_filenames...')
            self.populate_same_ad_filename_list()
            with open(join(cache_dir,"same_ad_filenames.json"),"w") as file:
                json.dump(self.same_ad_filenames, file)

        if self.num_diff_ad != len(self.diff_ad_filenames):
            print('Change in data size, recalculating diff_ad_filenames...')
            self.populate_diff_ad_filename_list()
            with open(join(cache_dir,"diff_ad_filenames.json"),"w") as file:
                json.dump(self.diff_ad_filenames,file)

        # set transforms
        self.transform = transforms

        #### If loading data into memory, then Load the data tensors from disk or create them again (and save them for future use)
        if self.memory:

            #Attempt number 1:
        
            # self.image_array = None
            # self.label_array = None
            
            if not os.path.exists(memory_dump_path):
                        os.makedirs(memory_dump_path)
                        
            # if os.path.exists(join(memory_dump_path,"image_dump.npz")) and os.path.exists(join(memory_dump_path,"label_dump.npz")):
            #     self.image_array = torch.load(join(memory_dump_path,"image_dump.npz"))
            #     self.label_array = torch.load(join(memory_dump_path,"label_dump.npz"))

            #     ###### check if data set size has changed
            #     if len(self) == self.label_array.size[0]:
            #         pass
            #     else:
            #         self.save_data_arrays(memory_dump_path,image_dim)
                
            # else:
            #     self.save_data_arrays(memory_dump_path,image_dim)
            print(len(self.same_ad_filenames),len(self.diff_ad_filenames))
            keys = set([image_path for image_tuple in self.same_ad_filenames+self.diff_ad_filenames for image_path in image_tuple])                
            self.image_database_dict = dict.fromkeys(keys) 

            if os.path.exists(join(memory_dump_path,f"{data_set_type}_image_dict_dump.p")):
                tmp_image_database_dict = cPickle.load(open(join(memory_dump_path,f"{data_set_type}_image_dict_dump.p"),"rb"))
                if len(tmp_image_database_dict.keys()) != len(keys):
                    for key in self.image_database_dict.keys():
                        self.image_database_dict[key] = self.transform(Image.open(key).convert("RGB")).type(torch.uint8)

                    cPickle.dump(self.image_database_dict,open(join(memory_dump_path,f"{data_set_type}_image_dict_dump.p"),"wb"),protocol=-1)
                else:
                    self.image_database_dict = tmp_image_database_dict
            else:
                for i,key in enumerate(self.image_database_dict.keys()):
                    print(i) if i%1000 == 0 else None
                    self.image_database_dict[key] = self.transform(Image.open(key).convert("RGB")).type(torch.uint8)

                cPickle.dump(self.image_database_dict,open(join(memory_dump_path,f"{data_set_type}_image_dict_dump.p"),"wb"),protocol=-1)
             
              
    def __len__(self):
        return self.num_same_ad + self.num_diff_ad

    def __getitem__(self, idx):

        if self.memory==False:
            return self.read_from_disk(idx)
        else:
            return self.read_from_memory(idx)
        
    def save_data_arrays(self,memory_dump_path, image_dim):
        
        #self.image_array = torch.zeros([len(self),2,3,image_dim, image_dim], dtype=torch.uint8)
        #self.label_array = torch.zeros(len(self),dtype=torch.uint8)
        
    # def __init__(self,root = "/scratch/datasets/raw/",batch_size=100,shuffle=True,num_workers=24,prefetch_factor=3,
    #                 transforms = torchvision.transforms.Compose([
    #                                                     SquarePad(),
    #                                                     Resize((256,256)),
    #                                                     ToTensor(),
    #                                                 ]),
    #                 normalize=True, cache_dir="./cache",
    #                 data_set_size = 10000,
    #                 balance = 0.5,
    #                 data_set_type = None,
    #                 data_splits = {'train': 50/60, 'val': 5/60, 'test': 5/60},
    #                 half=False,
    #                 **kwargs):

        # for i in range(len(self)):
        #     if i%10 == 0 :
        #         print(i)
        #     image_a,image_b,label = self.read_from_disk(i)
        #     self.image_array[i][0] = image_a
        #     self.image_array[i][1] = image_b
        #     self.label_array[i] = label

        torch.save(self.image_array, join(memory_dump_path,"image_dump.npz"))
        torch.save(self.label_array, join(memory_dump_path,"label_dump.npz"))
    
    def read_from_memory(self,idx):
        image_a = None
        image_b = None
        label = torch.Tensor(1)

        if idx < self.num_same_ad:
            image_a = self.image_database_dict[self.same_ad_filenames[idx][0]]
            image_b = self.image_database_dict[self.same_ad_filenames[idx][1]]
            label[0] = 1.0
        else:
            actual_idx = idx - self.num_same_ad
            image_a = self.image_database_dict[self.diff_ad_filenames[actual_idx][0]]
            image_b = self.image_database_dict[self.diff_ad_filenames[actual_idx][1]]
            label[0] = 0.0
        
        if self.half==True:
            return  image_a.half(),image_b.half(),label.half()
        else: 
            return  image_a.float(),image_b.float(),label.float()
    
    def read_from_disk(self,idx):
        image_a = None
        image_b = None
        label = torch.Tensor(1)
        if idx < self.num_same_ad:
            image_a = self.transform(Image.open(self.same_ad_filenames[idx][0]).convert("RGB"))
            image_b = self.transform(Image.open(self.same_ad_filenames[idx][1]).convert("RGB"))
            label[0] = 1.0
        else:
            actual_idx = idx - self.num_same_ad
            image_a = self.transform(Image.open(self.diff_ad_filenames[actual_idx][0]).convert("RGB"))
            image_b = self.transform(Image.open(self.diff_ad_filenames[actual_idx][1]).convert("RGB"))
            label[0] = 0.0
        
        if self.half==True:
            return  image_a.half(),image_b.half(),label.half()
        else: 
            return  image_a.float(),image_b.float(),label.float()
                
    
    def populate_ad_to_img_dicts(self):
        # Key: Ad filepath 
        # Value: List of image filepaths
        self.ad_to_img_pairs = OrderedDict()
        self.ad_to_img = OrderedDict()

        # Looping over ad filenames
        for date in os.listdir(self.root):
            for hour in os.listdir(join(self.root,date)):
                for ad_filepath in os.listdir(join(self.root,date,hour)):
                    imgs = self.imgs_from_ad(join(self.root,date,hour,ad_filepath))
                    # Check if valid ad
                    if imgs:
                        # Add to dict list of images
                        self.ad_to_img[ad_filepath] = [join(self.root,date,hour,ad_filepath,x) for x in imgs]
                        # Add to dict list of image pairs
                        self.ad_to_img_pairs[ad_filepath] = list(map(lambda x: (join(self.root,date,hour,ad_filepath,x[0]),join(self.root,date,hour,ad_filepath,x[1])) ,list(combs(imgs,2))))
                        
    
    def populate_same_ad_filename_list(self):
        # self.num_same_ad
        self.same_ad_filenames = []
        # Num of img pairs per ad
        combinations = [len(ad_imgs_combs) for ad_imgs_combs in self.ad_to_img_pairs.values()]
        cdf = np.cumsum(np.array(combinations))
        n = cdf.tolist()[-1]

        # Sampling num_same_ad img pairs from n possible img pairs
        all_ad_img_pairs = list(self.ad_to_img_pairs.values())
        global_pair_arr = np.arange(0, n) # lists all global_pair_arr.shuffle(inplace=True)possible global pair indices from 0 to n
        np.random.shuffle(global_pair_arr)

        #Check that the number of same pairs requested is not larger than the dataset
        if self.num_same_ad > n:
            raise ValueError('num_same_ad variable ({}) exceeds the number of same-ad image pairs ({})'.format(self.num_same_ad,n))
        
        for i in range(self.num_same_ad):
            # Grab random pair index
            global_pair_idx = global_pair_arr[i] # sample from the global pair indices without replacement
            ad_idx = cdf[cdf<=global_pair_idx].shape[0]-1
            
            # Computing img pair index from comb_idx
            if ad_idx >= 0:
                local_pair_idx = global_pair_idx - cdf[ad_idx]
            else:
                local_pair_idx = global_pair_idx
            # Grabbing imgs pairs from random ad
            ad_to_img_pairs = all_ad_img_pairs[ad_idx+1]
            # Store the single image pair
            self.same_ad_filenames.append(ad_to_img_pairs[local_pair_idx])
    
    def populate_diff_ad_filename_list(self):
        """
        Doing the easy sampling way rn which can have collisions. 

        Hard way needs to compute n = \sum_{k=1}^n \Big[m_k (\sum_{i=k+1}^N m_i)] where m_i is the number of images in the ith Ad
        """

        self.diff_ad_filenames = []
        all_ad_img = list(self.ad_to_img.values())
        imgs_per_ad = [len(val) for val in self.ad_to_img.values()]
        cdf = np.cumsum(np.array(imgs_per_ad))
        n = cdf.tolist()[-1] # total number of bike images
        global_arr = np.arange(0, n) # lists all possible global pair indices from 0 to n
        np.random.shuffle(global_arr)

        # Check that the number of different pairs requested is not larger than the dataset
        if self.num_diff_ad > n/2:
            raise ValueError('num_diff_ad variable ({}) exceeds the number of different-ad images ({})'.format(self.num_diff_ad,n))
        
        for i in range(self.num_diff_ad):
            global_idx_1 = 0
            global_idx_2 = 0

            # Grab first ad index
            global_idx_1 = global_arr[2*i]
            ad_idx_1 = cdf[cdf<=global_idx_1].shape[0]-1
            
            # Grab second ad index
            global_idx_2 = global_arr[2*i+1] 
            ad_idx_2 = cdf[cdf<=global_idx_2].shape[0]-1

            # !------ COLLISION -------!
            # Are the images from the same ad?
            collision_count = -1
            # Pairwise swap adjacent indices of global_arr to prevent the collision
            # Keeps going until we no longer collide
            while ad_idx_1 == ad_idx_2:
                collision_count += 1
                print(f'Sampling collision at index: {ad_idx_1}')
                temp = global_idx_2
                global_idx_2 = global_arr[(2*i+2+collision_count)%n]
                global_arr[(2*i+2+collision_count)%n] = temp
                ad_idx_2 = cdf[cdf<=global_idx_2].shape[0]-1
            
            # Computing img pair idx
            if ad_idx_1 >= 0:
                local_idx_1 = global_idx_1 - cdf[ad_idx_1]
            else:
                local_idx_1 = global_idx_1
                
            if ad_idx_2 >= 0:
                local_idx_2 = global_idx_2 - cdf[ad_idx_2]
            else:
                local_idx_2 = global_idx_2
            
            # Grabbing imgs from random ads
            ad_1_to_img = all_ad_img[ad_idx_1+1]
            ad_2_to_img = all_ad_img[ad_idx_2+1]

            # Store the single images 
            self.diff_ad_filenames.append((ad_1_to_img[local_idx_1],ad_2_to_img[local_idx_2]))
    
    def imgs_from_ad(self,filepath):
        """
        Input: filepath of an ad
        Output: the number of distinct bike images in that ad
        """
        images = [x for x in os.listdir(filepath) if x[-3:]=="jpg"]
        num_images = len(images)
        return False if num_images < 2 else images

class BikeDataLoader(DataLoader):

    def __init__(self,root = "/scratch/datasets/raw/",batch_size=100,shuffle=True,num_workers=24,prefetch_factor=3,
                    transforms = torchvision.transforms.Compose([
                                                        SquarePad(),
                                                        Resize((256,256)),
                                                        ToTensor(),
                                                    ]),
                    normalize=True, cache_dir="./cache",
                    data_set_size = 10000,
                    balance = 0.5,
                    data_set_type = None,
                    data_splits = {'train': 50/60, 'val': 5/60, 'test': 5/60},
                    half=False,
                    **kwargs):
        
        if data_set_type == 'train':
            cache_dir = join(root,f"cache_train")
            root = "/scratch/datasets/raw/train"
            data_set_size *= data_splits['train']
        elif data_set_type == 'val':
            cache_dir = join(root,f"cache_val")
            root = "/scratch/datasets/raw/val"
            data_set_size *= data_splits['val']
        elif data_set_type == 'test':
            cache_dir = join(root,f"cache_test")
            root = "/scratch/datasets/raw/test"
            data_set_size *= data_splits['test']
    
        if normalize:
            self.load_normalization_constants(cache_dir)
            self.dataset = BikeDataset(root,data_set_type,data_set_size,balance,transforms=torchvision.transforms.Compose([transforms,Normalize(self.means,self.stds)]),cache_dir=cache_dir,half=half,**kwargs)
        else:
            self.dataset = BikeDataset(root,data_set_type,data_set_size,balance,transforms=transforms,cache_dir=cache_dir,half=half,**kwargs)
        super().__init__(self.dataset,batch_size=batch_size,shuffle=shuffle,
                            num_workers=num_workers, prefetch_factor=prefetch_factor)

    def compute_normalization_constants(self):
        means = []
        stds = []
        for batch in tqdm(self):

            image_a, image_b, label = batch
            numpy_img = image_a.numpy()

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

    # dataset = BikeDataset(root="/scratch/datasets/raw",data_set_size=10000,balance=0.5,transforms = torchvision.transforms.Compose([
    #                                                     SquarePad(),
    #                                                     Resize((256,256)),
    #                                                     ToTensor(),
    #                                                 ]),cache_dir="./cache")
    

    dataloader = BikeDataLoader(data_set_type="train",data_set_size=250000,balance=0.5,normalize=False,prefetch_factor=1,batch_size=256,num_workers=1,
                                    transforms = torchvision.transforms.Compose([
                                                        SquarePad(),
                                                        Resize((512,512)),
                                                        ToTensor(),
                                                    ]),
                                                    memory=True,
                                                    image_dim=512,
                                                    memory_dump_path="/data_raid/memory_dump",pin_memory=False)
                                                    

    # for i,batch in enumerate(tqdm(dataloader)):
    #     a,b,l = batch

# raw normalization constants for randomcrop(256,256,pad_if_needed=True) on SeptOct dataset (batchsize 4096)
# 0.47847774624824524,0.45420822501182556,0.4112544357776642
# 0.26963523030281067,0.25664404034614563,0.2600036859512329

# 0.36192944645881653,0.34354230761528015,0.306730717420578
# 0.292623370885849,0.2772243022918701,0.260631799697876
