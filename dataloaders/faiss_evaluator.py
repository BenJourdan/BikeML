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
import faiss
from redis import Redis
from time import sleep
from torchvision.transforms import ToTensor, RandomHorizontalFlip,ColorJitter, RandomGrayscale
import sys
sys.path.append("/scratch/GIT/BikeML")
from dataloaders.dataloader import SquarePad
from baseline.BaselineModel_1b import BaselineModel_1b
from dataloaders.dataloader import SquarePad,Resize,SquareCrop
from collections import defaultdict
import seaborn as sb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Goal: Dataloader which spits out images and the number of imag
# Write function which extracts ad ID from filepath 
def load_image(file):
    return np.array(Image.open(file).convert("RGB"))


class FaissEvaluator:


    def __init__(self,root,model,transforms,embedding_dim=128,nearest_neighbours=60,ad_file_paths=None,target_dir="",**kwargs):
        self.root = root
        self.target_dir = target_dir
        self.model = model


        self.embedding_dim = embedding_dim
        self.k = nearest_neighbours
        self.transforms = transforms

        self.filepaths = []
        self.index_to_num_images = {}
        self.index_to_filepaths = {}
        self.filepaths_to_index = {}
        self.index_to_ad_id = {}
        self.index_to_img_id = {}
        self.ad_to_indices = defaultdict(lambda: [])
        

        self.index = None ## Will store the FAISS index
        
        self.populate_filepaths_and_dicts()

        self.filepaths = self.filepaths
        print(len(self.filepaths))
        self.batch_size = 256
        self.dataloader = FaissDataLoader(self.filepaths,self.transforms,batch_size=self.batch_size)
        print(f"test set size: {len(self.dataloader.dataset)} images from {len(set(self.index_to_ad_id.values()))} Ads.")
        self.embedding_vectors = None
        self.embed_imgs()

        self.initialize_faiss_index()

        self.vec_index_to_ad_id_func = np.vectorize(self.index_to_ad_id_func)
    
    def index_to_ad_id_func(self, idx):
        return self.index_to_ad_id[idx]

    def update_model(self, new_model):
        self.model = new_model
        self.model.eval()
        self.embed_imgs()

    def initialize_faiss_index(self):
        res = faiss.StandardGpuResources()
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.index_cpu_to_gpu(res,0,self.index)

    def embed_imgs(self):
        self.embedding_vectors = torch.zeros(len(self.filepaths),self.embedding_dim)
        firsttime = True
        i=0
        for imgs in tqdm(self.dataloader):
            with torch.no_grad():
                if self.batch_size*i<len(self.filepaths):
                    result =self.model.components["embedding"](imgs.to(device))
                    result = result.cpu()

                    self.embedding_vectors[self.batch_size*i:self.batch_size*(i+1)] = result
                else:
                    result =self.model.components["embedding"](imgs.to(device))
                    result = result.cpu()
                    self.embedding_vectors[self.batch_size*i:] = result
            i+=1

        
        del self.model
        torch.cuda.empty_cache()
        self.embedding_vectors = self.embedding_vectors.detach().numpy()        

    def evaluate(self):
        faiss.normalize_L2(self.embedding_vectors)
        self.index.add(self.embedding_vectors)
        self.D, self.I = self.index.search(self.embedding_vectors, self.k)
        self.I = self.I.astype(np.int32)
        np.savetxt(join(self.target_dir,"D.out"),self.D,delimiter=",")
        np.savetxt(join(self.target_dir,"I.out"),self.I.astype(np.int),delimiter=",",fmt="%i")
        with open(join(self.target_dir,"filenames.txt"),"w") as f:
            f.writelines([x+ "\n" for x in self.filepaths])
        with open(join(self.target_dir,"vectors.npz"),"wb") as f:
            np.save(f,self.embedding_vectors)
        #TODO: Calculate top-5 accuracy etc. from index matrix and class dictionaries       
        self.A = self.vec_index_to_ad_id_func(idx=self.I)

        # Subtract query id from all columns. 
        # If there is a match, the jth column with have a zero in it
        self.A_proper = self.A.copy()
        self.A[:,1:] -= self.A[:,0][:, None]
        k_accuracies = np.zeros(self.k)
        for k in range(1, self.k+1):
            k_accuracies[k-1]=(np.mean(np.count_nonzero(self.A[:,1:k]==0,axis=1)>=1))
        
        with open(join(self.target_dir,"k_accuracies.npz"),"wb") as f:
            np.save(f,k_accuracies)
        
        fig,ax = plt.subplots(1,1)

        ax.plot(np.arange(1, self.k+1), k_accuracies)
        plt.show()


    def evaluate_specified_ad_ids(self, ad_ids=None):

        n = len(ad_ids)

        ad_id_set = set(ad_ids)


        #image indices corresponding to carrera crossfire bikes in the test set
        carrera_idxs = set([key for key,value in self.index_to_ad_id.items() if value in ad_id_set])

        self._trimmed_I = []
        
        hit_counter = 0
        for row in self.I:
            if row[0] in carrera_idxs:
                self._trimmed_I.append(row)
                hit_counter += 1
        print("Hit counter!!!!!: \t\t"+str(hit_counter))


        evaluation_matrix = np.zeros((len(self._trimmed_I), self.k))

        for idx_idx, idx in enumerate(self._trimmed_I):
            for k in range(1, self.k+1):
                k_neighbours = self._trimmed_I[idx_idx][1:k+1]
                hits = 0
                for neighbour in k_neighbours:
                    if neighbour in carrera_idxs:
                        hits += 1

                evaluation_matrix[idx_idx][k-1] = hits

        with open(join(self.target_dir,"evaluation_matrix.npz"),"wb") as f:
            np.save(f,evaluation_matrix)


                
        

    def populate_filepaths_and_dicts(self):
        """
            Dics to populate:
                1. index_to_num_images
                2. index_to_filepaths
                3. index_to_ad_id
        """
        print(self.root)
        idx = 0
        for date in os.listdir(self.root):
            for hour in os.listdir(join(self.root,date)):
                for ad_id in os.listdir(join(self.root,date,hour)):
                    imgs = [x for x in os.listdir(join(self.root,date,hour,ad_id)) if x[-3:]=="jpg"]
                    # Check if valid ad
                    if len(imgs) > 1:
                        
                        #Update lists and dicts
                        for img in imgs:
                            img_filepath = join(self.root,date,hour,ad_id,img)
                            self.filepaths.append(img_filepath)
                            self.index_to_num_images[idx] = len(imgs)
                            self.index_to_filepaths[idx] = img_filepath
                            self.index_to_ad_id[idx] = np.int32(ad_id)
                            self.index_to_img_id[idx] = img
                            #self.ad_to_indices[ad_id].append(idx)
                            idx += 1
        
        

class FaissDataset(Dataset):
    def __init__(self,filenames,transforms):
        
        self.filenames = filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self,idx):
        return self.transforms(Image.open(self.filenames[idx]).convert("RGB"))

class FaissDataLoader(DataLoader):
    def __init__(self,filenames,transforms,batch_size=256,num_workers=28,prefetch_factor=2,**kwargs):
        self.dataset = FaissDataset(filenames=filenames,transforms=transforms)
        super().__init__(self.dataset,batch_size=batch_size,shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch_factor,
                            **kwargs)
        
if __name__ == "__main__":

    #seg stats
    means = 0.2421361356973648,0.2372220754623413,0.22987180948257446
    stds = 0.28368568420410156,0.2785511910915375,0.27250662446022034

    #preseg stats:
    # means = 0.44746488332748413,0.4358515441417694,0.41078630089759827
    # stds = 0.304537832736969,0.30065101385116577,0.2968434989452362

    transform = torchvision.transforms.Compose([
                                            SquareCrop((224,224)),
                                            # RandomHorizontalFlip(p=0.5),
                                            # ColorJitter(0.8, 0.8, 0.8, 0.2),
                                            # RandomGrayscale(p=0.2),
                                            ToTensor(),
                                            Normalize(means,stds)
                                            ])
                                                    
    model = BaselineModel_1b(input_shape=(256,3,224,224),mlp_layers=4,embedding_dimension=128)


    
    checkpoint = torch.load("/scratch/GIT/BikeML/baseline/baseline_1b/margin_tuning/margin0_9.tar")
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
                                                   



    ad_ids = []
    with open("/scratch/GIT/BikeML/embeddings/carrera_crossfire_Ad_IDs.txt","r") as f:
        ad_ids = f.readlines()
    ad_ids = [int(x) for x in ad_ids]

    target_dir = "/scratch/GIT/BikeML/embeddings/results/0_9/"

    # input_dir = "/data_raid/raw/test/"
    input_dir = "/scratch/datasets/detr_filtered/test/"

    faiss_eval = FaissEvaluator(input_dir,model,transform,128,nearest_neighbours=500,target_dir=target_dir)
    faiss_eval.evaluate()
    faiss_eval.evaluate_specified_ad_ids(ad_ids)