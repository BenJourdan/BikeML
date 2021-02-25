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

import sys
sys.path.append("/scratch/GIT/BikeML")
from dataloaders.dataloader import SquarePad
from baseline.BaselineModel_1b import BaselineModel_1b


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Goal: Dataloader which spits out images and the number of imag
# Write function which extracts ad ID from filepath 
def load_image(file):
    return np.array(Image.open(file).convert("RGB"))


class FaissEvaluator:
    def __init__(self,root,model,transforms,embedding_dim=128,nearest_neighbours=60,**kwargs):
        self.root = root
        self.model = model

        for params in model.parameters():

            params.require_grad = False
            print(params.requires_grad) # prints two True statements?

        self.embedding_dim = embedding_dim
        self.k = nearest_neighbours
        self.transforms = transforms

        self.filepaths = []
        self.index_to_num_images = {}
        self.index_to_filepaths = {}
        self.index_to_ad_id = {}

        self.index = None ## Will store the FAISS index
        
        self.populate_filepaths_and_dicts()
        self.filepaths = self.filepaths
        self.batch_size = 256
        self.dataloader = FaissDataLoader(self.filepaths,self.transforms,batch_size=self.batch_size)

        self.embedding_vectors = None
        self.embed_imgs()

        self.initialize_faiss_index()

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
                    result =self.model.components["backbone"](imgs.to(device))
                    result = result.cpu()

                    self.embedding_vectors[self.batch_size*i:self.batch_size*(i+1)] = result
                else:
                    result =self.model.components["backbone"](imgs.to(device))
                    result = result.cpu()
                    self.embedding_vectors[self.batch_size*i:] = result
            i+=1

        
        del self.model
        torch.cuda.empty_cache()
        self.embedding_vectors = self.embedding_vectors.detach().numpy()
        

    def evaluate(self):
        print(self.embedding_vectors.shape)
        faiss.normalize_L2(self.embedding_vectors)
        self.index.add(self.embedding_vectors)
        self.D, self.I = self.index.search(self.embedding_vectors, self.k)
        np.savetxt("D.out",self.D,delimiter=",")
        np.savetxt("I.out",self.I.astype(np.int),delimiter=",",fmt="%i")
        with open("filenames.txt","w") as f:
            f.writelines([x+ "\n" for x in self.filepaths])
        with open("vectors.npz","wb") as f:
            np.save(f,self.embedding_vectors)
        #TODO: Calculate top-5 accuracy etc. from index matrix and class dictionaries       
        
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
                            self.index_to_ad_id[idx] = ad_id
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

    

    means = 0.36321571469306946,0.3454224765300751,0.3075723350048065
    stds = 0.30779534578323364,0.29204192757606506,0.279329776763916
    transform = torchvision.transforms.Compose([
                                                SquarePad(),
                                                Resize((512,512)),
                                                ToTensor(),
                                                Normalize(means,stds)
                                                    ])
                                                    

    model = BaselineModel_1b(input_shape=(256,3,512,512),mlp_layers=4,embedding_dimension=128)


    checkpoint = torch.load(join("/scratch/GIT/BikeML/baseline/baseline_1b","models",f"model_5.tar"))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
                                                   
    faiss_eval = FaissEvaluator("/scratch/datasets/raw/test",model,transform,128)


    faiss_eval.evaluate()


# 0,233856,155579,157862,190990,250274,238919,236735,200224,34644,217146,233852,236739,157860,47882,233853,44689,239263,1335,190987,126841,3086,107655,125792,235077,83784,24700,190986,79018,224079,233854,191019,167863,70924,204278,150366,24699,140794,241902,21026,79016,43434,236737,59583,150367,34236,238259,24698,1,51693,24798,94744,236736,79017,324,190988,110373,94742,68943,68947
# 1,217255,189414,1335,34644,190986,171172,153341,83784,241902,190988,241899,16832,93552,217146,81875,77698,6361,168949,97332,125792,191019,24700,81562,189412,147920,59243,190987,233852,204276,238919,167307,68947,189410,79018,200224,150367,222792,324,79914,146217,19833,147919,175098,236739,223158,83785,233856,14014,224079,34643,178192,171253,153253,34236,0,211286,117083,14013,129977
# 2,228919,226656,94298,74371,29002,70924,94297,150115,236738,27226,224081,178494,177624,51405,74264,155577,233853,18975,157862,250274,59244,0,16086,233852,238919,70926,217146,28030,54323,236736,68945,233856,68947,228917,1338,106114,190988,236735,11109,191019,22494,147920,24698,74263,34644,22496,51693,1335,140794,16834,77857,24699,190987,246791,125281,190990,157860,94744,233854
# 3,184444,18992,243885,16521,162161,247433,161687,122665,208612,247896,222446,9026,51578,177644,184446,43607,249509,81353,185363,249753,87401,198292,43608,61934,97046,131121,138433,24904,162158,219477,219850,227159,10654,2869,116115,3949,42662,162352,16524,42995,247134,2927,169101,80019,119797,75233,2886,193959,108858,4987,162351,245752,98131,115065,196402,112809,52877,18380,195492
# 4,127798,113272,98650,165121,184167,134761,116157,31173,93113,154009,170401,111822,169430,98783,199284,145925,141044,98649,176025,138204,192182,71206,264454,149085,112938,18379,174309,169428,247884,165397,189379,48525,222355,8320,243958,243962,39993,109627,92629,19214,9206,9207,70745,81365,171796,39327,177214,192936,157969,197622,126986,141320,97530,257403,257400,255747,210438,41128,248647
# 5,209473,19886,30728,202530,107463,248697,251681,192185,251157,176976,82986,79714,99136,76482,167256,199516,270,66615,141328,227901,6800,84873,151079,235388,84871,254884,54819,54534,5716,123881,203908,109626,243305,259989,48587,211414,176977,222198,54815,169875,242935,81554,39048,237224,237185,36256,132878,173311,145641,61363,46836,255935,33893,133206,78278,187236,261885,46839,111605
# 6,62195,69791,264496,62192,171217,123753,87216,23684,109271,214461,79150,1556,69792,174308,192185,177960,67002,118636,22813,58828,184479,119351,100718,190855,89797,178963,86938,187236,112533,203908,2136,44613,255861,251630,237223,42233,240104,73627,54815,140091,222198,60964,23696,19407,229834,222375,214462,10300,237185,232432,50447,167910,187440,54818,22168,103688,151079,243999,43498
# 7,91752,262302,5875,11173,161055,45940,214962,179840,207063,230043,105611,197874,22014,144457,206439,193761,238267,194303,169821,56222,224769,204855,13040,128494,170380,222359,107231,24900,218488,106975,76923,119458,219526,69708,105609,219516,244207,233878,58770,72157,243102,55583,194353,115187,123210,36818,34161,39092,227002,148931,38846,64432,49580,5029,100513,12407,90543,144642,225568
# 8,140417,227532,202134,154956,61326,99991,191059,224764,229162,18710,79958,169906,127982,17600,28119,222168,211865,8486,101892,70869,128909,255846,164685,101549,138117,227533,50490,152711,172791,228501,166220,220306,57816,241606,1751,95457,109462,227972,231626,136972,195765,1520,191619,108854,37227,133270,70983,104828,152145,184882,188648,54343,144036,174640,22508,15792,231443,127339,10664
# 9,192608,133857,152098,250052,236730,201099,177068,57910,143767,164542,210901,22021,36752,240380,142544,235274,140509,151691,170678,261336,57811,14269,195649,150654,182597,130398,37471,92402,659,141208,112041,55922,88677,37592,31682,13510,105267,124160,78742,160394,193668,44106,14995,213695,172809,11203,50269,85273,135796,241665,31248,182600,224042,32352,237649,10740,31247,231413,90400