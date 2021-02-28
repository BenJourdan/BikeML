
import torchvision
import itertools
from torchvision.transforms import Resize,ToTensor
import numpy as np
import os
from os.path import join
import sys
sys.path.append("/scratch/GIT/BikeML")
from dataloaders.dataloader import BikeDataLoader
from dataloaders.dataloader import SquarePad,Id
from dataloaders.faiss_evaluator import FaissDataLoader
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import faiss
import matplotlib.pyplot as plt

def collate_fn(batch):
    return batch


class HistogramModel():
    def __init__(self, root, num_bins,transforms) -> None:

        self.root = root

        self.num_bins = num_bins



        self.filepaths = []
        self.transforms = transforms
        self.populate_filepaths()

        self.filepaths  = self.filepaths
        self.data_loader = FaissDataLoader(self.filepaths,self.transforms,batch_size=256,num_workers=28,collate_fn=collate_fn)

        self.embedding_vectors = None

        self.initialize_faiss_index()

    def initialize_faiss_index(self):
        res = faiss.StandardGpuResources()
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.index_cpu_to_gpu(res,0,self.index)

    def embed(self):
        flag = False
        for batch in tqdm(self.data_loader):
            if flag == False:
                self.embedding_vectors = self.embed_batch(batch)
                flag = True
            else:
                flag = True
                self.embedding_vectors = np.vstack([self.embedding_vectors,self.embed_batch(batch)])
        with open('/scratch/GIT/BikeML/embeddings/big_ass_hist_embeddings.npz', 'wb') as f:
                np.save(f,self.embedding_vectors)

    def load_embeddings(self):
        with open('/scratch/GIT/BikeML/embeddings/big_ass_hist_embeddings.npz', 'wb') as f:
            self.embedding_vectors = self.np.load(f)
    @staticmethod
    def not_a_lambda(i,batch,bins):
        np.array([np.histogram(batch[i][k,:,:], bins=bins)[0] for k in range(3)]).reshape(-1)

    def embed_batch(self,batch):
        
        embeddings =  np.array(list(map(lambda i: np.array([np.histogram(batch[i][k,:,:], bins=self.num_bins)[0] for k in range(3)]).reshape(-1),range(len(batch)))))
        # with Pool(28) as p:
        #     embeddings =  np.array(list(p.map(partial(HistogramModel.not_a_lambda,batch=batch,bins=self.num_bins),range(len(batch)))))

        return embeddings

    def evaluate(self):
        faiss.normalize_L2(self.embedding_vectors)
        self.index.add(self.embedding_vectors)
        self.D, self.I = self.index.search(self.embedding_vectors, self.k)
        self.I = self.I.astype(np.int32)
        np.savetxt("D_hist.out",self.D,delimiter=",")
        np.savetxt("I_hist.out",self.I.astype(np.int),delimiter=",",fmt="%i")
        with open("filenames_hist.txt","w") as f:
            f.writelines([x+ "\n" for x in self.filepaths])
        with open("vectors_hist.npz","wb") as f:
            np.save(f,self.embedding_vectors)
        #TODO: Calculate top-5 accuracy etc. from index matrix and class dictionaries       
        self.A = self.vec_index_to_ad_id_func(idx=self.I)

        # Subtract query id from all columns. 
        # If there is a match, the jth column with have a zero in it
        self.A[:,1:] -= self.A[:,0][:, None]
        k_accuracies = np.zeros(self.k)
        for k in range(1, self.k+1):
            k_accuracies[k-1]=(np.mean(np.count_nonzero(self.A[:,1:k]==0,axis=1)>=1))
        
        with open("k_accuracies.npz","wb") as f:
            np.save(f,k_accuracies)
        
        fig,ax = plt.subplots(1,1)

        ax.plot(np.arange(1, self.k+1), k_accuracies)
        plt.show()

    def populate_filepaths(self):
        idx = 0
        for date in os.listdir(self.root):
            for hour in os.listdir(join(self.root,date)):
                for ad_id in os.listdir(join(self.root,date,hour)):
                    imgs = [x for x in os.listdir(join(self.root,date,hour,ad_id)) if x[-3:]=="jpg"]
                    # Check if valid ad
                    if len(imgs) > 1:
                        for img in imgs:
                            img_filepath = join(self.root,date,hour,ad_id,img)
                            self.filepaths.append(img_filepath)
                            idx += 1


if __name__ == "__main__":
    model = HistogramModel("/scratch/datasets/raw/test",43,ToTensor())
    model.embed()

    
