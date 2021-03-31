
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
import seaborn as sb
def collate_fn(batch):
    return batch


class HistogramModel():
    def __init__(self, root, num_bins,transforms,k=500,target_dir="",load_file="") -> None:

        self.root = root

        self.num_bins = num_bins

        self.k=k

        self.target_dir = target_dir
        self.load_file= load_file
        self.filepaths = []
        self.transforms = transforms
        self.index_to_ad_id = {}
        self.populate_filepaths()

        self.filepaths  = self.filepaths
        print(self.filepaths[0])
        print(len(self.filepaths))
        self.data_loader = FaissDataLoader(self.filepaths,self.transforms,batch_size=256,num_workers=1,collate_fn=collate_fn)
        print(f"test set size: {len(self.data_loader.dataset)} images from {len(set(self.index_to_ad_id.values()))} Ads.")
        self.embedding_vectors = None

        
        self.initialize_faiss_index()
        self.vec_index_to_ad_id_func = np.vectorize(self.index_to_ad_id_func)
        
    def index_to_ad_id_func(self, idx):
        return self.index_to_ad_id[idx]

    def initialize_faiss_index(self):
        # res = faiss.StandardGpuResources()
        self.index = faiss.IndexFlatIP(129)
        # self.index = faiss.index_cpu_to_gpu(res,0,self.index)

    def embed(self):
        flag = False
        for batch in tqdm(self.data_loader):
            if flag == False:
                self.embedding_vectors = self.embed_batch(batch)
                flag = True
            else:
                flag = True
                self.embedding_vectors = np.vstack([self.embedding_vectors,self.embed_batch(batch)])
        self.embedding_vectors = self.embedding_vectors.astype(np.float32)
        with open(join(self.target_dir,'big_ass_hist_embeddings_on_seg.npz'), 'wb') as f:
                np.save(f,self.embedding_vectors)

    def load_embeddings(self):
        with open(self.load_file, 'rb') as f:
            self.embedding_vectors = np.load(f)
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
        np.savetxt(join(self.target_dir,"D.out"),self.D,delimiter=",")
        np.savetxt("I.out",self.I.astype(np.int),delimiter=",",fmt="%i")
        with open(join(self.target_dir,"filenames.txt"),"w") as f:
            f.writelines([x+ "\n" for x in self.filepaths])
        with open(join(self.target_dir,"vectors.npz"),"wb") as f:
            np.save(f,self.embedding_vectors)
        #TODO: Calculate top-5 accuracy etc. from index matrix and class dictionaries       
        self.A = self.vec_index_to_ad_id_func(idx=self.I)
        self.A_proper = self.A.copy()
        # Subtract query id from all columns. 
        # If there is a match, the jth column with have a zero in it
        self.A[:,1:] -= self.A[:,0][:, None]
        k_accuracies = np.zeros(self.k)
        for k in range(1, self.k+1):
            k_accuracies[k-1]=(np.mean(np.count_nonzero(self.A[:,1:k]==0,axis=1)>=1))
        
        with open(join(self.target_dir,"k_accuracies.npz"),"wb") as f:
            np.save(f,k_accuracies)
        
        fig,ax = plt.subplots(1,1)

        ax.plot(np.arange(1, self.k+1), k_accuracies)
        plt.savefig(join(self.target_dir,"hist_seg.png"))

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
    def populate_filepaths(self):
        idx = 0
        print("here")
        for date in os.listdir(self.root):
            print(date)
            for hour in os.listdir(join(self.root,date)):
                for ad_id in os.listdir(join(self.root,date,hour)):
                    imgs = [x for x in os.listdir(join(self.root,date,hour,ad_id)) if x[-3:]=="jpg"]
                    # Check if valid ad
                    if len(imgs) > 1:
                        for img in imgs:
                            img_filepath = join(self.root,date,hour,ad_id,img)
                            self.filepaths.append(img_filepath)
                            self.index_to_ad_id[idx] = np.int32(ad_id)
                            idx += 1


if __name__ == "__main__":


    ad_ids = []
    with open("/scratch/GIT/BikeML/embeddings/carrera_crossfire_Ad_IDs.txt","r") as f:
        ad_ids = f.readlines()
    ad_ids = [int(x) for x in ad_ids]

    target_dir = "/scratch/GIT/BikeML/embeddings/results/hist_post_seg/"

    load_file = "/scratch/GIT/BikeML/embeddings/results/hist_post_seg/vectors.npz"

    # input_dir = "/data_raid/raw/test/"
    input_dir = "/scratch/datasets/detr_filtered/test/"

    model = HistogramModel(input_dir,43,ToTensor(),k=500,target_dir=target_dir,load_file=load_file)
    model.embed()
    # model.load_embeddings()
    model.evaluate()
    model.evaluate_specified_ad_ids(ad_ids)

    
