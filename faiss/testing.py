"""
This script lets you time how long it takes to insert and querry data into a Faiss data structure. I'd suggest reading through the wiki:
https://github.com/facebookresearch/faiss/wiki

It's pretty cool.

Anyway if we can get our embedding down to 2048 (a compression factor of 96 given 256x256x3 images) then 600K images takes roughly 9GB on the GPU
without any preemptive fuckery (aka Quantitizers). A querry with nprobe=10 should take roughly 130 seconds for 100K querry points... Pretty damm fast lol.

"""

import numpy as np
import faiss
import time

res = faiss.StandardGpuResources()  # use a single GPU... lol

d = 2048                           # dimension
nb = 600000                      # database size
nq = 100000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

nlist = 100
k = 4
quantizer = faiss.IndexFlatL2(d)  # the other index
gpu=True

index = faiss.IndexIVFFlat(quantizer, d, nlist,faiss.METRIC_INNER_PRODUCT)
if gpu==True:
    index = faiss.index_cpu_to_gpu(res,0,index)
assert not index.is_trained
print("start training")
t= time.time()
faiss.normalize_L2(xb)
index.train(xb)
print(f"training finished in {time.time()-t}")
assert index.is_trained

print("adding data")
t = time.time()
index.add(xb)                  # add may be a bit slower as well
print(f"finished adding in {time.time()-t}")

print("starting querry")
t = time.time()
faiss.normalize_L2(xq)
D, I = index.search(xq, k)     # actual search
print(f"finished querrying with nprobe=1 in {time.time()-t}")
print(I[-5:])                  # neighbors of the 5 last queries
index.nprobe = 10              # default nprobe is 1, try a few more

print("starting querry with nprobe = 10")
t = time.time()
D, I = index.search(xq, k)
print(I[-5:])                  # neighbors of the 5 last queries

print(f"finished querrying with nprobe=10 in {time.time()-t}")
