import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from textwrap import wrap

folders = ["0_01","0_05","0_1","0_9","euclid","hist_post_seg","hist_pre_seg","model_pre_seg"]


pos = "histogram"
neg = "NN"
hist_types = {
    "0_01":neg,
    "0_05":neg,
    "0_1": neg,
    "0_9": neg,
    "euclid": neg,
    "hist_post_seg" : pos,
    "hist_pre_seg" : pos,
    "model_pre_seg" : neg
}

criterion_types = {
            "0_01":0,
    "0_05":0,
    "0_1": 0,
    "0_9": 0,
    "euclid": 1,
    "hist_post_seg" : 2,
    "hist_pre_seg" : 2,
    "model_pre_seg" : 0
}

names = {
    "0_01":"cosine: m = 0.01",
    "0_05":"cosine: m = 0.05",
    "0_1": "cosine: m = 0.1",
    "0_9": "cosine: m = 0.9",
    "euclid": "euclidean: m = 0.2",
    "hist_post_seg" : "histogram baseline post-segmentation",
    "hist_pre_seg" : "histogram baseline pre-segmentation",
    "model_pre_seg" : "euclidean: m = 0.2 pre-segmentation"
}

palette = {    
    "cosine: m = 0.01":"#336699",
   "cosine: m = 0.05":"#4da6ff",
    "cosine: m = 0.1": "#00ffff",
    "cosine: m = 0.9": "#000099",
    "euclidean: m = 0.2": "#000000",
    "histogram baseline post-segmentation" : "#ff3300",
    "histogram baseline pre-segmentation" : "#ff6666",
    "euclidean: m = 0.2 pre-segmentation" : "#33cc33"}



for key in list(palette.keys()):
    print(key)
    new_key = '\n'.join(wrap(key, 20))
    print(new_key)
    print()
    if new_key!=key:
        palette[new_key] = palette[key]
        del palette[key]

print(palette.keys())
for key in names.keys():
    names[key] = '\n'.join(wrap(names[key], 20))



vecs = []

fig,ax = plt.subplots(1,1,figsize=(7,7))
for folder in folders:

    with open(join(folder,"k_accuracies.npz"),"rb") as f:
        accs = np.load(f)
        vecs.append(accs)


        # ax.plot(accs,label=names[folder])

arr = np.array(vecs).T
print(arr.shape)

data = arr.flatten("F")
print(data)
indices = np.arange(1,501)
indices = np.hstack([indices]*len(folders))

hist_labels = []
for folder in folders:
    hist_labels+=[hist_types[folder]]*500

criterion_labels = []
for folder in folders:
    criterion_labels+=[criterion_types[folder]]*500

name_labels = []
for folder in folders:
    name_labels+=[names[folder]]*500

df = pd.DataFrame({"k":indices,"data":data,"experiment":name_labels,"criterion":criterion_labels,"model type":hist_labels})


sns.lineplot(data=df,x="k",y="data",hue="experiment",style="model type",palette=palette,ax=ax,legend="brief")




ax.set_xlabel("k",size="large")
ax.set_ylabel("Top k accuracy",size="large")
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.25,
                 box.width, box.height * 0.9])
leg = plt.legend(loc="center",bbox_to_anchor=(0.5,-0.25),ncol=3,fancybox=False,fontsize="large",frameon=False)
for line, text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
plt.grid(axis="y",which="major")
sns.despine()
plt.show()



fig,ax = plt.subplots(1,1,figsize=(7,7))
dfs = []
for folder in folders:

    with open(join(folder,"evaluation_matrix.npz"),"rb") as f:
        eval_matrix = np.load(f)
        print(eval_matrix.shape)
        df = pd.DataFrame(eval_matrix).melt(var_name="k",value_name="data")
        df["experiment"] = names[folder]
        df["model type"] = hist_types[folder]
        df["criterion"] = criterion_types[folder]
        dfs.append(df)
        # sns.lineplot(x="variable", y="value", data=df,label=names[folder])
        # ax.plot(accs,label=folder)

df = pd.concat(dfs).reset_index(drop=True)
print(df)
sns.lineplot(data=df,x="k",y="data",hue="experiment",style="model type",palette=palette,ax=ax,legend="brief")

ax.set_xlabel("k",size="large")
ax.set_ylabel("hits",size="large")
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.25,
                 box.width, box.height * 0.9])
leg = plt.legend(loc="center",bbox_to_anchor=(0.5,-0.25),ncol=3,fancybox=False,fontsize="large",frameon=False)
for line, text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
plt.grid(axis="y",which="major")
sns.despine()
plt.show()

# root = "/data_raid/raw/test/"

# print(root)
# idx = 0
# ads = 0
# for date in os.listdir(root):
#     for hour in os.listdir(join(root,date)):
#         for ad_id in os.listdir(join(root,date,hour)):
#             imgs = [x for x in os.listdir(join(root,date,hour,ad_id)) if x[-3:]=="jpg"]
#             # Check if valid ad
#             if len(imgs) > 1:
#                 ads+=1

# print(ads)


#Pre seg:
# 264555 test images in pre seg dataset
# 2944 Crossfire images in pre seg test dataset

 #Post seg:
# 165780 test images in post seg dataset
# 2141 crossfire images in post seg test dataset

