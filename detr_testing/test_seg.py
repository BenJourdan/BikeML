from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)
import numpy as np
import panopticapi
from panopticapi.utils import id2rgb, rgb2id
import cv2


# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog
# from google.colab.patches import cv2_imshow

# These are the COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Detectron2 uses a different numbering scheme, we build a conversion table
coco2d2 = {}
count = 0
for i, c in enumerate(CLASSES):
  if c != "N/A":
    coco2d2[i] = count
    count+=1

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
model.eval()

model.to(device)

url = "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/allbikes-1539286251.jpg?crop=0.659xw:1.00xh;0.317xw,0&resize=640:*"
im = Image.open(requests.get(url, stream=True).raw)

im = Image.open("/scratch/datasets/raw/train/2020-10-22/01/1387969394/XdEAAOSwS2dfkNAV.jpg")



# mean-std normalize the input image (batch-size: 1)
tensify_and_resize = T.Compose([T.Resize(800),T.ToTensor()])

raw_image = tensify_and_resize(im)
img = transform(im).unsqueeze(0)
img = img.to(device)
out = model(img)

# compute the scores, excluding the "no-object" class (the last one)
scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
# threshold the confidence
keep = scores > 0.85

# Plot all the remaining masks
ncols = 5
# fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(keep.sum().item() / ncols), figsize=(18, 10))
# for line in axs:
#     for a in line:
#         a.axis('off')
# for i, mask in enumerate(out["pred_masks"][keep]):
#     ax = axs[i // ncols, i % ncols]
#     # ax.imshow(mask.cpu(), cmap="cividis")
#     ax.axis('off')
# fig.tight_layout()
# print()
# plt.show()
result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

import itertools
import seaborn as sns
palette = itertools.cycle(sns.color_palette())

# The segmentation is stored in a special-format png
panoptic_seg = Image.open(io.BytesIO(result['png_string']))
panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()
# We retrieve the ids corresponding to each mask
panoptic_seg_id = rgb2id(panoptic_seg)

# Finally we color each mask individually
panoptic_seg[:, :, :] = 0
for id in range(panoptic_seg_id.max() + 1):
  panoptic_seg[panoptic_seg_id == id] = np.asarray(next(palette)) * 255
# plt.figure(figsize=(15,15))
# plt.imshow(panoptic_seg)
# plt.axis('off')
# plt.show()


from copy import deepcopy
# We extract the segments info and the panoptic result from DETR's prediction
segments_info = deepcopy(result["segments_info"])
# Panoptic predictions are stored in a special format png
panoptic_seg = Image.open(io.BytesIO(result['png_string']))
final_w, final_h = panoptic_seg.size
# We convert the png into an segment id map
panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))[None, :].int()

panoptic_seg.cpu()



transform = T.ToTensor()

resized_img = transform(im.resize((800,1071)))



print((panoptic_seg == 0).shape)


mask_ids = [segments_info[i]["id"] for i in range(len(segments_info)) if segments_info[i]["category_id"] == 2]

def extract_bboxes(mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)
    
for idx in mask_ids:
    print(idx)
    
    test = (panoptic_seg == idx) * raw_image
    mask = (panoptic_seg == idx).numpy()

    print(mask[0].T[:,:,None].shape)
    boxes = extract_bboxes(mask[0].T[:,:,None])[0].tolist()
    print(boxes)
    y1,x1,y2,x2 =boxes

    print(mask.shape)
    print(raw_image.shape)

    plt.imshow(test[:,x1:x2, y1:y2].T.transpose(dim0=0, dim1=1))
    plt.show()

