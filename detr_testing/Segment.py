from PIL import Image
from tqdm import tqdm
import io
import math
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.transforms import ToTensor
import os
from os.path import join
from pytorch_memlab import MemReporter

torch.set_grad_enabled(False)
import numpy as np
import panopticapi
from panopticapi.utils import id2rgb, rgb2id
import torchvision.transforms as T


class Segmentor():
    def __init__(self,keep=0.9):

        self.keep = keep
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
        self.model.eval()
        self.model.to(self.device)
        
        
    @staticmethod
    def extract_bboxes(mask):
        """
        Compute bounding boxes from masks.
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
    
    def remove_background(self, image, transforms = T.Compose([
                                                    T.Resize(800),
                                                    T.ToTensor(),
                                                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                    ])):
        torch.cuda.empty_cache()
        image_normalized = transforms(image).unsqueeze(0) # unsqueeze to add artificial first dimension
        image_normalized = image_normalized.to(self.device)
        segment = self.model(image_normalized)
        image = T.ToTensor()(image)
        result = self.postprocessor(segment, torch.as_tensor(image.shape[-2:]).unsqueeze(0))[0]

        # We extract the segments info and the panoptic result from DETR's prediction
        segments_info = deepcopy(result["segments_info"])


        # compute the scores, excluding the "no-object" class (the last one)
        scores = segment["pred_logits"].softmax(-1)[..., :].max(-1)[0]

        do_we_keep = (scores >self.keep).squeeze()

        # Panoptic predictions are stored in a special format png
        panoptic_seg = Image.open(io.BytesIO(result['png_string']))        
        # We convert the png into an segment id map
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))[None, :].int()
        panoptic_seg = panoptic_seg.cpu()
        
        bike_mask_ids = [segments_info[i]["id"] for i in range(len(segments_info)) if (segments_info[i]["category_id"] == 2) and (do_we_keep[segments_info[i]["id"]]==True) ]
        # Return None if either no bike, or more than one bike is found in the image
        if len(bike_mask_ids) != 1:
            return None
        # Else return the cropped, masked bike image
        else:
            mask_id = bike_mask_ids[0]    

            masked_img = (panoptic_seg == mask_id) * image.cpu()
            mask = (panoptic_seg == mask_id).numpy()
            #Extract bounding box
            boxes = self.extract_bboxes(mask[0].T[:,:,None])[0].tolist()
            y1, x1, y2, x2 = boxes
            cropped_masked_img = masked_img[:, x1:x2, y1:y2].T.transpose(dim0=0, dim1=1)
            return torch.squeeze(cropped_masked_img)

   
if __name__ == '__main__':

    raw_root = "/data_raid/raw/test"

    target_root = "/scratch/datasets/detr_filtered/test"

    # im = Image.open("/scratch/datasets/raw/train/2020-10-22/01/1387969394/XdEAAOSwS2dfkNAV.jpg")    
    reporter = MemReporter()
    segmentor = Segmentor(keep=0.85)


    dates = sorted(os.listdir(raw_root))
    
    print(dates)
    first_time_flag = True
    fist_date = "2020-11-28"
    first_hour = "12"
    first_ad = "1391315313"

    dates = dates[dates.index(fist_date):]

    with torch.no_grad():
        with tqdm(total=len(dates),ncols=120) as pbar_segment:
            for date in dates:
                hours = sorted(os.listdir(join(raw_root,date)))
                if first_time_flag == True:
                    hours = hours[hours.index(first_hour):]
                    first_time_flag = False
                for hour in hours:
                    ads = sorted(os.listdir(join(raw_root,date,hour)))
                    # print(ads[0],ads[-1])
                    # if first_time_flag == True:
                    #     ads = ads[ads.index(first_ad):]
                    #     first_time_flag = False
                    # print(ads)
                    for ad_id in ads:
                        imgs = [x for x in os.listdir(join(raw_root,date,hour,ad_id)) if x[-3:]=="jpg"]
                        # Check if valid ad
                        if len(imgs) > 1:                      
                            for img in imgs:
                                
                                target_path = join(target_root,date,hour,ad_id,img)
                                if os.path.isfile(target_path):
                                    continue
                                im = Image.open(join(raw_root,date,hour,ad_id,img)).convert("RGB")

                                try:
                                    output = segmentor.remove_background(im)
                                    if output!=None:
                                        os.makedirs(join(target_root,date,hour,ad_id),exist_ok=True)
                                        # plt.imshow(output)
                                        # plt.show()
                                    
                                        PIL_image = Image.fromarray((output.numpy()*255).astype(np.uint8))
                                        PIL_image.save(target_path)

                                except Exception as e:
                                    print(f"we don't like skinny bois: {join(raw_root,date,hour,ad_id,img)}")
                                    continue

                                pbar_segment.set_description(f" date: {date} hour: {hour} Ad: {ad_id} Im: {img}")
                pbar_segment.update(1)