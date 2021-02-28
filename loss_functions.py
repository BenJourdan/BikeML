"""
Adapted from https://github.com/adambielski/siamese-triplet
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import functional
import numpy as np
from torch.nn import CosineSimilarity

class SupervisedContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin, **kwargs):
        super(SupervisedContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    #adapted....
    def forward(self, output1, output2, target):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 - target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean()

    @staticmethod
    def embedding_dist(output1, output2):
        return (output2 - output1).pow(2).sum(1).mean()
        

class SupervisedCosineContrastiveLoss(nn.Module):
    """
    Cosine Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin, **kwargs):
        super(SupervisedCosineContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_metric = CosineSimilarity()
    def forward(self, output1, output2, target):
        return F.cosine_embedding_loss(output1, output2, target*2-1.0, margin=self.margin)

    def embedding_dist(self,output1, output2):
        return (1.0 - self.distance_metric(output1, output2)).mean()


        # print(output1.shape)
        # print(output2.shape)
        # print(output1.dtype)
        # print(output2.dtype)
        # distances = 1-self.cos(output1,output2)
        # print(distances)
        
        # losses = 0.5 * (target.float() * distances.pow +
        #                 (1 - target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        # mean = losses.mean()
        # print(mean)
        # return losses.mean()

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# class OnlineContrastiveLoss(nn.Module):
#     """ 
#     Online Contrastive loss
#     Takes a batch of embeddings and corresponding labels.
#     Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
#     and negative pairs
#     """

#     def __init__(self, margin, pair_selector):
#         super(OnlineContrastiveLoss, self).__init__()
#         self.margin = margin
#         self.pair_selector = pair_selector

#     def forward(self, embeddings, target):
#         positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
#         if embeddings.is_cuda:
#             positive_pairs = positive_pairs.cuda()
#             negative_pairs = negative_pairs.cuda()
#         positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
#         negative_loss = F.relu(
#             self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
#                 1).sqrt()).pow(2)
#         loss = torch.cat([positive_loss, negative_loss], dim=0)
#         return loss.mean()


# class OnlineTripletLoss(nn.Module):
#     """
#     Online Triplets loss
#     Takes a batch of embeddings and corresponding labels.
#     Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
#     triplets
#     """

#     def __init__(self, margin, triplet_selector):
#         super(OnlineTripletLoss, self).__init__()
#         self.margin = margin
#         self.triplet_selector = triplet_selector

#     def forward(self, embeddings, target):

#         triplets = self.triplet_selector.get_triplets(embeddings, target)

#         if embeddings.is_cuda:
#             triplets = triplets.cuda()

#         ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
#         an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
#         losses = F.relu(ap_distances - an_distances + self.margin)

#         return losses.mean(), len(triplets)