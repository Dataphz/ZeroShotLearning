import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch
import numpy as np 
import math

import config
from model.densenet import DenseNet
from model.darknet import Darknet19

class DistanceNetwork(nn.Module):
    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_features, query_features):
        """
        support_features: bs x way x shot x feature_len
        query_features: bs x query_num*way  x feature_len
        """
        # eps = 1e-10
        similarities = []   
        prototype_features = torch.mean(support_features, dim=2) # bs x way x feature_len
        for q in range(query_features.size(1)):
            distance_features = prototype_features - query_features[:, q:q+1, :]
            euclidean_similarity = torch.mean(-1 * torch.pow(distance_features, 2), dim=2)# bs x way
            max_similarity, _ = torch.max(euclidean_similarity, dim=1,keepdim=True)
            euclidean_similarity = euclidean_similarity -  max_similarity# normailze, bs x way
            similarities.append(euclidean_similarity)
        similarities = torch.transpose(torch.stack(similarities), 0, 1)#bs x query_num x way
        return similarities

class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc, self).__init__()

    def forward(self, similarities, query_set_y):
        """
        Input:
        similarities: batch_size x query_num x way
        query_set_y: batch_size x query_num
        """
        
        preds = torch.argmax(similarities, dim=2, keepdim=True) # batch_size
        pred_bools = torch.eq(preds.int(), query_set_y.int()) 
        acc = torch.mean(pred_bools.float())
        softmax_similarities = -1 * F.log_softmax(similarities, 1)
        gt_ind = query_set_y.view(query_set_y.size(0), query_set_y.size(1), 1)

        target_softmax_similarities = torch.gather(softmax_similarities, dim=2, index=gt_ind.long())
        loss = torch.mean(target_softmax_similarities)
        return acc, loss

class PrototypeNetwork(nn.Module):
    def __init__(self, way, shot, query_num=1, basemodel='densenet'):
        super(PrototypeNetwork, self).__init__()

        self.way = way
        self.shot = shot
        assert query_num == 1, 'query_num: {}'.format(query_num)
        self.query_num = query_num
        if basemodel == 'densenet':
            self.embedding_model = DenseNet(growth_rate=12, drop_rate=0.0, require_feature=True)
        elif basemodel == 'darknet':
            self.embedding_model = Darknet19(cls_num=config.image_feature_num)
        self.dn = DistanceNetwork()

    def forward(self, support_set_x, support_set_y, query_set_x):
        #embedding
        support_embedding_features = []
        query_embedding_features = []

        batch_size, _, _, c, h, w, = support_set_x.size()
        support_set_x = support_set_x.view(-1, c, h, w)
        query_set_x = query_set_x.view(-1, c, h, w)
        
        support_embedding_features = self.embedding_model(support_set_x)
        query_embedding_features = self.embedding_model(query_set_x)
        support_embedding_features = support_embedding_features.view(batch_size, self.way, self.shot, -1)
        query_embedding_features = query_embedding_features.view(batch_size, self.way, -1)

        similarities = self.dn(support_embedding_features, query_embedding_features)
        return similarities


        
            