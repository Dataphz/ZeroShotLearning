import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from utils import l2_similarity, cosine_similarity
import config
class DEMLoss(nn.Module):
    def __init__(self):
        super(DEMLoss, self).__init__()

    def forward(self, attribute_features, image_features):
        """
        attribute_features:  bs x fl
        image_features: bx x fl
        """
        attribute_features = attribute_features / torch.sqrt(torch.sum(torch.pow(attribute_features, 2), dim=-1, keepdim=True))
        image_features = image_features / torch.sqrt(torch.sum(torch.pow(image_features, 2), dim=1, keepdim=True))
        distance = attribute_features - image_features
        loss = torch.sum(torch.pow(distance, 2))
        return loss

class DEMSupportLoss(nn.Module):
    def __init__(self, similarity='cosine'):
        super(DEMSupportLoss, self).__init__()
        if similarity == 'cosine':
            self.similarity = cosine_similarity
        elif similarity == 'l2':
            self.similarity = l2_similarity
        self.loss = nn.CrossEntropyLoss()

    def forward(self, attribute_features, image_features, label):
        similarity = self.similarity(attribute_features, image_features)
        loss = self.loss(similarity.cpu(), label)
        return loss 

class Predictor(nn.Module):
    def __init__(self, similarity='l2'):
        super(Predictor, self).__init__()
        if similarity == 'cosine':
            self.similarity = cosine_similarity
        elif similarity == 'l2':
            self.similarity = l2_similarity

    def forward(self, trans_attributes, image_features, label):
        """
        Input:
            attribute_features:  A x l
            image_features: I x l
            label: I x 1 
        Return:
            acc
        """
        similarity = self.similarity(trans_attributes, image_features)
        pred = torch.argmax(similarity, dim = 1)
        # print(pred)
        # print(label)
        acc = torch.eq(pred.cpu(), label).float().mean()
        return acc

class DEM(nn.Module):
    def __init__(self,):
        super(DEM, self).__init__()

        # self.class2img_features = nn.Sequential(
        #         nn.Linear(config.class_attribute_num, 1024),
        #         nn.ReLU(),
        #         nn.Linear(1024, config.image_feature_num),
        #         nn.ReLU()
        #     )
        
        self.attribute2img_features = nn.Sequential(
                nn.Linear(config.defined_attribute_num, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),# 512, 1024
                nn.ReLU()
                )

        self.word2vec_features = nn.Sequential(
                nn.Linear(config.class_attribute_num, 1024),
                nn.ReLU(),
                nn.Linear(1024, config.image_feature_num),
                nn.ReLU()
            )

        self.fused_features = nn.Sequential(
            nn.Linear(1024+config.image_feature_num, config.image_feature_num),
            nn.ReLU()
        )

        for name, param in self.named_parameters():
            if 'linear' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))

    def forward(self, word2vec_feature):#attribute_feature
        """
        hyper-parameters acquire from original papers. 
        "Learning a Deep Embedding Model for Zero-Shot Learning"
        """
        # class2image = self.class2img_features(class_feature)
        # attribute2image = self.attribute2img_features(attribute_feature)
        word2vec2image = self.word2vec_features(word2vec_feature)
        # fused_feature = torch.cat((word2vec2image, attribute2image), 1)
        # fused_image = self.fused_features(fused_feature)
        # fused_image_temp = (class2image + attribute2image) * 2 / 3.0
        # fused_image = 1.7159 * torch.tanh(fused_image_temp)

        return word2vec2image#fused_image


