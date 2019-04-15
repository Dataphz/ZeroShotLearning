import numpy as np 
from sklearn.manifold import TSNE
import os
from tqdm import tqdm 
import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import numpy as np 
import pickle

from model.dem import DEM, DEMLoss, Predictor, DEMSupportLoss
from dataloader import TranditionalTrainDatasetloader
from logger import Log as log
import config
from utils import adjust_learning_rate, DataUtils

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
parser.add_argument('--load', help='load model')
parser.add_argument('--epoch', help='epoch num', default=1)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def generate_features():

    
    # parser.add_argument('--feature_file', help='the file save generated features')

    

    feature_model = torch.load(args.load)
    feature_embedding = list(feature_model.module.children())[0]
    feature_embedding = nn.DataParallel(feature_embedding).cuda()
    feature_embedding.eval()

    # dem_model = DEM()
    # dem_model = nn.DataParallel(dem_model).cuda()
    # criterion = DEMLoss().cuda()#DEMSupportLoss().cuda()#
    # predictor = Predictor()
    
    # optimizer = torch.optim.Adam(dem_model.parameters(), lr= config.lr, weight_decay=config.weight_decay)


    train_dataset = TranditionalTrainDatasetloader('train')
    train_label_map = train_dataset.get_train_label_map()
    train_embeddings, train_attributes, train_word2vec_embeddings = train_dataset.get_class_features()
    train_embeddings = torch.Tensor(train_embeddings).cuda().float()
    train_attributes = torch.Tensor(train_attributes).cuda().float()
    train_word2vec_embeddings = torch.Tensor(train_word2vec_embeddings).cuda().float()
    trainloader = DataLoader(\
        dataset=train_dataset,\
            batch_size=config.batch_size_dem, num_workers=config.num_workers, shuffle=True )

    val_dataset = TranditionalTrainDatasetloader('val')
    val_label_map = val_dataset.get_train_label_map()
    val_embeddings, val_attributes, val_word2vec_embeddings = val_dataset.get_class_features()
    val_embeddings = torch.Tensor(val_embeddings).cuda().float()
    val_attributes = torch.Tensor(val_attributes).cuda().float()
    val_word2vec_embeddings = torch.Tensor(val_word2vec_embeddings).cuda().float()
    valloader = DataLoader(\
        dataset=val_dataset,\
            batch_size=config.batch_size_dem, num_workers=config.num_workers, shuffle=True )
    features = {
        'train_features':[],
        'train_labels':[],
        'val_features':[],
        'val_labels':[]}

    train_epoch_loss = []
    train_epoch_acc = []
    val_epoch_loss = []
    val_epoch_acc = []
    val_acc = 0.0

    # adjust_learning_rate(optimizer, epoch=p, gamma=0.1)
    train_acc = 0.0
    with tqdm(total = len(trainloader)) as bbar:
        for b, (image, label) in enumerate(trainloader):
            
            tlabel = list()
            for i in range(len(label)):
                tlabel.append(train_label_map[int(label[i])])
            tlabel = torch.LongTensor(tlabel)
            image = image.cuda().float()
            
            image_feature = feature_embedding(image)
            # print(image_feature.size())
            features['train_features'].extend(image_feature.cpu().detach().numpy())
            features['train_labels'].extend(label.numpy())
            image_feature_label = image_feature.detach()

            bbar.update(1)

    with torch.no_grad():
        with tqdm(total = len(valloader)) as vbar:
            for b, (image, label) in enumerate(valloader):
                tlabel = list()
                for i in range(len(label)):
                    tlabel.append(val_label_map[int(label[i])])
                tlabel = torch.LongTensor(tlabel)
                image = image.cuda().float()

                image_feature = feature_embedding(image)
                features['val_features'].extend(image_feature.cpu().detach().numpy())
                features['val_labels'].extend(label.numpy())
                vbar.update(1)
    with open('{}_feature_embedding.pkl'.format(args.load), 'wb') as file:
        pickle.dump(features, file)

            


def tsne():
    """
    Input:
        n_components: 
        embedding: N x L
    """
    with open('{}_feature_embedding.pkl'.format(args.load), 'rb') as file:
        data = pickle.load(file)

    train_features = data['train_features']
    train_labels = data['train_labels']
    val_features = data['val_features']
    val_labels = data['val_labels']

    tsne_features = {
        'train_features':[],
        'train_labels':[],
        'val_features':[],
        'val_labels':[]}
    tsne_features['train_labels'] = train_labels
    tsne_features['val_labels'] = val_labels
    

    train_x_embedded = TSNE(n_components=2).fit_transform(train_features)# N x
    tsne_features['train_features'] = train_x_embedded
    val_x_embedded = TSNE(n_components=2).fit_transform(val_features)
    tsne_features['val_features'] = val_x_embedded
    name = args.load.split('/')
    with open('{}_tsne.pkl'.format(name[-2]), 'wb') as file:
        pickle.dump(tsne_features, file)

if __name__ == '__main__':
    # features = generate_features()
    tsne()