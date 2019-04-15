import os
from tqdm import tqdm 
import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import numpy as np 

from model.dem import DEM, DEMLoss, Predictor, DEMSupportLoss
from dataloader import TranditionalTrainDatasetloader
from logger import Log as log
import config
from utils import adjust_learning_rate, DataUtils

class PARAM():
    batch_size =32
    lr = 0.001
    lr_schedule = [10]
    base_model = 'densenet'# 'darknet'#

def train_dem():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--epoch', help='epoch num', default=10)
    # parser.add_argument('--feature_file', help='the file save generated features')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    feature_model = torch.load(args.load)
    feature_embedding = list(feature_model.module.children())[0]
    feature_embedding = nn.DataParallel(feature_embedding).cuda()
    feature_embedding.eval()

    dem_model = DEM()
    dem_model = nn.DataParallel(dem_model).cuda()
    criterion = DEMLoss().cuda()#DEMSupportLoss().cuda()#
    predictor = Predictor()
    
    optimizer = torch.optim.Adam(dem_model.parameters(), lr= PARAM.lr, weight_decay=config.weight_decay)


    train_dataset = TranditionalTrainDatasetloader('train')
    train_label_map = train_dataset.get_train_label_map()
    train_embeddings, train_attributes, train_word2vec_embeddings = train_dataset.get_class_features()
    train_embeddings = torch.Tensor(train_embeddings).cuda().float()
    train_attributes = torch.Tensor(train_attributes).cuda().float()
    train_word2vec_embeddings = torch.Tensor(train_word2vec_embeddings).cuda().float()
    trainloader = DataLoader(\
        dataset=train_dataset,\
            batch_size=PARAM.batch_size, num_workers=config.num_workers, shuffle=True )

    val_dataset = TranditionalTrainDatasetloader('val')
    val_label_map = val_dataset.get_train_label_map()
    val_embeddings, val_attributes, val_word2vec_embeddings = val_dataset.get_class_features()
    val_embeddings = torch.Tensor(val_embeddings).cuda().float()
    val_attributes = torch.Tensor(val_attributes).cuda().float()
    val_word2vec_embeddings = torch.Tensor(val_word2vec_embeddings).cuda().float()
    valloader = DataLoader(\
        dataset=val_dataset,\
            batch_size=PARAM.batch_size, num_workers=config.num_workers, shuffle=True )

    with tqdm(total = args.epoch) as pbar:
        for p in range(args.epoch):
            train_epoch_loss = []
            train_epoch_acc = []
            val_epoch_loss = []
            val_epoch_acc = []
            val_acc = 0.0

            adjust_learning_rate(optimizer, PARAM.lr, PARAM.lr_schedule, epoch=p, gamma=0.1)
            dem_model.train()
            train_acc = 0.0
            with tqdm(total = len(trainloader)) as bbar:
                for b, (image, label) in enumerate(trainloader):
                    
                    tlabel = list()
                    for i in range(len(label)):
                        tlabel.append(train_label_map[int(label[i])])

                    tlabel = torch.LongTensor(tlabel)
                    image = image.cuda().float()
                    trans_embeddings = dem_model(train_word2vec_embeddings)#train_embeddings, train_attributes
                    
                    image_feature = feature_embedding(image)
                    image_feature_label = image_feature.detach()
                    
                    loss = criterion(trans_embeddings[tlabel], image_feature_label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_acc = predictor(trans_embeddings, image_feature_label, tlabel)
                    train_acc += batch_acc
                
                    train_epoch_loss.append(loss)
                    bbar.set_description('train_batch_loss: {}, train_batch_acc: {}'.format(loss, batch_acc))
                    bbar.update(1)
            train_acc = train_acc / len(trainloader)

            with torch.no_grad():
                dem_model.eval()
                
                trans_embeddings = dem_model(val_word2vec_embeddings)#val_embeddings  val_attributes
                print("VAL:", len(valloader))
                acc = 0.0
                with tqdm(total = len(valloader)) as vbar:
                    for b, (image, label) in enumerate(valloader):
                        tlabel = list()
                        for i in range(len(label)):
                            tlabel.append(val_label_map[int(label[i])])
                        tlabel = torch.LongTensor(tlabel)

                        image = image.cuda().float()

                        image_feature = feature_embedding(image)
                        image_feature_label = image_feature.detach()
                        batch_acc = predictor(trans_embeddings, image_feature_label, tlabel)
                        acc += batch_acc

                        loss = criterion(trans_embeddings[tlabel], image_feature_label)

                        val_epoch_loss.append(loss)
                        vbar.set_description('val_batch_loss: {}, val_batch_acc:{}'.format(loss, batch_acc))
                        vbar.update(1)
                val_acc = acc / len(valloader)
            log.logger.info('*******************Train Epoch {}*************'.format(p))
            log.logger.info('train_loss:{}, train_acc:{}, val_loss:{}, val_acc:{}'.\
                    format(torch.mean(torch.tensor(train_epoch_loss)), train_acc, torch.mean(torch.tensor(val_epoch_loss)), val_acc))

            # torch.save(dem_model, os.path.join(config.model_dir, 'dem_model_{}.pth'.format(p)))
            pbar.update(1)
            
if __name__ == '__main__':
    print('train dem.py')
    train_dem()
    