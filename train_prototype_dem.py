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


def train_dem():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--epoch', help='epoch num', default=50)
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
    
    optimizer = torch.optim.Adam(dem_model.parameters(), lr= config.lr, weight_decay=config.weight_decay)


    if not config.environ == 'all':
        train_dataset = TranditionalTrainDatasetloader('train')
        train_label_map = train_dataset.get_train_label_map()
        train_embeddings = train_dataset.get_embeddings()
        train_embeddings = torch.Tensor(train_embeddings).cuda().float()
        trainloader = DataLoader(\
            dataset=train_dataset,\
                batch_size=config.batch_size_dem, num_workers=config.num_workers, shuffle=True )

        val_dataset = TranditionalTrainDatasetloader('val')
        val_label_map = val_dataset.get_train_label_map()
        val_embeddings = val_dataset.get_embeddings()
        val_embeddings = torch.Tensor(val_embeddings).cuda().float()
        valloader = DataLoader(\
            dataset=val_dataset,\
                batch_size=config.batch_size_dem, num_workers=config.num_workers, shuffle=True )
    else:
        train_dataset = TranditionalTrainDatasetloader('train')
        train_label_map = train_dataset.get_train_label_map()
        train_embeddings = train_dataset.get_embeddings()
        train_embeddings = torch.Tensor(train_embeddings).cuda().float()

        trainloader = DataLoader(\
            dataset=train_dataset,\
                batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True )
    # open set       
    Datautils = DataUtils()
    embeddings, attributes = Datautils.load_embedding_and_attribute()
    embeddings = torch.Tensor(embeddings).cuda().float()
    attributes = torch.Tensor(attributes).cuda().float()

    with tqdm(total = args.epoch) as pbar:
        for p in range(args.epoch):
            train_epoch_loss = []
            train_epoch_acc = []
            val_epoch_loss = []
            val_epoch_acc = []
            val_acc = 0.0

            adjust_learning_rate(optimizer, epoch=p, gamma=0.1)
            dem_model.train()
            train_acc = 0.0
            with tqdm(total = len(trainloader)) as bbar:
                for b, (image, label) in enumerate(trainloader):
                    tlabel = list()
                    for i in range(len(label)):
                        tlabel.append(train_label_map[int(label[i])])
                    tlabel = torch.LongTensor(tlabel)
                    image = image.cuda().float()
                    trans_embeddings = dem_model(train_embeddings)
                    
                    image_feature = feature_embedding(image)
                    image_feature_label = image_feature.detach()
                    
                    loss = criterion(trans_embeddings[tlabel], image_feature_label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_acc, batch_acc_num = predictor(trans_embeddings, image_feature_label, tlabel)
                    train_acc += batch_acc
                
                    train_epoch_loss.append(loss)
                    bbar.set_description('train_batch_loss: {}, train_batch_acc: {}'.format(loss, batch_acc))
                    bbar.update(1)
            train_acc = train_acc / len(trainloader)

            if not config.environ == 'all':
                with torch.no_grad():
                    dem_model.eval()
                    
                    trans_embeddings = dem_model(val_embeddings)
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
                            batch_acc, batch_acc_num = predictor(trans_embeddings, image_feature_label, tlabel)
                            acc += batch_acc

                            loss = criterion(trans_embeddings[tlabel], image_feature_label)

                            val_epoch_loss.append(loss)
                            vbar.set_description('val_batch_loss: {}, val_batch_acc:{}'.format(loss, batch_acc))
                            vbar.update(1)
                    val_acc = acc / len(valloader)

            log.logger.info('*******************Train Epoch {}*************'.format(p))
            log.logger.info('train_loss:{}, train_acc:{}, val_loss:{}, val_acc:{}'.\
                        format(torch.mean(torch.tensor(train_epoch_loss)), train_acc, torch.mean(torch.tensor(val_epoch_loss)), val_acc))

            # if not config.environ == 'all' and p % 2 == 0:
            #     torch.save(model, os.path.join(config.model_dir, 'model_{}.pth'.format(p)))
            pbar.update(1)
            
if __name__ == '__main__':
    print('train dem.py')
    train_dem()
    