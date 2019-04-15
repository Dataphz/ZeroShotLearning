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
from logger import Logger
import config
from utils import adjust_learning_rate, DataUtils
# from tensorboard import Logger as tensorboardlog

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
parser.add_argument('--load', help='load model')
parser.add_argument('--epoch', help='epoch num', default=10)

class PARAM():
    batch_size =32
    lr = 0.1
    lr_schedule = [2,10]
    base_model = 'densenet'# 'darknet'#

def train_dem(args, log, model_dir):#, tensor_log

    feature_model = torch.load(args.load)
    feature_embedding = list(feature_model.module.children())[0]
    feature_embedding = nn.DataParallel(feature_embedding).cuda()
    feature_embedding.eval()

    dem_model = DEM()
    dem_model = nn.DataParallel(dem_model).cuda()
    criterion = DEMSupportLoss().cuda()
    predictor = Predictor()
    
    optimizer = torch.optim.Adam(dem_model.parameters(), betas=(0.9, 0.99), lr= PARAM.lr)


    train_dataset = TranditionalTrainDatasetloader('train')
    train_label_map = train_dataset.get_train_label_map()
    train_embeddings, train_attributes, train_word2vec_embeddings = train_dataset.get_class_features()
    train_embeddings = torch.Tensor(train_word2vec_embeddings).cuda().float()
    trainloader = DataLoader(\
        dataset=train_dataset,\
            batch_size=PARAM.batch_size, num_workers=1, shuffle=True )

    val_dataset = TranditionalTrainDatasetloader('val')
    val_label_map = val_dataset.get_train_label_map()
    val_embeddings, val_attributes, val_word2vec_embeddings = val_dataset.get_class_features()

    val_embeddings = torch.Tensor(val_word2vec_embeddings).cuda().float()
    valloader = DataLoader(\
        dataset=val_dataset,\
            batch_size=PARAM.batch_size, num_workers=1, shuffle=True )
    
    with tqdm(total = args.epoch) as pbar:
        for p in range(args.epoch):

            train_epoch_loss = []
            train_epoch_acc = []
            val_epoch_loss = []
            val_epoch_acc = []
            val_acc = 0.0
            train_acc = 0.0

            # adjust_learning_rate(optimizer, PARAM.lr, PARAM.lr_schedule, epoch=p, gamma=0.1)
            dem_model.train()
            
            with tqdm(total = len(trainloader)) as bbar:
                for b, (image, label) in enumerate(trainloader):
                    tlabel = list()
                    for i in range(len(label)):
                        tlabel.append(train_label_map[int(label[i])])
                    tlabel = torch.LongTensor(tlabel)
                    image = image.cuda().float()
                    trans_embeddings = dem_model(train_embeddings)
                    # trans_embeddings.retain_grad()
                    image_feature = feature_embedding(image)
                    image_feature_label = image_feature.detach()
                    
                    loss = criterion(trans_embeddings, image_feature_label, tlabel)

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
                feature_embedding.eval()
                trans_embeddings = dem_model(val_embeddings)
                print("VAL:", len(valloader))
                acc = 0.0
                val_epoch_loss = []
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
                        loss = criterion(trans_embeddings, image_feature_label, tlabel)

                        val_epoch_loss.append(loss)
                        vbar.set_description('val_batch_loss: {}, val_batch_acc:{}'.format(loss, batch_acc))
                        vbar.update(1)
                val_acc = acc / len(valloader)
                val_loss = torch.mean(torch.tensor(val_epoch_loss))


            log.logger.info('*******************Train Epoch {}*************'.format(p))
            log.logger.info('train_loss:{}, train_acc:{}, val_loss:{}, val_acc:{}'.\
                        format(torch.mean(torch.tensor(train_epoch_loss)), train_acc, val_loss, val_acc))

            # torch.save(dem_model, os.path.join(model_dir, 'model_{}.pth'.format(p)))
            pbar.update(1)

def val_model(dem_model, feature_embedding, valloader, val_embeddings, val_label_map):
    with torch.no_grad():
        dem_model.eval()
        feature_embedding.eval()
        trans_embeddings = dem_model(val_embeddings)
        print("VAL:", len(valloader))
        acc = 0.0
        val_epoch_loss = []
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
                loss = criterion(trans_embeddings, image_feature_label, tlabel)

                val_epoch_loss.append(loss)
                vbar.set_description('val_batch_loss: {}, val_batch_acc:{}'.format(loss, batch_acc))
                vbar.update(1)
        val_acc = acc / len(valloader)
        val_loss = torch.mean(torch.tensor(val_epoch_loss))
    return val_acc, val_loss
            
if __name__ == '__main__':
    print('train dem.py')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    feature_model_name = args.load.split('/')
    exp_name = 'dem_lr_{}_bs_{}_basedon_{}_{}'.format(PARAM.lr, PARAM.batch_size, feature_model_name[-2], feature_model_name[-1])
    model_dir =  os.path.join(config.pro_dir, 'save', exp_name)
    log_dir = os.path.join(config.pro_dir, 'log', exp_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    model_log_file = os.path.join(log_dir, 'log_record.log')
    log = Logger(model_log_file)
    # tensor_log_file = os.path.join(log_dir, 'tensor_log_record.log')
    # tensor_log = tensorboardlog(tensor_log_file)

    train_dem(args, log, model_dir)#, tensor_log)
    