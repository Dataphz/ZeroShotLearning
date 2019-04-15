import os
import argparse
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader 
import torch.nn as nn
import numpy as np

from model.fewshot_model import PrototypeNetwork, LossFunc
from dataloader import FewShotTrainDatasetloader
from logger import Logger
import config
from utils import adjust_learning_rate
from tensorboard import Logger as tensorboardlog

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
parser.add_argument('--epoch', help='epoch num', default=100)
# parser.add_argument('--load', help='load model')
# parser.add_argument('--datadir', help='override config.BASEDIR')
# parser.add_argument('--evaluate', help='path to the output pkl eval file')
# parser.add_argument('--predict', help='path to the input image file')

class PARAM():
    way = 10
    shot = 2
    query = 1
    batch_size = 4
    batch_num = 500
    lr = 0.01
    lr_schedule = [50,100]
    base_model = 'darknet'#'densenet'# 
    # weight_decay = 1e-6
    # image_feature_num = 1368 

def train(args, log, model_dir, tensor_log):

    model = PrototypeNetwork(PARAM.way, PARAM.shot, PARAM.query, basemodel = PARAM.base_model)
    model = nn.DataParallel(model).cuda()
    criterion = LossFunc().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr= PARAM.lr )

    trainloader = DataLoader(dataset=FewShotTrainDatasetloader('train', PARAM.way, PARAM.shot, PARAM.query, PARAM.batch_num, PARAM.batch_size),\
            batch_size=PARAM.batch_size, num_workers=config.num_workers, shuffle=False )
    valloader = DataLoader(dataset=FewShotTrainDatasetloader('val', PARAM.way, PARAM.shot, PARAM.query, PARAM.batch_num * 5, PARAM.batch_size),\
            batch_size=PARAM.batch_size, num_workers=config.num_workers, shuffle=False )

    with tqdm(total = args.epoch) as pbar:
        for p in range(args.epoch):
            train_epoch_loss = []
            train_epoch_acc = []
            val_epoch_loss = []
            val_epoch_acc = []

            adjust_learning_rate(optimizer, PARAM.lr, PARAM.lr_schedule, epoch=p, gamma=0.1)
            model.train()

            with tqdm(total = len(trainloader)) as bbar:
                print('*******************TRAIN*******************')
                for b, (support_set_x, support_set_y, query_set_x, query_set_y) in enumerate(trainloader):
                    
                    support_set_x = support_set_x.cuda().float()
                    support_set_y = support_set_y.cuda().float()
                    query_set_x = query_set_x.cuda().float()
                    query_set_y = query_set_y.cuda().float()
                    similarities = model(support_set_x, support_set_y, query_set_x)
                    acc, loss = criterion(similarities, query_set_y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_epoch_loss.append(loss)
                    train_epoch_acc.append(acc)
                    bbar.set_description('train_batch_loss: {}, train_batch_acc:{}'.format(loss, acc))
                    bbar.update(1)
            print('*******************VAL*******************')
            if p % 2 == 0:
                with torch.no_grad():
                    model.eval()
                    with tqdm(total = len(valloader)) as vbar:
                        for b, (support_set_x, support_set_y, query_set_x, query_set_y) in enumerate(valloader):
                            support_set_x = support_set_x.cuda().float()
                            support_set_y = support_set_y.cuda().int()
                            query_set_x = query_set_x.cuda().float()
                            query_set_y = query_set_y.cuda().int()

                            similarities = model(support_set_x, support_set_y, query_set_x)
                            acc, loss = criterion(similarities, query_set_y)
                            val_epoch_loss.append(loss.data)
                            val_epoch_acc.append(acc.data)
                            vbar.set_description('val_batch_loss: {}, val_batch_acc:{}'.format(loss, acc))
                            vbar.update(1)
                    val_epoch_avr_loss = np.mean(val_epoch_loss)
                    val_epoch_avr_acc = np.mean(val_epoch_acc)

            log.logger.info('*******************Train Epoch {}*************'.format(p))
            train_epoch_avr_loss = torch.mean(torch.tensor(train_epoch_loss))
            train_epoch_avr_acc = torch.mean(torch.tensor(train_epoch_acc))
            log.logger.info('train_batch_loss: {}, train_batch_acc:{}'.format(train_epoch_avr_loss, train_epoch_avr_acc))
            
            tensor_log.scalar_summary('train_epoch_avr_loss', train_epoch_avr_loss.item(), p)
            tensor_log.scalar_summary('train_epoch_avr_acc', train_epoch_avr_acc.item(), p)
            if p % 2 == 0 :
                log.logger.info('val_batch_loss: {}, val_batch_acc:{}'.format(val_epoch_avr_loss, val_epoch_avr_acc))
                tensor_log.scalar_summary('val_epoch_avr_loss', val_epoch_avr_loss, p)
                tensor_log.scalar_summary('val_epoch_avr_acc', val_epoch_avr_acc, p)
            
            
            if p % 2 == 0:
                torch.save(model, os.path.join(model_dir, 'model_{}.pth'.format(p)))
            pbar.update(1)

if __name__ == '__main__':

    print('train fewshot.py')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    exp_name = '{}_{}way_{}shot_prototype_lr_{}_bs_{}'.format(PARAM.base_model, PARAM.way, PARAM.shot, PARAM.lr, PARAM.batch_size)
    model_dir =  os.path.join(config.pro_dir, 'save', exp_name)
    log_dir = os.path.join(config.pro_dir, 'log', exp_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    model_log_file = os.path.join(log_dir, 'log_record.log')
    log = Logger(model_log_file)
    tensor_log_file = os.path.join(log_dir, 'tensor_log_record.log')
    tensor_log = tensorboardlog(tensor_log_file)
    train(args, log, model_dir, tensor_log)