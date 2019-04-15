import os
from tqdm import tqdm 
import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import numpy as np 

from dem import DEM, DEMLoss, Predictor, DEMSupportLoss
from dataloader import TranditionalTrainDatasetloader, Imageloader
from logger import Log as log
import config
from utils import adjust_learning_rate, DataUtils
import pickle


def predict():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--load_img', help='load image model')
    parser.add_argument('--load_zsl', help='load zsl model')
    # parser.add_argument('--feature_file', help='the file save generated features')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    feature_model = torch.load(args.load_img)
    feature_embedding = list(feature_model.module.children())[0]
    feature_embedding = nn.DataParallel(feature_embedding).cuda()
    feature_embedding.eval()

    zsl_model = torch.load(args.load_zsl)
    zsl_embedding = zsl_model#nn.DataParallel(zsl_model).cuda()
    zsl_embedding.eval()

    test_dataset = Imageloader()
    test_attributes, test_word2vec_embeddings, label_to_ZJL = test_dataset.get_test_class_features()
    print(np.shape(test_attributes), np.shape(test_word2vec_embeddings), len(label_to_ZJL))

    testloader = DataLoader(\
        dataset=test_dataset,\
            batch_size=1, num_workers=config.num_workers, shuffle=True )

    # with torch.cuda.device(0):
    test_attributes = torch.Tensor(test_attributes)#.cuda().float()
    test_word2vec_embeddings = torch.Tensor(test_word2vec_embeddings)#.cuda().float()
        
    class2image_features = zsl_embedding(test_attributes, test_word2vec_embeddings)
    
    # 40 x fl

    pred_zjls = []
    pred_filenames = []
    with tqdm(total = len(testloader)) as bar:
        for i, (image, filename) in enumerate(testloader):
            # with torch.cuda.device(0):
            image = image.cuda().float()

            image_feature = feature_embedding(image)
            class2image_features = class2image_features / torch.sqrt(torch.sum(torch.pow(class2image_features, 2), dim=-1, keepdim=True))
            image_feature = image_feature / torch.sqrt(torch.sum(torch.pow(image_feature, 2), dim=-1, keepdim=True))
            distance = torch.sum(torch.pow((class2image_features - image_feature), 2), dim=1)
            # print(class2image_features)
            # print(image_feature)
            # print(distance)
            # print(distance.size())
            # print(distance.size())
            pred = torch.argmin(distance)
            # print(pred)
            pred_zjls.append(label_to_ZJL[int(pred)])
            pred_filenames.append(filename)
            bar.update(1)
    # with open('result_zjl.pkl','wb') as file:
    #     pickle.dump(pred_zjls, file)
    # with open('result_filename', 'wb') as file:
    #     pickle.dump(pred_filenames, file)
    with open(config.result_file, 'w') as file:
        for i,filename in enumerate(pred_filenames):
            result = "{}\t{}\n".format(filename[0], pred_zjls[i])
            file.write(result)

if __name__ == '__main__':
    predict()