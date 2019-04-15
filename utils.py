import numpy as np
import os
import cv2
import tqdm
import config

import pickle
import json
import torch
from logger import Log as log

#****************************************** Data Part ****************************************** #

class DataUtils:
    def __init__(self):

        self.label_to_attributes_map = self.load_attribute()
        self.ZJL_to_labelname_map, self.labelname_to_ZJL_map, self.ZJL_to_labellogits_map = self.load_label_name()
        self.label_to_class_embedding_map = self.load_class_embedding()
        self.label_to_class_word2vec_embedding_map = self.load_word2vec_embedding()

    def load_attribute(self):
        label_to_attributes_map = dict()
        with open(config.attributes_file, 'r') as f:
            for i, line in enumerate(f):
                l = line.split('\t')
                attributes = list()
                for a in l[1:]:
                    attributes.append(float(a.strip()))
                label_to_attributes_map[l[0].strip()] = attributes
        return label_to_attributes_map
    #check
    # print(len(label_to_attributes_map), label_to_attributes_map.keys())
    # label_to_attributes_map = load_attribute()

    def load_label_name(self):
        ZJL_to_labelname_map = dict()
        labelname_to_ZJL_map = dict()
        ZJL_to_labellogits_map = dict()
        labellogits_to_ZJL_map = dict()
        with open(config.label_list_file, 'r') as f:
            for i, line in enumerate(f):
                l = line.split('\t')
                ZJL_to_labelname_map[l[0].strip()] = l[1].strip()
                labelname_to_ZJL_map[l[1].strip()] = l[0].strip()
            for i, zjl in enumerate(ZJL_to_labelname_map.keys()):
                ZJL_to_labellogits_map[zjl] = i
                labellogits_to_ZJL_map[i] = zjl
        return ZJL_to_labelname_map, labelname_to_ZJL_map, ZJL_to_labellogits_map
    #check
    # ZJL_to_labelname_map, labelname_to_ZJL_map, ZJL_to_labellogits_map = load_label_name()
    # print(len(ZJL_to_labelname_map))
    # print(ZJL_to_labelname_map.keys())
    # print(labelname_to_ZJL_map.keys())
    # print(ZJL_to_labellogits_map)

    def load_class_embedding(self):
        class_embedding_map = dict()
        with open(config.class_embedding_file, 'r') as f:
            for i,line in enumerate(f):
                l = line.split(' ')
                embedding_features = list()
                for a in l[1:]:
                    embedding_features.append(float(a.strip()))
                class_embedding_map[self.labelname_to_ZJL_map[l[0].strip()]] = embedding_features
            return class_embedding_map

    def load_word2vec_embedding(self):
        class_embedding_map = dict()
        with open(config.class_word2vec_embedding_file, 'r') as f:
            for i,line in enumerate(f):
                l = line.split(' ')
                embedding_features = list()
                for a in l[1:]:
                    embedding_features.append(float(a.strip()))
                class_embedding_map[self.labelname_to_ZJL_map[l[0].strip()]] = embedding_features
            return class_embedding_map

    #check
    # label_to_class_embedding_map = load_class_embedding()
    # print(len(label_to_class_embedding_map), label_to_class_embedding_map.keys())
    # print(label_to_class_embedding_map['ZJL177'])
    def load_embedding_and_attribute(self):
        zjl_set = self.ZJL_to_labelname_map.keys()
        attributes = np.zeros((len(zjl_set), config.defined_attribute_num), dtype=np.float32)
        embeddings = np.zeros((len(zjl_set), config.class_attribute_num), dtype=np.float32)
        for zjl in zjl_set:
            label = self.ZJL_to_labellogits_map[zjl]
            embeddings[label] = self.label_to_class_embedding_map[zjl]
            attributes[label] = self.label_to_attributes_map[zjl]

        return embeddings, attributes

    def load_test_class_features(self):

        submit_zsl_set = set()
        with open(config.test_submit_class_file,'r') as f:
            for i, line in enumerate(f):
                l = line.split('\t')
                ZJL = l[0].strip()
                submit_zsl_set.add(ZJL)
        
        test_word2vec_embeddings = []
        test_attributes = []
        label_to_ZJL = {}
        for i, zjl in enumerate(submit_zsl_set):
            test_attributes.append(self.label_to_attributes_map[zjl])
            test_word2vec_embeddings.append(self.label_to_class_word2vec_embedding_map[zjl])
            label_to_ZJL[i] = zjl

        return test_attributes, test_word2vec_embeddings, label_to_ZJL

    def load_images_and_labels(self, image_label_file, image_dir, class_num):
        """
        load traning dataset.
        """
        imgs = list()
        labels = list()
        train_zsl_set = set()

        with open(image_label_file,'r') as f:
            for i, line in enumerate(f):
                l = line.split('\t')
                ZJL = l[1].strip()
                train_zsl_set.add(ZJL)
                image_filename = l[0].strip()
                string_label = l[1].strip()
                img = cv2.imread(os.path.join(image_dir,image_filename)) /255.0
                img = np.transpose(img, axes=[2,0,1])
                imgs.append(img)
                labels.append(self.ZJL_to_labellogits_map[ZJL])   
        # if class_num == config.train_class_num['all']:
        #     pickle.dump(train_zsl_set, open(config.train_set_file, 'wb'))
        assert len(train_zsl_set) == class_num, 'train class num:{}'.format(len(train_zsl_set))
        self.train_zsl_set = train_zsl_set
        log.logger.debug('train_zsl_set_num:{}'.format(len(self.train_zsl_set)))
        

        imgs = np.array(imgs)
        labels = np.array(labels)

        attributes = list()
        class_embeddings = list()
        word2vec_embeddings = list()
        self.train_label_map = dict()
        if class_num == 50:
            test_seen_classes = ['ZJL105', 'ZJL160', 'ZJL15', 'ZJL172', 'ZJL176', 'ZJL129', 'ZJL179', 'ZJL196', 'ZJL184', 'ZJL163', 'ZJL169']
        else:
            test_seen_classes = []

        for i,zjl in enumerate(self.train_zsl_set):
            # if zjl not in test_seen_classes:
            self.train_label_map[self.ZJL_to_labellogits_map[zjl]] = i
            attributes.append(self.label_to_attributes_map[zjl])
            class_embeddings.append(self.label_to_class_embedding_map[zjl])
            word2vec_embeddings.append(self.label_to_class_word2vec_embedding_map[zjl])
        attributes = np.array(attributes)
        class_embeddings = np.array(class_embeddings)
        word2vec_embeddings = np.array(word2vec_embeddings)

        return imgs, labels, attributes, class_embeddings, word2vec_embeddings
    #check
    # imgs, labels, attributes, class_embeddings = load_images_and_labels(config.train_image_label_file, config.train_image_dir, 'test')
    # print(len(imgs))

    def load_images_and_labels_few_shot(self, train_file, train_image_dir, class_num):

        return_imgs = []
        # return_labels = []
        imgs, labels, _, _, _= self.load_images_and_labels(train_file, train_image_dir, class_num)# nx64x64x3, nx1
        log.logger.debug('train_zsl_set:{}, shape:{}'.format(len(self.train_zsl_set), np.shape(imgs)))
        for zsl in self.train_zsl_set:
            label = self.ZJL_to_labellogits_map[zsl]
            label_inds = np.where(labels == label)
            class_imgs = imgs[label_inds]
            return_imgs.append(np.array(class_imgs))
            # return_labels.append(class_labels)
        return return_imgs

    def save_dataset_split_index(self):
        """
        split traindatset into sub-train dataset and sub-val dataset and save into trian_ids/ val_ids file.
        """
        a = np.arange(config.train_num)
        np.random.shuffle(a)
        train_num = int(config.train_num * config.split)
        train_ids = a[:train_num]
        val_ids = a[train_num:]
        np.save(config.train_ids, train_ids)
        np.save(config.val_ids, val_ids)
    #check
    # save_dataset_split_index()
    # train_ids = np.load(config.train_ids)

    def load_images(self, image_list_file, image_dir):
        imgs = list()
        filenames = list()
        with open(image_list_file, 'r') as f:
            for i,line in enumerate(f):
                image_filename = line.strip()
                filenames.append(image_filename)
                img = cv2.imread(os.path.join(image_dir, image_filename))/255.0
                img = np.transpose(img, axes=[2,0,1])
                imgs.append(img)
        return imgs, filenames
    #check
    # imgs = load_images(config.test_image_list_file, config.test_image_dir)
    # print(np.shape(imgs))


#****************************************** Model Part ****************************************** #



def save_options(opt, path,model,criterion, optimizer):
    file_path = os.path.join(path, 'opt.json')
    model_struc = model.__str__()
    model_struc = {'Model': model_struc, 'Loss Function': criterion, 'Optimizer': optimizer}

    with open(file_path, 'w') as f:
        f.write(json.dumps(vars(opt), sort_keys=True, indent=4))
        f.write(json.dumps(model_struc, sort_keys=True, indent=4))

def save_model(state, checkpoint, filename='checkpoint.pth.tar'):
    filename = 'epoch'+str(state['epoch']) + filename
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, lr, lr_schedule, epoch, gamma = 0.1):
    """Sets the learning rate to the initial LR decayed by schedule"""
    for i, p in enumerate(lr_schedule):
        if epoch < p:
            lr = lr * (gamma**i)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            break
    if epoch > lr_schedule[-1]:
        lr = lr * (gamma**len(lr_schedule))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
#check
#a = torch.Tensor([[1,2,3],[2,3,4]])
#b = torch.Tensor([[1,2,3],[3,4,5],[2,3,4]])
def l2_similarity(attribute_features, image_features):
    """
    Input:
        attribute_features: A x L
        image_features: I x L
    Return:
        l2_distance: IxA
    """
    attribute_features = attribute_features / torch.sqrt(torch.sum(torch.pow(attribute_features, 2), dim=-1, keepdim=True))
    image_features = image_features / torch.sqrt(torch.sum(torch.pow(image_features, 2), dim=1, keepdim=True))
    l2_distance =  image_features.view(image_features.size(0), 1, image_features.size(1)) - attribute_features
    l2_distance = -1 * torch.sum(torch.pow(l2_distance, 2), dim=2)
    l2_distance = l2_distance - torch.max(l2_distance, dim=1, keepdim=True)[0]
    return l2_distance

def cosine_similarity(attribute_features, image_features, eps=1e-6):
    """
    Input:
        attribute_features: A x L
        image_features: I x L
    Return:
        cos_distance: IxA
    """
    attribute_amplitude = torch.sqrt(torch.sum(torch.pow(attribute_features, 2), dim=-1, keepdim=True))#Ax1
    image_amplitude = torch.sqrt(torch.sum(torch.pow(image_features, 2), dim=-1, keepdim=True))# Ix1
    amplitude = image_amplitude * torch.t(attribute_amplitude)#IxA
    cos_distance = torch.matmul(image_features, torch.t(attribute_features) ) / torch.max(amplitude, torch.ones_like(amplitude)*eps)#IxA
    return cos_distance
