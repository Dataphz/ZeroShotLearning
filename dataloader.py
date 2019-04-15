import sys
import random
import torch.utils.data as data

from logger import Log as log
from utils import *

class TranditionalTrainDatasetloader(data.Dataset):
    """
    For training dataset: 
    
    """
    def __init__(self, train_name):
        """
        Parameters  
        train_name: ['train', 'val', 'all']
        """
        self.data_utils = DataUtils()

        self.train_name = train_name
        self.class_num = config.train_class_num[self.train_name]
        self.classes_list = range(self.class_num)

        self.images, self.labels, self.attributes, self.class_embeddings, self.word2vec_embeddings = self.get_dataset(train_name)

        print(len(self.images))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]#, self.attributes[index], self.class_embeddings[index]
    
    def get_dataset(self, train_name):
        # self.train_list_file = os.path.join(config.train_dataset_dir, config.train_image_label_file[self.train_name])
        # log.logger.info('train filename: {}'.format(self.train_list_file ))
        return self.data_utils.load_images_and_labels(config.dataset_image_list_file_map[self.train_name], config.dataset_image_dir_map[self.train_name], self.class_num)

    def get_class_features(self):
        return self.class_embeddings, self.attributes, self.word2vec_embeddings

    def get_train_label_map(self):
        return self.data_utils.train_label_map
        
class FewShotTrainDatasetloader(data.Dataset):
    """
    For training dataset: 
    
    """
    def __init__(self, train_name,  way=10, shot=5, query=1, batch_num=200, batch_size=64):
        """
        Parameters  
        way: class_num
        shot: sampel_num
        train_name: ['train', 'val', 'all']
        """
        self.data_utils = DataUtils()
        self.way = way
        self.shot = shot
        self.query = query
        self.batch_num = batch_num
        self.batch_size = batch_size
        
        self.train_name = train_name
        self.class_num = config.train_class_num[self.train_name]
        self.classes_list = range(self.class_num)

        self.images = self.get_dataset()
        log.logger.debug('Images classes number: {}'.format(len(self.images)))
        
    def __len__(self):
        return self.batch_size * self.batch_num
    
    def __getitem__(self, index):
        """
        Return: way x shot x h x w x c
        """
        support_set_x = [] # batch_size * way * shot
        support_set_y = []
        query_set_x = []
        query_set_y = []
        # for _ in range(config.batch_size):
        _set_x = []
        _set_y = []
        classes = random.sample(self.classes_list, self.way)

        for i,c in enumerate(classes):
            samples_num = len(self.images[c])
            sample_inds = random.sample(range(samples_num), self.shot + self.query)
            _set_x.append(self.images[c][sample_inds])
            _set_y.append([i for _ in range(self.shot + self.query)])
        
        _set_x = np.array(_set_x)
        _set_y = np.array(_set_y)
        
        support_set_x = np.array(_set_x[:, :self.shot])
        support_set_y = np.array(_set_y[:, :self.shot])
        query_set_x = np.array(_set_x[:, self.shot:])
        query_set_y = np.array(_set_y[:, self.shot:])

        return support_set_x, support_set_y, query_set_x, query_set_y
    
    def get_dataset(self):
        return self.data_utils.load_images_and_labels_few_shot(config.dataset_image_list_file_map[self.train_name], \
                                    config.dataset_image_dir_map[self.train_name], self.class_num)


class Imageloader(data.Dataset):
    """
    For test dataset
    Return : only images
    """
    
    def __init__(self):
        self.data_utils = DataUtils()
        self.images, self.filenames = self.get_testImages()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.filenames[index]
    
    def get_testImages(self):
        imgs, filenames = self.data_utils.load_images(config.test_image_list_file, config.test_image_dir)
        return imgs, filenames
    
    def get_test_class_features(self):
        return self.data_utils.load_test_class_features()