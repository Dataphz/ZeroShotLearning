import os
import socket

name = socket.gethostname()
# host_address = socket.gethostbyname(socket.gethostname()).split('.')
print('host_name:', name)
stage = 1

# environ = 'all'#'all' # ['all', '']
stage_name = ['embedding_training', 'zsl_training']

#******************************************  Dataset Part ****************************************** #
address_to_homedir_map = {'122':'/disk4/penghao/workspace', '93':'/home/leon/workspace', 'gpgpu02':'/home/xxc/xcfashi','q-System-Product-Name':'/home/q/workspace'}

pro_dir = os.path.join(address_to_homedir_map[name], 'ZSL')
result_file = os.path.join(pro_dir, 'submit.txt')

train_dataset_dir = os.path.join(pro_dir,'dataset/Dataset_train')
test_dataset_dir = os.path.join(pro_dir, 'dataset/Dataset_test')
# train_set_file = os.path.join(train_dataset_dir, 'train_zsl_set.pkl')# 为了只给unseen的类进行预测。

train_image_dir = os.path.join(train_dataset_dir,'train')
train_image_label_file = os.path.join(train_dataset_dir, 'train.txt')
val_image_dir = os.path.join(train_dataset_dir,'val')
val_image_label_file = os.path.join(train_dataset_dir, 'val.txt')
test_image_dir = os.path.join(test_dataset_dir, 'test')
test_image_list_file = os.path.join(test_dataset_dir, 'image.txt')

dataset_image_dir_map = {'train':train_image_dir,'val':val_image_dir,'test':test_image_dir}
dataset_image_list_file_map = {'train':train_image_label_file,'val':val_image_label_file,'test':test_image_list_file}



test_submit_class_file = os.path.join(test_dataset_dir, 'submit_test_label.pkl')

# train_image_label_file = {'all':'train.txt', 'train':'subtrain.txt', 'val':'subval.txt'}


class_word2vec_embedding_file = os.path.join(train_dataset_dir, 'word2vec_embedding.txt')
class_embedding_file = os.path.join(train_dataset_dir, 'class_wordembeddings.txt')
attributes_file = os.path.join(train_dataset_dir, 'attributes_per_class.txt')
label_list_file = os.path.join(train_dataset_dir, 'label_list.txt')


# test_label_list_file = os.path.join(test_dataset_dir, 'label_list.txt')
# test_class_embedding_file = os.path.join(test_dataset_dir, 'class_wordembeddings.txt')
# test_attributes_file = os.path.join(test_dataset_dir, 'attributes_per_class.txt')

#****************************************** Param Part ****************************************** #
way = 10
shot = 5
query = 1
batch_size = 4
batch_size_dem = 32
if stage == 0:
    lr = 0.1
    lr_schedule = [10, 25, 60]
elif stage == 1:
    lr = 0.01
    lr_schedule = [2, 20]

weight_decay = 1e-5
batch_num = 64
width, height = 64, 64
# train_num = 38221
# split = 0.9
train_class_num = {'train': 205, 'val':50}

defined_attribute_num = 30
class_attribute_num = 300
image_feature_num = 1368

num_workers = 10
# experiment_name = '{}way_{}shot_{}prototype_{}_lr_{}_bs_{}'.format(way, shot, environ, stage_name[stage], lr, batch_size)

# model_dir = os.path.join(pro_dir, 'save', experiment_name)


# if not os.path.exists(model_dir):
#     os.mkdir(model_dir)
# if not os.path.exists(log_dir):
#     os.mkdir(log_dir)
log_dir = os.path.join(pro_dir, 'log')
debug_log_file = os.path.join(log_dir, 'debug_log_record.log')

# record log:
#              ''           
# embedding:    46 densenet 10way2shot
# zsl :      
