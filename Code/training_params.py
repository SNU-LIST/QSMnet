import os
import numpy as np

#
# Description:
#  Training parameter for QSMnet and QSMnet+
#
#  Copyright @ Woojin Jung & Jaeyeon Yoon
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : wjjung93@snu.ac.kr
#

data_folder = "../Data/"
net_name = 'QSMnet+'
PS = '64' # patch_size
act_func = 'leaky_relu'
net_model = 'qsmnet_deep'

C = {
    'data': {
        'data_folder': data_folder,
        'train_data_path': data_folder + 'Train/Training_data_patch/training_data_patch_' + net_name + '_' + PS + '.hdf5',
        'val_input_path': data_folder + 'Train/Input/val_input.mat',
        'val_label_path': data_folder + 'Train/Label/val_label.mat',
        'save_path': '../Checkpoints/' + net_name + '_'+ PS + '/'
    },

    'train': {
        'batch_size': 5, # batch size
        'learning_rate': 0.001, # initial learning rate
        'train_epochs': 25, # The number of training epochs
        'save_step': 5 # Step for saving network
    },
    'validation': {
        'display_step': 2, # display step of validation images
        'display_slice_num': [52,72,92,112], # slice number of validation images for displaying
    }
}

train_data_path = C['data']['train_data_path']
val_input_path = C['data']['val_input_path']
val_label_path = C['data']['val_label_path']
save_path = C['data']['save_path']
if not os.path.exists(save_path+ 'validation_result'):
    os.makedirs(save_path + 'validation_result')
network_info = [act_func, net_model]                
np.save(save_path + 'network_info_' + net_name + '_' + PS, network_info)
batch_size = C['train']['batch_size']
learning_rate = C['train']['learning_rate']
train_epochs = C['train']['train_epochs']
save_step = C['train']['save_step']

display_step = C['validation']['display_step']
display_slice_num = C['validation']['display_slice_num']
