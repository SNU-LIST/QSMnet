import os
import h5py
import numpy as np
import scipy.io
import shutil
#
# Description:
#  Training parameter for QSMnet and QSMnet+
#
#  Copyright @ Woojin Jung & Jaeyeon Yoon
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : wjjung93@snu.ac.kr
#

data_folder = "/data/pycharm/QSMnet_spatial/"
net_name = 'QSMnet_resize2_imfilt3'
PS = '64' # patch_size
training_data_name = 'training_64_resize2_imfilter3.mat'
mask_data_name = 'mask_total.mat'


C = {
    'data': {
        'data_folder': data_folder,
        'train_data_path': data_folder + training_data_name,
        #'mask_data_patch': '/data/trset/' + mask_data_name,
        'val_input_path': data_folder + 'Train/Input/val_input.mat',
        'val_label_path': data_folder + 'Train/Label/val_label.mat',
        'dipole_path': '/data/trset/dipole_64.mat',#whole_brain.mat',
        'save_path': data_folder + net_name + '/'
    },

    'train': {
        'batch_size': 12, # batch size
        'learning_rate': 0.001, # initial learning rate
        'train_epochs': 25, # The number of training epochs
        'save_step': 5 # Step for saving network
    },
    'validation': {
        'display_step': 2, # display step of validation images
        'display_slice_num': [72,92,112,132], # slice number of validation images for displaying
    }
}

train_data_path = C['data']['train_data_path']
#mask_data_path = C['data']['mask_data_path']
val_input_path = C['data']['val_input_path']
val_label_path = C['data']['val_label_path']
dipole_path = C['data']['dipole_path']
save_path = C['data']['save_path']
if not os.path.exists(save_path+ 'validation_result'):
    os.makedirs(save_path + 'validation_result')
#shutil.copy2('/data/pycharm/Loss_study/QSMnet_ft_nopatch/norm_factor.mat', save_path + 'norm_factor.mat')
batch_size = C['train']['batch_size']
learning_rate = C['train']['learning_rate']
train_epochs = C['train']['train_epochs']
save_step = C['train']['save_step']

display_step = C['validation']['display_step']
display_slice_num = C['validation']['display_slice_num']


class dataset():
    def __init__(self):
        f = h5py.File(train_data_path, "r")
        #fm = scipy.io.loadmat(mask_data_path)
        f2i = scipy.io.loadmat(val_input_path)
        f2l = scipy.io.loadmat(val_label_path)
        f3 = scipy.io.loadmat(save_path + 'norm_factor.mat')
        f4 = scipy.io.loadmat(dipole_path)
        self.trfield = f['temp_i']
        self.trsusc = f['temp_l']
        self.trmask = f['temp_m']

        #self.trfield = self.trfield[:, 8:168, 8:168, 8:152, :]
        #self.trsusc = self.trsusc[:, 8:168, 8:168, 8:152, :]
        #self.trmask = self.trmask[:, 8:168, 8:168, 8:152, :]



        self.tefield = f2i['phs_tissue']
        self.tesusc = f2l['chi_cosmos']
        self.tefield = np.expand_dims(self.tefield, axis=0)
        self.tefield = np.expand_dims(self.tefield, axis=4)
        self.tesusc = np.expand_dims(self.tesusc, axis=0)
        self.tesusc = np.expand_dims(self.tesusc, axis=4)
        self.dipole = f4['D']
        self.dipole = np.expand_dims(self.dipole, axis=0)
        self.dipole = np.expand_dims(self.dipole, axis=4)
        self.X_mean = 0#f3["input_mean"]
        self.X_std = f3["input_std"]
        self.Y_mean = 0#f3["label_mean"]
        self.Y_std = f3["label_std"]

