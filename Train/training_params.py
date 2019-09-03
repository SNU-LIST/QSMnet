
#
# Description:
#  Training parameter for QSMnet and QSMnet+
#
#  Copyright @ Woojin Jung & Jaeyeon Yoon
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : wjjung93@snu.ac.kr
#

data_folder = './'
net_name = 'QSMnet_sym'
C = {
    'data': {
        'data_folder': data_folder,
        'train_data_path': data_folder + 'training_data_patch_64_' + net_name + '.mat',
        'norm_factor_path': data_folder + 'norm_factor_' + net_name + '.mat',
        'dipole_path': data_folder + 'dipole_t_64.mat',
        'test_data_path': data_folder + 'test.mat',
        'save_path': data_folder + 'Checkpoints/' + net_name +'/'
    },
    'loss': {
        'w1': 5,
        'w2': 0.1
    },
    'train': {
        'GPU_num': 0,
        'patch_size': 64,
        'batch_size': 12,
        'step': 1,
        'learning_rate': 0.001,
        'decay_step': 600,
        'decay_factor': 0.95,
        'kernel_size': 5,
        'train_epochs': 25,
        'display_step': 1,
        'epsilon': 1e-3,
        'save_step': 5
    }
}

train_data_path = C['data']['train_data_path']
norm_factor_path = C['data']['norm_factor_path']
dipole_path = C['data']['dipole_path']
test_data_path = C['data']['test_data_path']
save_path = C['data']['save_path']

loss_w1 = C['loss']['w1']
loss_w2 = C['loss']['w2']

GPU_num = C['train']['GPU_num']
PS = C['train']['patch_size']
batch_size = C['train']['batch_size']
initial_step = C['train']['step']
learning_rate = C['train']['learning_rate']
decay_step = C['train']['decay_step']
decay_factor = C['train']['decay_factor']
ks = C['train']['kernel_size']
train_epochs = C['train']['train_epochs']
display_step = C['train']['display_step']
epsilon = C['train']['epsilon']
save_step = C['train']['save_step']
