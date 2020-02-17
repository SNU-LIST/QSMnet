import scipy.io
import numpy as np
import h5py
import time
import os
import sys
#
# Description:
# Patching training data
#
# Inputs:
#   training_bf_patch_dir : directory of training data before patch
#                         : inputs - local field data (phs_tissue)
#                         : outputs - cosmos susceptibility data (chi_cosmos)
#   mask_dir : directory of mask data (mask)
#   PS : patch size
#   sub_num : subject number
#   aug_num : augmentation number
#   patch_num : the number of patches [x,y,z] per subject
#
# Outputs:
#   save training_data_patch_64.mat
#
# Copyright @ Woojin Jung
# Laboratory for Imaging Science and Technology
# Seoul National University
# email: wjjung93@snu.ac.kr
#

'''
File Path
'''
FILE_PATH_INPUT = '../Data/Train/Input/train_input'
FILE_PATH_MASK = '../Data/Train/Input/mask'
FILE_PATH_LABEL = '../Data/Train/Label/train_label'
start_time = time.time()

'''
Constant Variables
'''
PS = 32  # Patch size
net_name = 'QSMnet'
sub_num = 1  # number of subjects
dir_num = 1  # number of directions
patch_num = [5, 5, 5]  # Order of Dimensions: [x, y, z]

'''
Code Start
'''

# Create Result File
result_file = h5py.File(
    '../Data/Train/Training_data_patch/training_data_patch_'+ str(net_name) + '_' + str(PS) + '.hdf5', 'w')

# Patch the input & mask file ----------------------------------------------------------------


print("####patching input####")
patches = []
patches_mask = []
for dataset_num in range(1, sub_num + 1):
    field = scipy.io.loadmat(FILE_PATH_INPUT + str (dataset_num) + '.mat')
    mask = scipy.io.loadmat(FILE_PATH_MASK + str(dataset_num) + '.mat')
    matrix_size = np.shape(field['phs_tissue'])
    strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]
    
    if np.size(matrix_size) == 3:
        field['phs_tissue'] = np.expand_dims(field['phs_tissue'], axis = 3)
        matrix_size = np.append(matrix_size, [1], axis=0)
        mask['mask'] = np.expand_dims(mask['mask'], axis = 3)
    
    if matrix_size[3] < dir_num:
        sys.exit("dir_num is bigger than data size!")
        
    for idx in range(dir_num):
        for i in range(patch_num[0]):
            for j in range(patch_num[1]):
                for k in range(patch_num[2]):
                    patches.append(field['phs_tissue'][
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS,
                                   idx])
                    patches_mask.append(mask['mask'][
                                        i * strides[0]:i * strides[0] + PS,
                                        j * strides[1]:j * strides[1] + PS,
                                        k * strides[2]:k * strides[2] + PS,
                                        idx])
print("Done!")

patches = np.array(patches, dtype='float32', copy=False)
patches_mask = np.array(patches_mask, dtype='float32', copy=False)

patches = np.expand_dims(patches,axis=4)
patches_mask = np.expand_dims(patches_mask,axis=4)
print("Final input data size : " + str(np.shape(patches)))

input_mean = np.mean(patches[patches_mask > 0])
input_std = np.std(patches[patches_mask > 0])

#patches = (patches - input_mean) / input_std

result_file.create_dataset('temp_i', data=patches)
result_file.create_dataset('temp_m', data=patches_mask)

del patches

# Patch the label file --------------------------------------------------------------------

patches = []
print("####patching label####")
for dataset_num in range(1, sub_num + 1):
    susc = scipy.io.loadmat(FILE_PATH_LABEL + str (dataset_num) + '.mat')
    matrix_size = np.shape(susc['chi_cosmos'])
    strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]
    
    if np.size(matrix_size) == 3:
        susc['chi_cosmos'] = np.expand_dims(susc['chi_cosmos'], axis = 3)
        matrix_size = np.append(matrix_size, [1], axis=0)
        
    if matrix_size[3] < dir_num:
        sys.exit("dir_num is bigger than data size!")
        
    for idx in range(dir_num):
        for i in range(patch_num[0]):
            for j in range(patch_num[1]):
                for k in range(patch_num[2]):
                    patches.append(susc['chi_cosmos'][
                                   i * strides[0]:i * strides[0] + PS,
                                   j * strides[1]:j * strides[1] + PS,
                                   k * strides[2]:k * strides[2] + PS,
                                   idx])
print("Done!")

patches = np.array(patches, dtype='float32', copy=False)
patches = np.expand_dims(patches,axis=4)    
print("Final label data size : " + str(np.shape(patches)))

label_mean = np.mean(patches[patches_mask > 0])
label_std = np.std(patches[patches_mask > 0])

#patches = (patches - label_mean) / label_std

result_file.create_dataset('temp_l', data=patches)

del patches
del patches_mask
result_file.close()

save_path = '../Checkpoints/' + str(net_name) + '_'+ str(PS) + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
scipy.io.savemat(save_path + 'norm_factor.mat',
                 mdict={'input_mean': input_mean, 'input_std': input_std,
                        'label_mean': label_mean, 'label_std': label_std})
print("Total Time: {:.3f}".format(time.time() - start_time))
