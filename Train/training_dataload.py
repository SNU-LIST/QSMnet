import h5py
from training_params import *

#
# Description:
#  Training data of QSMnet and QSMnet+
#
#  Copyright @ Woojin Jung & Jaeyeon Yoon
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : wjjung93@snu.ac.kr
#

class dataset():
    def __init__(self):
        f = h5py.File(train_data_path)
        f2 = h5py.File(test_data_path)
        f3 = h5py.File(dipole_path)
        f4 = h5py.File(norm_factor_path)

        self.trfield = f['temp_i']
        self.trsusc = f['temp_l']
        self.maskt = f["mask_t"]
        self.tefield = f2['test_i']
        self.tesusc = f2['test_l']
        self.trdd = f3["Dpatch_train"]
        self.X_mean = f4["input_mean"]
        self.X_std = f4["input_std"]
        self.Y_mean = f4["output_mean"]
        self.Y_std = f4["output_std"]