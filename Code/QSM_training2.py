import tensorflow as tf
import numpy as np
import time
import os
import re

from QSMnet_dataload2 import *
from QSMnet_utils2 import *


#
# Description:
#  Training code of QSMnet and QSMnet+
#
#  Copyright @ Woojin Jung & Jaeyeon Yoon
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : wjjung93@snu.ac.kr
#


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

valid = 1

class VarHandler:
    def __init__(self):
        self._last_vars_map = dict()

    def start(self, cname=tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, tag="", match=r".*"):
        """
        used to select variable subset: start()-end()
        :param cname: used in tf.get_collection as collection name
        :param tag: used for start, end pair
        :param match: sub-selecting variable by name match
        :return: None
        """
        re.match
        res = tf.compat.v1.get_collection(cname)
        res = [v for v in res if re.match(match, v.name)]
        if self._last_vars_map.get(tag) == None:
            self._last_vars_map[tag] = []
            self._last_vars_map[tag].append(res)
        else:
            self._last_vars_map[tag].append(res)

    def end(self, cname=tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, tag="", match=r".*"):
        """
        used to select variable subset: start()-end()
        :param cname: used in tf.get_collection as collection name
        :param tag: used for start, end pair
        :param match: sub-selecting variable by name match
        :return: None
        """

        try:
            cur_vars = tf.get_collection(cname)
            cur_vars = [v for v in cur_vars if re.match(match, v.name)]
            res_vars = []

            if self._last_vars_map.get(tag) == None or len(self._last_vars_map.get(tag)) == 0:
                _last_vars = []
                raise Exception("start(tag='{}') needed to use end()".format(tag))
            else:
                _last_vars = self._last_vars_map[tag].pop()
            for var in cur_vars:
                if var not in _last_vars:
                    res_vars.append(var)
            return res_vars
        except Exception as e:
            raise e
            print("VarHandler:end:", e)

    def add_to_collection(self, vars, cname):
        """
        add variable to collection named as cname
        :param vars: list of tensor, will be added to collection
        :param cname: collection name
        :return: None
        """
        for var in vars:
            tf.add_to_collection(cname, var)

    def get_collection(self, cname=tf.GraphKeys.GLOBAL_VARIABLES, match=r".*"):
        """
        select variable subset.
        :param cname: used in tf.get_collection as collection name
        :param match: sub-selecting variable by name match
        :return: selected variables
        """

        cur_vars = tf.get_collection(cname)
        cur_vars = [v for v in cur_vars if re.match(match, v.name)]
        res_vars = cur_vars
        return res_vars

    def filter(self, var_list, match=r".*"):
        var_list = [v for v in var_list if re.match(match, v.name)]
        res_vars = var_list
        return res_vars


# %% Train
def train():


    tf.compat.v1.reset_default_graph()
    # %% Loading dataset
    train_dataset = dataset()  # Training set, validation set

    X_mean = train_dataset.X_mean
    X_std = train_dataset.X_std
    Y_mean = train_dataset.Y_mean
    Y_std = train_dataset.Y_std
    print(X_std)
    M = np.shape(train_dataset.trfield)  # matrix size of validation set
    X = tf.compat.v1.placeholder("float", [None, M[1], M[2], M[3], 1])  # Training input
    Y = tf.compat.v1.placeholder("float", [None, M[1], M[2], M[3], 1])  # Training label
    M = tf.compat.v1.placeholder("float", [None, M[1], M[2], M[3], 1])  # Training mask
    D = tf.compat.v1.placeholder("float", [None, 64, 64, 64, 1])  # Training mask

    keep_prob = tf.compat.v1.placeholder("float")  # dropout rate

    # %% Definition of model

    vh = VarHandler()
    vh.start(tag="qsmnet", match="qsmnet")
    predX = qsmnet_deep(X, 'leaky_relu', keep_prob, False, True)

    if valid ==1:
        N = np.shape(train_dataset.tefield)  # matrix size of validation set
        X_val = tf.compat.v1.placeholder("float", [None, N[1], N[2], N[3], 1])  # Validation input
        Y_val = tf.compat.v1.placeholder("float", [None, N[1], N[2], N[3], 1])  # Validation label
        predX_val = qsmnet_deep(X_val, 'relu', keep_prob, True, False)
        loss_val = l1(predX_val, Y_val)

    # %% Definition of loss function
    l1c, mdc, gdc, tc = total_loss(predX, X, Y, M, D, 5, 0.1, X_std, X_mean, Y_std, Y_mean)


    #global_step = tf.Variable(0, trainable=False)
    #rate = tf.compat.v1.train.exponential_decay(learning_rate, global_step, 400, 0.95) + 1e-6

    #train_op = tf.compat.v1.train.RMSPropOptimizer(learning_rate=rate).minimize(mdc, global_step=global_step)
    # %% Definition of optimizer
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(tc)

    qsm_var = vh.end(tag="qsmnet", match="qsmnet")


    for v in qsm_var:
        print("qsm_vars", v.name)
    # %% Generate saver instance


    qsm_saver = tf.compat.v1.train.Saver(var_list=qsm_var)
    # %% Running session
    #Training_network_l1(train_dataset, X, Y, l1c, train_op, keep_prob, qsm_saver)

    if valid ==0:
        Training_network_novad(train_dataset, X, Y, M, D, l1c, mdc, gdc, tc, train_op, keep_prob, qsm_saver)
    elif valid==1:
        Training_network(train_dataset, X, Y, M, D, X_val, Y_val, predX_val, l1c, mdc, gdc, tc, loss_val,
                         train_op, keep_prob, qsm_saver)


if __name__ == '__main__':
    total_time = time.time()
    train()
    print("Total training time : {} sec".format(time.time() - total_time))
