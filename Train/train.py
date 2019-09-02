import tensorflow as tf
import numpy as np
import random
import os
import re
import time
import scipy.io

from tqdm import tqdm
from training_model import *
from training_loss import *
from training_params import *
from training_dataload import *

#
# Description:
#  Training code of QSMnet and QSMnet+
#
#  Copyright @ Woojin Jung & Jaeyeon Yoon
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : wjjung93@snu.ac.kr
#



class VarHandler:
    def __init__(self):
        self._last_vars_map = dict()

    def start(self, cname=tf.GraphKeys.GLOBAL_VARIABLES, tag="", match=r".*"):
        """
        used to select variable subset: start()-end()
        :param cname: used in tf.get_collection as collection name
        :param tag: used for start, end pair
        :param match: sub-selecting variable by name match
        :return: None
        """
        re.match
        res = tf.get_collection(cname)
        res = [v for v in res if re.match(match, v.name)]
        if self._last_vars_map.get(tag) == None:
            self._last_vars_map[tag] = []
            self._last_vars_map[tag].append(res)
        else:
            self._last_vars_map[tag].append(res)

    def end(self, cname=tf.GraphKeys.GLOBAL_VARIABLES, tag="", match=r".*"):
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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(GPU_num)
st = time.time()


def train():
    X = tf.placeholder("float", [None, PS, PS, PS, 1])
    Y = tf.placeholder("float", [None, PS, PS, PS, 1])
    D = tf.placeholder("float", [None, PS, PS, PS, 1])
    M = tf.placeholder("float", [None, PS, PS, PS, 1])
    Z = tf.placeholder("float", [None, 176, 176, 160, 1])
    W = tf.placeholder("float", [None, 176, 176, 160, 1])
    keep_prob = tf.placeholder("float")

    train_dataset = dataset()
    X_mean = train_dataset.X_mean
    X_std = train_dataset.X_std
    Y_mean = train_dataset.Y_mean
    Y_std = train_dataset.Y_std

    ind = range(len(train_dataset.trfield))

    vh = VarHandler()
    vh.start(tag="qsmnet", match="qsmnet")


    predX = qsmnet_deep(X, keep_prob, False, True, ks)
    global_step = tf.Variable(0, trainable=False)
    rate = tf.train.exponential_decay(learning_rate, global_step, decay_step, decay_factor) + 1e-6
    l1loss, mdloss, gdloss, cost = total_loss(predX, X, Y, M, loss_w1, loss_w2, X_std, X_mean, Y_std, Y_mean)
    train_op = tf.train.RMSPropOptimizer(learning_rate=rate).minimize(cost, global_step=global_step)

    qsm_var = vh.end(tag="qsmnet", match="qsmnet")
    for v in qsm_var:
        print("qsm_vars", v.name)

    init = tf.global_variables_initializer()
    qsm_saver = tf.train.Saver(var_list=qsm_var)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    sess = tf.Session(config=config)
    sess.run(init)

    print("restore finished!")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    step = initial_step

    #Training stage
    for epoch in range(step, train_epochs+1):
        print(step)
        st = time.time()

        random.shuffle(ind)
        d_batch = train_dataset.trdd[0:batch_size, :, :, :, :]

        total_c = 0
        total_l1 = 0
        total_md = 0
        total_gd = 0
        for i in tqdm(range(0, len(ind) - batch_size, batch_size)):

            ind2 = ind[i:i + batch_size]
            ind2 = np.sort(ind2)
            x_batch = (train_dataset.trfield[ind2, :, :, :, :] - X_mean) / X_std
            y_batch = (train_dataset.trsusc[ind2, :, :, :, :] - Y_mean) / Y_std
            m_batch = train_dataset.maskt[ind2, :, :, :, :]
            c, c1, c2, c3, _ = sess.run([cost, l1loss, mdloss, gdloss, train_op],
                                        feed_dict={X: x_batch, Y: y_batch, D: d_batch, M: m_batch, keep_prob: 0.5})
            total_c = total_c + c
            total_l1 = total_l1 + c1
            total_md = total_md + c2
            total_gd = total_gd + c3

        print("epoch:", '%04d' % (step), "total_cost=", "{:.5f}".format(total_c), "l1=", "{:.5f}".format(total_l1),
              "mdl=", "{:.5f}".format(total_md), "gdl=", "{:.5f}".format(total_gd))
        print(sess.run(rate))
        if epoch % save_step == 0:
            saver_path = qsm_saver.save(sess, save_path + "/l" + str(step) + ".ckpt")
        print("[one epoch time : ] {} sec".format(time.time() - st))
        step += 1

        # Validation
        if epoch % display_step == 0:
            j = random.randrange(0, len(train_dataset.trfield) - batch_size - 1)
            test_im, cost_test = sess.run([predX, l1loss], feed_dict={Z: train_dataset.tefield, W: train_dataset.tesusc, keep_prob : 1.0})
            print("epoch:", '%04d' % (step), "l1_test=", "{:.5f}".format(cost_test))
            scipy.io.savemat(save_path + 'test_result_' + str(step) + '.mat', mdict={'out1': test_im})


if __name__ == '__main__':
    train()
