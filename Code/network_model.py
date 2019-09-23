import tensorflow as tf
from utils import *
#
# Description:
#  3D U-net architecture of QSMnet and QSMnet+
#
#  Copyright @ Woojin Jung & Jaeyeon Yoon
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : wjjung93@snu.ac.kr
#


def qsmnet_toy(x, act_func, keep_prob, reuse, isTrain):
    with tf.compat.v1.variable_scope("qsmnet", reuse=reuse) as scope:
        with tf.compat.v1.variable_scope("conv11", reuse=reuse) as scope:
            conv11 = conv3d(x, [5, 5, 5, 1, 2], [2], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv12", reuse=reuse) as scope:
            conv12 = conv3d(conv11, [5, 5, 5, 2, 2], [2], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("maxpool1", reuse=reuse) as scope:
            pool1 = max_pool(conv12, 2)
            scope.reuse_variables()


        with tf.compat.v1.variable_scope("l_conv1", reuse=reuse) as scope:
            l_conv1 = conv3d(pool1, [5, 5, 5, 2, 4], [4], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("l_conv2", reuse=reuse) as scope:
            l_conv2 = conv3d(l_conv1, [5, 5, 5, 4, 4], [4], keep_prob, act_func, isTrain)
            scope.reuse_variables()


        with tf.compat.v1.variable_scope("deconv1", reuse=reuse) as scope:
            deconv1 = deconv3d(l_conv2, [2, 2, 2, 2, 4], [2], 2, isTrain)
            deconv_concat1 = tf.concat([conv12, deconv1], axis=4)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv81", reuse=reuse) as scope:
            conv81 = conv3d(deconv_concat1, [5, 5, 5, 4, 2], [2], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv82", reuse=reuse) as scope:
            conv82 = conv3d(conv81, [5, 5, 5, 2, 2], [2], keep_prob, act_func, isTrain)
            scope.reuse_variables()
            
        with tf.compat.v1.variable_scope("out", reuse=reuse) as scope:
            out_image = conv(conv82, [1, 1, 1, 2, 1], [1], isTrain)
            scope.reuse_variables()

    return out_image



def qsmnet_deep(x, act_func, keep_prob, reuse, isTrain):
    with tf.compat.v1.variable_scope("qsmnet", reuse=reuse) as scope:
        with tf.compat.v1.variable_scope("conv11", reuse=reuse) as scope:
            conv11 = conv3d(x, [5, 5, 5, 1, 32], [32], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv12", reuse=reuse) as scope:
            conv12 = conv3d(conv11, [5, 5, 5, 32, 32], [32], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("maxpool1", reuse=reuse) as scope:
            pool1 = max_pool(conv12, 2)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("conv21", reuse=reuse) as scope:
            conv21 = conv3d(pool1, [5, 5, 5, 32, 64], [64], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv22", reuse=reuse) as scope:
            conv22 = conv3d(conv21, [5, 5, 5, 64, 64], [64], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("maxpool2", reuse=reuse) as scope:
            pool2 = max_pool(conv22, 2)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("conv31", reuse=reuse) as scope:
            conv31 = conv3d(pool2, [5, 5, 5, 64, 128], [128], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv32", reuse=reuse) as scope:
            conv32 = conv3d(conv31, [5, 5, 5, 128, 128], [128], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("maxpool3", reuse=reuse) as scope:
            pool3 = max_pool(conv32, 2)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("conv41", reuse=reuse) as scope:
            conv41 = conv3d(pool3, [5, 5, 5, 128, 256], [256], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv42", reuse=reuse) as scope:
            conv42 = conv3d(conv41, [5, 5, 5, 256, 256], [256], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("maxpool4", reuse=reuse) as scope:
            pool4 = max_pool(conv42, 2)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("l_conv1", reuse=reuse) as scope:
            l_conv1 = conv3d(pool4, [5, 5, 5, 256, 512], [512], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("l_conv2", reuse=reuse) as scope:
            l_conv2 = conv3d(l_conv1, [5, 5, 5, 512, 512], [512], keep_prob, act_func, isTrain)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("deconv4", reuse=reuse) as scope:
            deconv4 = deconv3d(l_conv2, [2, 2, 2, 256, 512], [256], 2, isTrain)
            deconv_concat4 = tf.concat([conv42, deconv4], axis=4)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv51", reuse=reuse) as scope:
            conv51 = conv3d(deconv_concat4, [5, 5, 5, 512, 256], [256], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv52", reuse-reuse) as scope:
            conv52 = conv3d(conv51, [5, 5, 5, 256, 256], [256], keep_prob, act_func, isTrain)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("deconv3", reuse=reuse) as scope:
            deconv3 = deconv3d(conv52, [2, 2, 2, 128, 256], [128], 2, isTrain)
            deconv_concat3 = tf.concat([conv32, deconv3], axis=4)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv61", reuse=reuse) as scope:
            conv61 = conv3d(deconv_concat3, [5, 5, 5, 256, 128], [128], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv62", reuse=reuse) as scope:
            conv62 = conv3d(conv61, [5, 5, 5, 128, 128], [128], keep_prob, act_func, isTrain)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("deconv2", reuse=reuse) as scope:
            deconv2 = deconv3d(conv62, [2, 2, 2, 64, 128], [64], 2, isTrain)
            deconv_concat2 = tf.concat([conv22, deconv2], axis=4)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv71", reuse=reuse) as scope:
            conv71 = conv3d(deconv_concat2, [5, 5, 5, 128, 64], [64], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv72", reuse=reuse) as scope:
            conv72 = conv3d(conv71, [5, 5, 5, 64, 64], [64], keep_prob, act_func, isTrain)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("deconv1", reuse=reuse) as scope:
            deconv1 = deconv3d(conv72, [2, 2, 2, 32, 64], [32], 2, isTrain)
            deconv_concat1 = tf.concat([conv12, deconv1], axis=4)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv81", reuse=reuse) as scope:
            conv81 = conv3d(deconv_concat1, [5, 5, 5, 64, 32], [32], keep_prob, act_func, isTrain)
            scope.reuse_variables()
        with tf.compat.v1.variable_scope("conv82", reuse=reuse) as scope:
            conv82 = conv3d(conv81, [5, 5, 5, 32, 32], [32], keep_prob, act_func, isTrain)
            scope.reuse_variables()

        with tf.compat.v1.variable_scope("out", reuse=reuse) as scope:
            out_image = conv(conv82, [1, 1, 1, 32, 1], [1], isTrain)
            scope.reuse_variables()

    return out_image

