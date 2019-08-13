import tensorflow as tf

#
# Description :
#   3D U-net architecture of QSMnet+
#
# Copyright @ Woojin Jung & Jaeyeon Yoon
# Laboratory for Imaging Science and Technology
# Seoul National University
# email : wjjung93@snu.ac.kr
#



def batch_norm(x, channel, isTrain, decay=0.99, name="bn"):

   with tf.variable_scope(name):
      beta = tf.get_variable(initializer=tf.constant(0.0, shape=[channel]), name='beta')
      gamma = tf.get_variable(initializer=tf.constant(1.0, shape=[channel]), name='gamma')
      batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2, 3], name='moments')
      mean_sh = tf.get_variable(initializer=tf.zeros([channel]), name="mean_sh", trainable=False)
      var_sh = tf.get_variable(initializer=tf.ones([channel]), name="var_sh", trainable=False)

      def mean_var_with_update():
         mean_assign_op = tf.assign(mean_sh, mean_sh * decay + (1 - decay) * batch_mean)
         var_assign_op = tf.assign(var_sh, var_sh * decay + (1 - decay) * batch_var)
         with tf.control_dependencies([mean_assign_op, var_assign_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

      mean, var = tf.cond(tf.cast(isTrain, tf.bool), mean_var_with_update, lambda: (mean_sh, var_sh))
      normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3, name="normed")

   return normed

def conv3d(x, w_shape, b_shape, keep_prob_, train, isTrain):
    weights = tf.get_variable("conv_weights", w_shape,
                              initializer=tf.contrib.layers.xavier_initializer(), trainable=train)
    conv_3d = tf.nn.conv3d(x, weights, strides=[1, 1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable("biases", b_shape,
                             initializer=tf.random_normal_initializer(), trainable=train)

    conv_3d = tf.nn.bias_add(conv_3d, biases)

    channel = conv_3d.get_shape().as_list()[-1]
    #print(channel)

    bn_x = batch_norm(conv_3d, channel, isTrain)

    #return tf.nn.relu(bn_x)
    return tf.nn.leaky_relu(bn_x, alpha = 0.1)

def conv(x, w_shape, b_shape, train):
    weights = tf.get_variable("weights", w_shape,
                              initializer=tf.contrib.layers.xavier_initializer(), trainable=train)
    biases = tf.get_variable("biases", b_shape,
                             initializer=tf.random_normal_initializer(), trainable=train)
    return tf.nn.conv3d(x, weights, strides=[1, 1, 1, 1, 1], padding='SAME') + biases

def deconv3d(x, w_shape, b_shape, stride, train):
    x_shape = tf.shape(x)
    weights = tf.get_variable("deconv_weights", w_shape,
                              initializer=tf.contrib.layers.xavier_initializer(), trainable=train)
    biases = tf.get_variable("biases", b_shape,
                             initializer=tf.random_normal_initializer(), trainable=train)

    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4] // 2])
    return tf.nn.conv3d_transpose(x, weights, output_shape, strides=[1, stride, stride, stride, 1],
                                  padding='SAME') + biases

def max_pool(x, n):
    return tf.nn.max_pool3d(x, ksize=[1, n, n, n, 1], strides=[1, n, n, n, 1], padding='SAME')

def avg_pool(x, n):
    return tf.nn.avg_pool3d(x, ksize=[1, n, n, n, 1], strides=[1, n, n, n, 1], padding='SAME')

def crop_and_concat(x2,x1):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, abs((x1_shape[1] - x2_shape[1])) // 2, abs(x1_shape[2] - x2_shape[2]) // 2, abs(x1_shape[3] - x2_shape[3]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 4)

def qsmnet_deep(x, keep_prob, reuse, isTrain):
    with tf.variable_scope("qsmnet", reuse=reuse) as scope:
        with tf.variable_scope("conv11", reuse=reuse) as scope:
            conv11 = conv3d(x, [5, 5, 5, 1, 32], [32], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("conv12", reuse=reuse) as scope:
            conv12 = conv3d(conv11, [5, 5, 5, 32, 32], [32], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("maxpool1", reuse=reuse) as scope:
            pool1 = max_pool(conv12, 2)
            scope.reuse_variables()

        with tf.variable_scope("conv21", reuse=reuse) as scope:
            conv21 = conv3d(pool1, [5, 5, 5, 32, 64], [64], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("conv22", reuse=reuse) as scope:
            conv22 = conv3d(conv21, [5, 5, 5, 64, 64], [64], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("maxpool2", reuse=reuse) as scope:
            pool2 = max_pool(conv22, 2)
            scope.reuse_variables()

        with tf.variable_scope("conv31", reuse=reuse) as scope:
            conv31 = conv3d(pool2, [5, 5, 5, 64, 128], [128], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("conv32", reuse=reuse) as scope:
            conv32 = conv3d(conv31, [5, 5, 5, 128, 128], [128], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("maxpool3", reuse=reuse) as scope:
            pool3 = max_pool(conv32, 2)
            scope.reuse_variables()

        with tf.variable_scope("conv41", reuse=reuse) as scope:
            conv41 = conv3d(pool3, [5, 5, 5, 128, 256], [256], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("conv42", reuse=reuse) as scope:
            conv42 = conv3d(conv41, [5, 5, 5, 256, 256], [256], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("maxpool4", reuse=reuse) as scope:
            pool4 = max_pool(conv42, 2)
            scope.reuse_variables()

        with tf.variable_scope("l_conv1", reuse=reuse) as scope:
            l_conv1 = conv3d(pool4, [5, 5, 5, 256, 512], [512], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("l_conv2", reuse=reuse) as scope:
            l_conv2 = conv3d(l_conv1, [5, 5, 5, 512, 512], [512], keep_prob, True, isTrain)
            scope.reuse_variables()

        with tf.variable_scope("deconv4", reuse=reuse) as scope:
            deconv4 = deconv3d(l_conv2, [2, 2, 2, 256, 512], [256], 2, True)
            deconv_concat4 = tf.concat([conv42, deconv4], axis=4)
            scope.reuse_variables()
        with tf.variable_scope("conv51", reuse=reuse) as scope:
            conv51 = conv3d(deconv_concat4, [5, 5, 5, 512, 256], [256], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("conv52", reuse-reuse) as scope:
            conv52 = conv3d(conv51, [5, 5, 5, 256, 256], [256], keep_prob, True, isTrain)
            scope.reuse_variables()

        with tf.variable_scope("deconv3", reuse=reuse) as scope:
            deconv3 = deconv3d(conv52, [2, 2, 2, 128, 256], [128], 2, True)
            deconv_concat3 = tf.concat([conv32, deconv3], axis=4)
            scope.reuse_variables()
        with tf.variable_scope("conv61", reuse=reuse) as scope:
            conv61 = conv3d(deconv_concat3, [5, 5, 5, 256, 128], [128], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("conv62", reuse=reuse) as scope:
            conv62 = conv3d(conv61, [5, 5, 5, 128, 128], [128], keep_prob, True, isTrain)
            scope.reuse_variables()

        with tf.variable_scope("deconv2", reuse=reuse) as scope:
            deconv2 = deconv3d(conv62, [2, 2, 2, 64, 128], [64], 2, True)
            deconv_concat2 = tf.concat([conv22, deconv2], axis=4)
            scope.reuse_variables()
        with tf.variable_scope("conv71", reuse=reuse) as scope:
            conv71 = conv3d(deconv_concat2, [5, 5, 5, 128, 64], [64], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("conv72", reuse=reuse) as scope:
            conv72 = conv3d(conv71, [5, 5, 5, 64, 64], [64], keep_prob, True, isTrain)
            scope.reuse_variables()

        with tf.variable_scope("deconv1", reuse=reuse) as scope:
            deconv1 = deconv3d(conv72, [2, 2, 2, 32, 64], [32], 2, True)
            deconv_concat1 = tf.concat([conv12, deconv1], axis=4)
            scope.reuse_variables()
        with tf.variable_scope("conv81", reuse=reuse) as scope:
            conv81 = conv3d(deconv_concat1, [5, 5, 5, 64, 32], [32], keep_prob, True, isTrain)
            scope.reuse_variables()
        with tf.variable_scope("conv82", reuse=reuse) as scope:
            conv82 = conv3d(conv81, [5, 5, 5, 32, 32], [32], keep_prob, True, isTrain)
            scope.reuse_variables()

        with tf.variable_scope("out", reuse=reuse) as scope:
            out_image = conv(conv82, [1, 1, 1, 32, 1], [1], True)  # output = [10,512,512,9] #segment_data = [10,512,512]
            scope.reuse_variables()

    return out_image

