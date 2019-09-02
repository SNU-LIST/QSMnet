import tensorflow as tf

#
# Description:
#  Loss function for training QSMnet and QSMnet+
#
#  Copyright @ Woojin Jung & Jaeyeon Yoon
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : wjjung93@snu.ac.kr
#

def loss1(x, y, f, d):
    #   l2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y - xsa), [1, 2, 3, 4])))

    f2 = d

    gen_out = f - f2

    l21 = tf.reduce_mean(tf.reduce_sum(tf.square(gen_out), [1, 2, 3, 4]))

    l22 = tf.reduce_mean(tf.reduce_sum(tf.square(x - y), [1, 2, 3, 4]))

    return l22

def loss2(x, y, f, d):
    dx_gen2 = x[:, 1:, :, :, :] - x[:, :-1, :, :, :]

    dx_real2 = y[:, 1:, :, :, :] - y[:, :-1, :, :, :]

    dx2 = tf.abs(tf.abs(dx_gen2) - tf.abs(dx_real2))
    # dx2 = tf.abs(dx_gen2 - dx_real2) + 1e-8
    dx2 = tf.pow(dx2, 2)

    dx2 = tf.reduce_sum(dx2, [1, 2, 3, 4])
    # print(dx.shape)

    dy_gen2 = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]

    dy_real2 = y[:, :, 1:, :, :] - y[:, :, :-1, :, :]

    dy2 = tf.abs(tf.abs(dy_gen2) - tf.abs(dy_real2))
    # dy2 = tf.abs(dy_gen2 - dy_real2) + 1e-8
    dy2 = tf.pow(dy2, 2)
    dy2 = tf.reduce_sum(dy2, [1, 2, 3, 4])

    dz_gen2 = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]

    dz_real2 = y[:, :, :, 1:, :] - y[:, :, :, :-1, :]

    dz2 = tf.abs(tf.abs(dz_gen2) - tf.abs(dz_real2))
    # dz2 = tf.abs(dz_gen2 - dz_real2) + 1e-8
    dz2 = tf.pow(dz2, 2)
    dz2 = tf.reduce_sum(dz2, [1, 2, 3, 4])

    dd2 = dx2 + dy2 + dz2

    gdl2 = tf.reduce_mean(dd2)

    return gdl2

def relu3dgdlloss(x, y):
    x_cen = x[:, 1:-1, 1:-1, 1:-1, :]
    x_shape = tf.shape(x)
    grad_x = tf.zeros_like(x_cen)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                x_slice = tf.slice(x, [0, i+1, j+1, k+1, 0], [x_shape[0], x_shape[1]-2, x_shape[2]-2, x_shape[3]-2, x_shape[4]])
                if i*i + j*j + k*k == 0:
                    temp = tf.zeros_like(x_cen)
                else:
                    temp = tf.scalar_mul(1.0 / tf.sqrt(tf.cast(i * i + j * j + k * k, tf.float32)), tf.nn.relu(x_slice - x_cen))
                grad_x = grad_x + temp

    y_cen = y[:, 1:-1, 1:-1, 1:-1, :]
    y_shape = tf.shape(y)
    grad_y = tf.zeros_like(y_cen)
    for ii in range(-1, 2):
        for jj in range(-1, 2):
            for kk in range(-1, 2):
                y_slice = tf.slice(y, [0, ii + 1, jj + 1, kk + 1, 0], [y_shape[0], y_shape[1] - 2, y_shape[2] - 2, y_shape[3] - 2, y_shape[4]])
                if ii*ii + jj*jj + kk*kk == 0:
                    temp = tf.zeros_like(y_cen)
                else:
                    temp = tf.scalar_mul(1.0 / tf.sqrt(tf.cast(ii * ii + jj * jj + kk * kk, tf.float32)), tf.nn.relu(y_slice - y_cen))
                grad_y = grad_y + temp

    gd = tf.abs(grad_x - grad_y)
    gdl = tf.reduce_mean(gd, [1, 2, 3, 4])
    gdl = tf.reduce_mean(gdl)
    return gdl

def l2(f, d):

    l21 = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(f - d), [1, 2, 3, 4]) + 1e-8))
    #l21 = tf.reduce_mean(tf.reduce_mean(tf.square(f - d), [1, 2, 3, 4]))

    return l21

def l1(f, d):

    l1 = tf.reduce_mean(tf.reduce_mean(tf.abs(f - d), [1, 2, 3, 4]))

    return l1

def model_loss(pred, x, m, input_std,input_mean,label_std,label_mean):
    pred_sc = pred * label_std + label_mean
    x2 = tf.complex(pred_sc, tf.zeros_like(pred_sc))
    x2 = tf.transpose(x2, perm=[0, 4, 1, 2, 3])
    x2k = tf.fft3d(x2)

    d2 = tf.complex(D, tf.zeros_like(D))
    d2 = tf.transpose(d2, perm=[0, 4, 1, 2, 3])
    fk = tf.multiply(x2k, d2)

    f2 = tf.ifft3d(fk)
    f2 = tf.transpose(f2, perm=[0, 2, 3, 4, 1])
    f2 = tf.real(f2)

    slice_f = tf.multiply(f2, m)
    X_c = (x * input_std) + input_mean
    X_c2 = tf.multiply(X_c, m)
    return l1(X_c2, slice_f)

def total_loss(pred, x, y, m, w1, w2, input_std, input_mean, label_std, label_mean):
    l1loss = l1(pred, y)
    mdloss = model_loss(pred, x, m, input_std, input_mean, label_std, label_mean)
    gdloss = relu3dgdlloss(pred, y)
    tloss = l1loss + mdloss * w1 + gdloss * w2
    return l1loss, mdloss, gdloss, tloss
