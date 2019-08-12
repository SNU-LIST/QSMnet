import tensorflow as tf
import numpy as np
import scipy.io
import argparse
import os
import nibabel as nib

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def save_nii(data, save_folder, name):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    nifti_affine = np.array([[1,0,0,1], [0,1,0,1], [0,0,1,1], [0,0,0,1]], dtype=np.float)

    #data = data.squeeze().numpy()
    data = np.fliplr(data)
    data = np.pad(data, ((2, 2), (6, 7), (6, 7)), mode='constant')
    nifti = nib.Nifti1Image(data, affine=nifti_affine)
    nib.save(nifti, os.path.join(save_folder, name + '.nii.gz'))

def padding_data(input_field):
    N = np.shape(input_field)
    N_16 = np.ceil(np.divide(N,16.))*16
    N_dif = np.int16((N_16 - N) / 2)
    npad = ((N_dif[0],N_dif[0]),(N_dif[1],N_dif[1]),(N_dif[2],N_dif[2]))
    pad_field = np.pad(input_field, pad_width = npad, mode = 'constant', constant_values = 0)
    return pad_field, N_dif, N_16


def crop_data(result_pad, N_dif):
    result_pad = result_pad.squeeze()
    N_p = np.shape(result_pad)
    result_final  = result_pad[N_dif[0]:N_p[0]-N_dif[0],N_dif[1]:N_p[1]-N_dif[1],N_dif[2]:N_p[2]-N_dif[2]]
    return result_final

def load_module_func(module_name):
    mod = __import__('model.model_%s' %(module_name), fromlist=[module_name])
    return mod

def inf(dir_git, dir_patient, net_name):
    keep_prob = tf.placeholder("float")
    f = scipy.io.loadmat(str(dir_git) + '/' + str(net_name) + '/norm_factor.mat')
    b_mean = f['input_mean_total'].__array__().item()
    b_std = f['input_std_total'].__array__().item()
    y_mean = f['label_mean_total'].__array__().item()
    y_std = f['label_std_total'].__array__().item()

    input_data = scipy.io.loadmat(str(dir_patient) + '/inf_data.mat')
    field = input_data["field"]
    field = (field - b_mean) / b_std
    [pfield, N_difference, N] = padding_data(field)

    Z1 = tf.placeholder("float", [None, N[0], N[1], N[2], 1])
    arc = load_module_func(str(net_name))
    feed_result = arc.qsmnet_deep(Z1, keep_prob, False, False)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    sess = tf.Session(config=config)
    sess.run(init)
    print('Restore Network')
    saver.restore(sess, str(dir_git) + '/' + str(net_name) + "/network.ckpt")
    print('Done!')
    pfield = np.expand_dims(pfield, axis=0)
    pfield = np.expand_dims(pfield, axis=4)
    Mask = input_data["Mask"]

    res = y_std * sess.run(feed_result, feed_dict={Z1: pfield, keep_prob: 1.0}) + y_mean
    res = np.multiply(crop_data(res.squeeze(), N_difference),Mask)

    print("Saving MATLAB & NII file...")
    scipy.io.savemat(str(dir_p) + '/result_' + str(net_name) + '.mat', mdict={'sus': res})
    save_nii(res, dir_p, 'result_' + str(net_name))

    print("All done!")

dir = './'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_patient',type=str, help="directory for patient data")
    parser.add_argument('n_name',type=str, help="network name")
    args = parser.parse_args()
    dir_p = args.dir_patient
    network_name = args.n_name
    inf(dir, dir_p, network_name)




