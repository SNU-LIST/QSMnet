import tensorflow as tf
import scipy.io
import time
import network_model

from utils import *

#
# Description :
#   Inference code of QSMnet and QSMnet+
#   Save susceptibility map in Matlab and NII format
# Outputs :
#   results_<network_name>.mat & results_<network_name>.nii
#   ppm unit
#
# Copyright @ Woojin Jung & Jaeyeon Yoon
# Laboratory for Imaging Science and Technology
# Seoul National University
# email : wjjung93@snu.ac.kr
#
'''
Network model
'''
network_name = 'QSMnet+_64'
net_model = 'qsmnet_deep'
sub_num = 1 #number of subjects in testset
act_func = 'leaky_relu'
'''
File Path
'''
FILE_PATH_INPUT = '../Data/Test/Input/test_input'
FILE_PATH_PRED = '../Data/Test/Prediction/'

def inf():
    f = scipy.io.loadmat('../Checkpoints/'+ network_name + '/' + 'norm_factor.mat')
    
    
    b_mean = f['input_mean']
    b_std = f['input_std']
    y_mean = f['label_mean']
    y_std = f['label_std']
    
    for i in range(1, sub_num + 1):
        input_data = scipy.io.loadmat(FILE_PATH_INPUT + str(i) +'.mat')
        tf.compat.v1.reset_default_graph()
        
        print('Subject number: ' + str(i))
        field = input_data["phs_tissue"]
        field = (field - b_mean) / b_std
        [pfield, N_difference, N] = padding_data(field)
        
        Z = tf.compat.v1.placeholder("float", [None, N[0], N[1], N[2], 1])
        keep_prob = tf.compat.v1.placeholder("float")
    
        net_func = getattr(network_model, net_model)
        feed_result = net_func(Z, act_func, False, False)
    
        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            print('##########Restore Network##########')
            saver.restore(sess, '../Checkpoints/'+ network_name + '/' + network_name + '-25')
            print('Done!')
            print('##########Inference...##########')
            result_im = y_std * sess.run(feed_result, feed_dict={Z: pfield, keep_prob: 1.0}) + y_mean
            result_im = crop_data(result_im.squeeze(), N_difference)
            
            display_slice_inf([52,72,92,112], result_im)
            print('##########Saving MATLAB & NII file...##########')
            scipy.io.savemat(FILE_PATH_PRED + '/subject' + str(i) + '_' + str(network_name) + '-25.mat', mdict={'sus': result_im})
            save_nii(result_im, FILE_PATH_PRED, 'subject' + str(i) + '_' + str(network_name)+'-25')
        print('All done!')

if __name__ == '__main__':
    start_time = time.time()
    inf()
    print("Total inference time : {} sec".format(time.time() - start_time))
    




