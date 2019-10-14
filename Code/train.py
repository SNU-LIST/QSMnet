import tensorflow as tf
import numpy as np
import time
import network_model
from training_params import *    
from utils import *

#
# Description:
#  Training code of QSMnet and QSMnet+
#
#  Copyright @ Woojin Jung & Jaeyeon Yoon
#  Laboratory for Imaging Science and Technology
#  Seoul National University
#  email : wjjung93@snu.ac.kr
#


#%% Train
def train():
    tf.compat.v1.reset_default_graph()
    #%% Loading dataset
    train_dataset = dataset() # Training set, validation set

    #%% Declaration of tensor
    X = tf.compat.v1.placeholder("float", [None, PS, PS, PS, 1]) # Training input
    Y = tf.compat.v1.placeholder("float", [None, PS, PS, PS, 1]) # Training label
        
    N = np.shape(train_dataset.tefield) # matrix size of validation set 
    X_val = tf.compat.v1.placeholder("float", [None, N[1], N[2], N[3], 1]) # Validation input
    Y_val = tf.compat.v1.placeholder("float", [None, N[1], N[2], N[3], 1]) # Validation label
    keep_prob = tf.compat.v1.placeholder("float") #dropout rate

    #%% Definition of model
    net_func = getattr(network_model, net_model)
    predX = net_func(X, act_func, False, True)
    predX_val = net_func(X_val, act_func, True, False)
    
    #%% Definition of loss function
    loss = l1(predX, Y)
    loss_val = l1(predX_val,Y_val)
           
    #%% Definition of optimizer
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    #%% Generate saver instance
    qsm_saver = tf.compat.v1.train.Saver()
    
    #%% Running session
    Training_network(train_dataset, X, Y, X_val, Y_val, predX_val, loss, loss_val, train_op, keep_prob, qsm_saver)

if __name__ == '__main__':
    start_time = time.time()
    train()
    print("Total training time : {} sec".format(time.time() - start_time))




#%%
