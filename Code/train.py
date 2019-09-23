import tensorflow as tf
import numpy as np
import time

from training_params import *    
from utils import *
from network_model import *

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
    predX = qsmnet_toy(X, 'relu', keep_prob, False, True) # Prediction result of training data
    predX_val = qsmnet_toy(X_val, 'relu', keep_prob, True, False) # Prediction result of validation data
            
    #%% Definition of loss function
    loss = l1(predX, Y) # training loss
    loss_val = l1(predX_val,Y_val) # validation loss
           
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
