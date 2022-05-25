# -*- coding: utf-8 -*-
"""
<vol_01_train_model>
@author: oze_tech
                
this cord is function for machine Learning
                
=============== 
1. train        
2. show curve   
=============== 
                
"""             
                
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall




#==============================================================================
# 1.train       
#==============================================================================

def shuffle_batch(X):
    """
    input :
        X : shuffled target 
    """
    shuffle_idx = np.arange(len(X))
    np.random.seed(seed = 0)
    np.random.shuffle(shuffle_idx)
    return X[shuffle_idx]
    

def train(model, x_train, y_train, weight_path = "", fin_epochs = 50, batch_size = 256, auto_shuffle = False):
        
    """
    input:
        model       : set by keras
        x_train     : train data (batch, x_size, y_size, ch_size)
        y_train     : label data (batch, x_size, y_size)
        weight_path : save path
        fin_epochs  : epoch
        batch_size  : min_batch size
    """
    
    if auto_shuffle:
        x_train = shuffle_batch(x_train)
        y_train = shuffle_batch(y_train)
        
        
    print(" == Learning will be started == ")
    # precision = tp/(tp+fp), tp:true positive, fp:false positive : best is 1
    # recall    = tp/(tp+fn), tp:true positive, fn:false negative : best is 1
    
    model.compile(
            loss      = 'binary_crossentropy',
            optimizer = 'adam',
            metrics   = ['binary_accuracy', Precision(), Recall()] 
        )
    
    # training
    history = model.ﬁt(
        x_train, y_train, 
        batch_size = batch_size, 
        epochs = fin_epochs,
        verbose = 1,
        validation_split = 0.2,
        )

    # saving       
    if weight_path != "":
        
        
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv( weight_path + "learn_curve.csv")
        model.save_weights(weight_path + "best_weights.h5")
    


#==============================================================================
# 2.show train curve
#==============================================================================

def show_val_curve(weight_path = "", Ylabel = ["Loss",'binary_accuracy',"Precision","Recall"]): #
    """
    weight_path : save path
    Ylabel      : plt.ylabel name
    """
    
    load_his = pd.read_csv(weight_path + "learn_curve.csv", index_col=0)
    plt_his  = load_his.values.T
    
    
    plt.figure()
    for i,ylabel in enumerate(Ylabel):
        plt.subplot(len(Ylabel),1,i+1)
        plt.plot(plt_his[i],   label = 'train', color = 'black')
        plt.plot(plt_his[len(Ylabel)+i], label = 'test',  color = 'red')
        plt.ylabel(ylabel)
        plt.xlabel('Epoch')
        plt.ylim(0,1)
        plt.legend()
        plt.tight_layout()
        
    plt.savefig(weight_path + "learn_curve.png")
    
