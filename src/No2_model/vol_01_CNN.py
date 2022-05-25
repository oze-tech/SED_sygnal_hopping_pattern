# -*- coding: utf-8 -*-
"""
<vol_01_CNN>
@author: oze_tech

( x,  y, ch) == (64, 64,  1) -> (64, 16, 32) -> ( 64, 4, 64) -> ( 64, 1, 128) -> (64, 128) -> (64, 40)

===============
1. Define
2. Predict test data
==============="""

import os

from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, MaxPooling2D, Dropout
from tensorflow.keras.layers import Reshape, TimeDistributed, Dense
from tensorflow.keras.models import Model

class CNN:
        
    # =========================================================================
    # 1. Define CNN
    # =========================================================================
        
    def __init__(self, weights_name = "", input_shape  = ((64, 64,  1)),
                 output_shape = ((64, 20)), cnn_param = [32, 64, 128] , param_init = False): 
        """
        weights_name : file name,
        input_shape  : train data shape,
        output_shape : label data shape,
        cnn_param    : CNN ch parameter,
        param_init   : reset parameter? True or False
        """
        
        self.cnn_x  =  input_shape[0]
        self.cnn_y  =  input_shape[1]
        self.cnn_ch = output_shape[1]
        
        def create_cnn_block(x,ch):
            x = Conv2D(ch, 3, activation = "relu", padding="same")(x)
            x = BatchNormalization()(x)
            x = Dropout(0.25)(x)
            x = MaxPooling2D((1, 4),strides=(1, 4))(x)
            return x
        
        self.Lin = Input(input_shape)  
        
        for i,Cp in enumerate(cnn_param):            
            if i==0:
                self.Lcnn = create_cnn_block(self.Lin,  Cp)
            else:
                self.Lcnn = create_cnn_block(self.Lcnn, Cp)


        # case of FC layer
        self.Lfnn = Reshape((self.cnn_x , Cp))(self.Lcnn)
        self.Lfnn = TimeDistributed(Dense(Cp, activation = "linear"))(self.Lfnn) 
        self.Lfnn = TimeDistributed(Dense(self.cnn_ch, activation = "sigmoid"))(self.Lfnn) 
        
        
        self.model = Model( inputs = self.Lin, outputs = self.Lfnn)
        
                
        # saving
        self.weight_path = (os.path.dirname(os.path.abspath(__file__)) 
                            + "/weights/CNN_Lc"+str(len(cnn_param)) 
                            + "_" + weights_name
                            + "_weights.h5")
        
        
        if os.path.isfile(self.weight_path) and not param_init:
            print(" == CNN model is loading a weight==")
            self.model.load_weights( self.weight_path)
        else:
            print("! CNN model is not loading weighnt !")


    # =========================================================================
    # 2.predict test data
    # =========================================================================
    def __call__(self, x_test):
        return self.model.predict(x_test)
    
    def show_model(self):       
        self.model.summary()
        


if __name__ == '__main__':
    
    
    cnn = CNN()
    cnn.show_model()
    #cnn(x_test)        
            
        