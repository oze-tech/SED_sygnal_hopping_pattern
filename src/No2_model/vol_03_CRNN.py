# -*- coding: utf-8 -*-
"""
<vol_03_CRNN>
@author: oze_tech

( x,  y, ch) : 
CNN    : (64, 64,  1) -> (64, 16, 32) -> ( 64, 4, 64) -> ( 64, 1, 128) ->
reshape: (64, 128) ->
RNN    : (64, 64(32)) -> (64, 64(32)) ->
FNN    : (64, 32) -> (64, 20) 

===============
1. Define
2. Predict test data
==============="""

import os

from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, MaxPooling2D, Dropout
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.layers import Reshape, TimeDistributed, Dense
from tensorflow.keras.models import Model


class CRNN:
    
    
    # =========================================================================
    # 1. Define CRNN
    # =========================================================================
        
    def __init__(self, weights_name = "", input_shape  = ((64, 64,  1)),
                 output_shape = ((64, 20)), cnn_param = [32, 64, 128] , rnn_param = [32, 32], param_init = False): 
        """
        weights_name : file name,
        input_shape  : train data shape,
        output_shape : label data shape,
        cnn_param    : CNN ch parameter, length = leyer num
        rnn_param    : RNN ch parameter, length = leyer num
        param_init   : reset parameter? True or False
        """
        
        self.cnn_x  =  input_shape[0]
        self.cnn_y  =  input_shape[1]
        self.fnn_ch = output_shape[1]
        
        def create_cnn_block(x,ch):
            x = Conv2D(ch, 3, activation = "relu", padding="same")(x)
            x = BatchNormalization()(x)
            x = Dropout(0.25)(x)
            x = MaxPooling2D((1, 4),strides=(1, 4))(x)
            return x
        
        def create_rnn_block(x,ch):
            return Bidirectional(GRU(ch, return_sequences=True))(x)
               
        
        
        self.Lin = Input(input_shape)  
        
        # CNN layer
        for i,Cp in enumerate(cnn_param):
            
            if i==0:
                self.Lcnn = create_cnn_block(self.Lin, Cp)
            else:
                self.Lcnn = create_cnn_block(self.Lcnn, Cp)
                
        self.Lcnn = Reshape((self.cnn_x , 1*Cp))(self.Lcnn) #　1は縦の大きさ
            
                
        # RNN layer
        for i,Rp in enumerate(rnn_param):
            
            if i==0:
                self.Lrnn = create_rnn_block(self.Lcnn, Rp)
            else:
                self.Lrnn = create_rnn_block(self.Lrnn, Rp)
        
        # FNN layer
        self.Lfnn = TimeDistributed(Dense(Rp*2, activation = "linear"))(self.Lrnn) 
        self.Lfnn = TimeDistributed(Dense(self.fnn_ch, activation = "sigmoid"))(self.Lfnn) 
        
        
        self.model = Model( inputs = self.Lin, outputs = self.Lfnn)
        
                
        # saving
        self.weight_path = (os.path.dirname(os.path.abspath(__file__)) 
                            + "/weights/CRNN_Lc" + str(len(cnn_param))
                            + "_Lr"+str(len(rnn_param)) 
                            + "_" + weights_name
                            + "_weights.h5")
        
        
        if os.path.isfile(self.weight_path) and not param_init:
            print(" == CRNN model is loading a weight ==")
            self.model.load_weights(self.weight_path)
        else:
            print("! CNRN model is not loading weighnt !")


    # =========================================================================
    # 2.predict test data
    # =========================================================================
    def __call__(self, x_test):
        return self.model.predict(x_test)
    
    def show_model(self):       
        self.model.summary()
        


if __name__ == '__main__':
    
    
    crnn = CRNN()
    crnn.show_model()
      
            
        