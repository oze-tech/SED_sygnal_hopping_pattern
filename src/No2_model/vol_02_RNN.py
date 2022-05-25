# -*- coding: utf-8 -*-
"""
<vol_02_RNN>
@author: oze_tech

( x,  y, ch) : 
CNN    : (64, 64,  1) -> 
reshape: (64, 64) ->
RNN    : (64, 64(32)) -> (64, 64(32)) ->
FNN    : (64, 32) -> (64, 40) 

===============
1. Define
2. Predict test data
==============="""

import os

from tensorflow.keras.layers import Input, Bidirectional, GRU
from tensorflow.keras.layers import Reshape, TimeDistributed, Dense
from tensorflow.keras.models import Model


class RNN:
    
    
    # =========================================================================
    # 1. Define RNN
    # =========================================================================
        
    def __init__(self, weights_name = "", input_shape  = ((64, 64,  1)),
                 output_shape = ((64, 20)), rnn_param = [32, 32], param_init = False): 
        """
        weights_name = file name,
        input_shape  = train data shape,
        output_shape = label data shape,
        rnn_param    = RNN ch parameter, length = leyer num
        param_init   : reset parameter? True or Falss
        """
        
        self.cnn_x  =  input_shape[0]
        self.cnn_y  =  input_shape[1]
        self.fnn_ch = output_shape[1]
        
        
        def create_rnn_block(x,units):
            # x = (time_step, feature_size) : width, height
            # https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU#call_arguments_2
            # <ex> https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU#for_example_2
            
            return Bidirectional(GRU(units, return_sequences=True))(x)
               
        
        self.Lin = Input(input_shape)                               # (64,64,1)
        self.Lrnn = Reshape((self.cnn_x , self.cnn_y))(self.Lin)    # (64,64)
            
                
        # RNN layer
        for Rp in rnn_param: 
            self.Lrnn = create_rnn_block(self.Lrnn, Rp)
           
        # FNN layer
        self.Lfnn = TimeDistributed(Dense(Rp*2, activation = "linear"))(self.Lrnn) 
        self.Lfnn = TimeDistributed(Dense(self.fnn_ch, activation = "sigmoid"))(self.Lfnn) 
        
        
        self.model = Model( inputs = self.Lin, outputs = self.Lfnn)
        
        # saving
        self.weight_path = (os.path.dirname(os.path.abspath(__file__)) 
                            + "/weights/RNN_Lr"+str(len(rnn_param))
                            + "_" + weights_name
                            + "_weights.h5")
        
        
        if os.path.isfile(self.weight_path) and not param_init:
            print(" == RNN model is loading a weight==")
            self.model.load_weights( self.weight_path)
        else:
            print("! RNN model is not loading weighnt !")


    # =========================================================================
    # 2.predict test data
    # =========================================================================
    def __call__(self, x_test):
        return self.model.predict(x_test)
    
    def show_model(self):       
        self.model.summary()
        


if __name__ == '__main__':
    
    
    rnn = RNN()
    rnn.show_model()
      
            
        