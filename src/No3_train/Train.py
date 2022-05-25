# -*- coding: utf-8 -*-
"""
<Train>
@author: oze_tech

this cord is for simulation training

===============
1. Prepare Dataset
2. Define model
3. Training
==============="""

import os
import sys  
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from No1_train_data import Simulation_Wave_Maker as Smaker
from No2_model      import vol_01_CNN            as cnn
from No2_model      import vol_02_RNN            as rnn
from No2_model      import vol_03_CRNN           as crnn
from No2_model      import ML_Data_Maker         as Dset
from No3_train.func import vol_01_train          as func1

# =============================================================================
# 01.prepare dataset
# =============================================================================
class Train_Dataset_Maker(Dset.ML_Data_Maker):
    
    def __init__(self, dataset_len = 512, input_shape  = (64, 64, 1), output_shape = (64, 20)):  
        
        # class : ML_Data_Maker
        super().__init__(input_shape, output_shape)
        
        # save Dataset
        self.Dx_set = []                # spectrogram
        self.Dy_set = []                # event label
        self.Dh_set = []                # hopping event label
        
        self.dataset_len = dataset_len
        
    
    
    # =========================================================================
    # 1.make Dataset for train 
    # =========================================================================
    
    def train_set(self, wave, event, seedi=0):
                
        # prepare data and label
        self.Dx = self.make_spec(wave)               # (x,y,z) = (64, 64, 1)
        self.Dy = self.make_hop_label(event)         # (x,y)   = (64, 20)
    
        # get "batch_size" cut_point by random
        np.random.seed(seed = seedi)
        self.cut_point = np.random.randint(0, self.Dx.shape[0] - self.cnn_x, self.dataset_len)  
            
        # cutting
        for i_s in self.cut_point:
                        
            i_e = i_s + self.cnn_x 
            
            Y = self.Dy[i_s : i_e ]
            X = self.Dx[i_s : i_e ]
            X = self.data_norm(X)
            
            self.Dy_set.append(Y)
            self.Dx_set.append(X)
            
    # =========================================================================
    # 2. zero event label for noise wave 
    # =========================================================================
    def dataset0maker(self, noise_wave, batch_size = 512):

        # Define Noise label
        Y   = np.zeros((self.cnn_x, self.cnn_ch))
        
        # Noise spec
        Dx = self.make_spec(noise_wave)
        
        # Cut point
        np.random.seed(seed = 0)
        step = np.random.randint(0, Dx.shape[1] // self.cnn_x, batch_size)
        
        for i in step:
            
            i_s = self.cnn_x * i
            i_e = self.cnn_x *(i+1)
            
            X = self.data_norm(Dx[i_s : i_e])
            
            self.Dx_set.append(X)
            self.Dy_set.append(Y)
            
            
            
    # =========================================================================
    # 3. list to ndarray
    # =========================================================================
            
    def get_dataset(self, show=True):
        
        x = np.array(self.Dx_set)
        y = np.array(self.Dy_set)
        
        self.Dx_set, self.Dy_set = [], []
        
        if show:
            print(" == Dataset will be prepared ==")
            print("  data  shape : ", x.shape, ", max : ",x.max(),", min : ", x.min())
            print("  label shape : ", y.shape, ", max : ",x.max(),", min : ", x.min())
            
        return x, y
    
    
# =============================================================================
# 02.train
# =============================================================================
class sim_training(Train_Dataset_Maker):
    """
    input
        overlap     : Int (1, 1/2, 1/4)
        dataset_len : Int, The default is 512*512.
        batch       : Int, The default is 256.
        epoch       : Int, The default is 20.
    """
    def __init__(self, overlap, batch = 512, epoch = 20) :
        
        # class : Train_Dataset_Maker
        super().__init__()
        
        
        # model param
        self.overlap     = overlap
        self.epoch       = epoch
        self.batch       = batch
        self.dataset_len = 10000        
        self.SNR         = np.append(np.arange(-5, 10), np.arange(20, 101, 20))
        self.dataset_len = 200000
        self.snr_len     = self.dataset_len // len(self.SNR)
        self.weight_name = "simulation_overlap_{0:03}_per_".format(100-int(100 * self.overlap))
        
        
        # =====================================================================
        # 1. Prepare Dataset 
        # =====================================================================
    
        for i, snr in enumerate(self.SNR):
            
            sim = Smaker.simulation_signal(self.overlap, self.dataset_len, i)
            wave, event = sim(snr) 
            self.train_set(wave, event, i)
            
        # noise only
        self.dataset0maker(np.random.random(sim.wave_size))
            
        # list -> ndarray
        self.x_train, self.y_train = self.get_dataset()
        
        
        
    def __call__(self, model_class):
        
        
        # =====================================================================
        # 2. Define model
        # =====================================================================
        
        self.model        = model_class.model
        self.weight_path  = model_class.weight_path
        
        
        # =====================================================================
        # 3. Train
        # =====================================================================
        func1.train(self.model, 
                    self.x_train, 
                    self.y_train, 
                    self.weight_path, 
                    self.epoch, 
                    self.batch,
                    auto_shuffle = True
                   )
        
        func1.show_val_curve(self.weight_path)
        
        
        
def train():
    
    for lap in [1, 1/2, 1/4]:  
        

        train = sim_training(lap)
        
        CNN  = cnn.CNN(  train.weight_name,  param_init = True)    
        RNN  = rnn.RNN(  train.weight_name,  param_init = True)    
        CRNN = crnn.CRNN(train.weight_name,  param_init = True)    
        
        train(CNN)
        train(RNN)
        train(CRNN)
        
        
if __name__ == '__main__':
    
    train()
    