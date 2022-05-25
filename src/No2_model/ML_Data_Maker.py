# -*- coding: utf-8 -*-
"""
<vol_01_Train_Dataset_Maker>
@author: oze_tech

 - This program is preparing dataset for ML
   - wave                    -> spectrogram data
   - event data (time stamp) -> hopping event label (strong label)


<ML_Data_Maker>
======================================
1. Process wave to spectrogram
2. Process event_data to event_label


<NN_Dataset_Maker>
======================================
1. make Dataset
2. zero event label for noise wave 
3. list to ndarray
"""


import numpy as np
import librosa

# =============================================================================
# Class : ML_Data_Maker ( Data & Event Label )
# =============================================================================
    
class ML_Data_Maker:
    
    def __init__(self, input_shape  = ((64, 64,  1)), output_shape = ((64, 20))):  
        
        
        # data size
        self.cnn_x  =  input_shape[0]
        self.cnn_y  =  input_shape[1]
        self.cnn_ch = output_shape[1]
        
        # stft param
        self.fs    = 8000
        self.n_fft = 128
        self.hop_length = self.n_fft // 4
        
        # guard interval is at start and end point for test dataset
        self.guard_Dx = self.cnn_x //4
        self.hop_Dx   = self.cnn_x //2  # overlap size of real prosess
        
        
    
    # =========================================================================
    # 1. Process wave to spectrogram (x_train , x_test)     
    # =========================================================================
    def make_spec(self, in_wave):
        """
        input : signal wave
        output: spectrogram(amp), Dx:(x,y,z) = (64, 64, 1)
        """
        Dx = librosa.stft(in_wave, n_fft = self.n_fft)      # (y, x)
        Dx = np.abs(Dx).T                                   # (x, y) amplitude
        return Dx[:,:self.cnn_y,np.newaxis]                 # (x, y, 1)
        
    
    def data_norm(self, X):
        """
        normalize from 0 to 1
        """        
        X = (X-np.min(X))/np.max(X-np.min(X))
        return X
    
    
    # =========================================================================
    # 2. Process event_data to event_label (y_train , y_test)     
    # =========================================================================
    def make_hop_label(self, event_data):
        """
        input : event_data
        output: event_label of hopping pattern, Dy:(x, y) = (64, 20)
        """
        # zero padding
        sp_x = event_data[-1,3] // self.hop_length + 1      # cal X size
        Dy   = np.zeros((sp_x, self.cnn_ch))                # (64, 20)
        
        # anotate event data to event label
        for i in range(len(Dy)):    
            event = self.hop_length * i
            index = np.where((event_data[:,2] <= event ) & ( event < event_data[:,3] ))
            Dy[i, event_data[index, 1]] = 1                 # (i , 20)
        
        return Dy                                           # (64, 20)
    
   
        
        
                
     