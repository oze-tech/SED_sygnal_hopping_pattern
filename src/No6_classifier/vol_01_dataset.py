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
    
   
        
        
        
# =============================================================================
# Class : CNN_Dataset_Maker ( Data set & Event Label set)
# =============================================================================

class ML_Dataset_Maker(ML_Data_Maker):
    
    def __init__(self, batch_size = None, input_shape  = (64, 64,  1), output_shape = (64, 20)):  
    
        """
        input
            batch_size   : dataset length (None -> zeropadding, Number(int)-> Augmentation until Num)
            input_shape  : train dataset, default:(64,64,1)
            output_shape : train dataset, default:(64,40)
            hop_label    : (bool), make a hopping label?
        """
        # class : symbol_making
        super().__init__(input_shape, output_shape)
        
        # save Dataset
        self.Dx_set = []                # spectrogram
        self.Dy_set = []                # event label
        self.Dh_set = []                # hopping event label
        
        self.batch_size = batch_size
        
        # guard interval is at start and end point for test dataset
        self.guard_Dx = self.cnn_x //4
        self.hop_Dx   = self.cnn_x //2  # overlap size of real prosess
        

        
    def __call__(self, wave):
        """
        input
            wave     : signal wave
            event    : event_data ( if None -> zero Dy)
            set_size : dataset length (None -> zeropadding, Number(int)-> Augmentation until Num)
        """
        
        self.test_data_set(wave)
        self.cut_and_append()
        x = np.array(self.Dx_set)
        self.Dx_set = []
        return x
        
    # =========================================================================
    # 1.make Dataset for train 
    # =========================================================================
    def cut_and_append(self):
        
        # cutting
        for i_s in self.cut_point:
                        
            i_e = i_s + self.cnn_x 
            
            Y = self.Dy[i_s : i_e ]
            X = self.Dx[i_s : i_e ]
            X = self.data_norm(X)
            
            self.Dy_set.append(Y)
            self.Dx_set.append(X)
        
    # make a dataset for train 
    def train_set(self, wave, event, seedi=0):
                
        # prepare data and label
        self.Dx = self.make_spec(wave)               # (x,y,z) = (64, 64, 1)
        self.Dy = self.make_hop_label(event)         # (x,y)   = (64, 20)
    
        # get "batch_size" cut_point by random
        np.random.seed(seed = seedi)
        self.cut_point = np.random.randint(0, self.Dx.shape[0] - self.cnn_x, self.batch_size)  
            
        # cutting
        self.cut_and_append()
        
        
    # =========================================================================
    # 2.make Dataset for test
    # =========================================================================
      
    def test_data_set(self, wave):
        
        # connect each spectrogram 
        Dx = self.make_spec(wave)                                         # (x,y,z) = (64, 64, 1)
        zero_x  = (len(Dx) + self.cnn_x) // self.cnn_x *self.cnn_x        #  X+0padding size
        self.Dx = np.zeros((zero_x, self.cnn_y,  1))                      # (X+0pad, 64, 1)
        self.Dx[self.guard_Dx : self.guard_Dx + len(Dx)] = Dx
        
        # get cut_point by sequence            
        self.cut_point = np.arange(zero_x // self.hop_Dx-1) * self.hop_Dx
        
    def test_label_set(self, event):
        
        # connect each event label
        Dy = self.make_hop_label(event)                                   # (x,y)   = (X,20)
        zero_x  = (len(Dy) + self.cnn_x) // self.cnn_x *self.cnn_x        #  X+0padding size
        self.Dy = np.zeros((zero_x, self.cnn_ch))                         # (X+0pad, 40)
        self.Dy[self.guard_Dx : self.guard_Dx + len(Dy)] = Dy
        
    # make a dataset for test
    def test_set(self, wave, event):
        
        self.test_data_set(wave)
        self.test_label_set(event)
        
        # cutting
        self.cut_and_append()
    
            
            
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
            
            
    
if __name__ == '__main__':
    
    # if prepare sequential dataset 
    set_maker = ML_Dataset_Maker()
    
    
    # if prepare random dataset  
    set_maker = ML_Dataset_Maker(512)    # (512, 64, 64, 1), (512, 64, 20, 1)
            
    