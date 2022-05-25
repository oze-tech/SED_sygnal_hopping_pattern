# -*- coding: utf-8 -*-
"""
<Test>
@author: oze_tech

this cord is test of classifier SNR by simulation 

class Classifiers
===============
1. Prepare Dataset
2. Define model
3. Test 
===============

"""


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from No1_train_data import Simulation_Wave_Maker as Smaker
from No2_model      import vol_01_CNN            as cnn
from No2_model      import vol_02_RNN            as rnn
from No2_model      import vol_03_CRNN           as crnn
from No2_model      import ML_Data_Maker         as Dset
from No4_test.func  import vol_01_reputate_Fscore_BE as func1



# =============================================================================
# 01.prepare dataset
# =============================================================================
class Test_Dataset_Maker(Dset.ML_Data_Maker):
    
    def __init__(self, input_shape  = (64, 64, 1), output_shape = (64, 20)):  
        
        # class : ML_Data_Maker
        super().__init__(input_shape, output_shape)
        
        # save Dataset
        self.Dx_set = []                # spectrogram
        self.Dy_set = []                # event label
        self.Dh_set = []                # hopping event label
        
        

    # =========================================================================
    # 1.make Dataset for test 
    # =========================================================================
    
    def test_set(self, wave, event):
                
        # prepare data and label
        Dx = self.make_spec(wave)               # (x,y,z) = (64, 64, 1)
        Dy = self.make_hop_label(event)         # (x,y)   = (64, 20)
    
        # prepare guard interval by zero padding on x axis
        zero_x  = (len(Dx) + self.cnn_x) // self.cnn_x *self.cnn_x        #  X+0padding size
        self.Dx = np.zeros((zero_x, self.cnn_y,  1))                      # (X+0pad, 64, 1)
        self.Dy = np.zeros((zero_x, self.cnn_ch))                         # (X+0pad, 40)
        
        # guard interval is at start and end point
        self.guard_Dx = self.cnn_x //4
        self.hop_Dx   = self.cnn_x //2          # overlap size of real prosess
        
        self.Dx[self.guard_Dx : self.guard_Dx + len(Dx)] = Dx
        self.Dy[self.guard_Dx : self.guard_Dx + len(Dy)] = Dy
        
        # get cut_point by sequence            
        self.cut_point = np.arange(zero_x // self.hop_Dx-1) * self.hop_Dx            
        
        # cutting
        for i_s in self.cut_point:
                        
            i_e = i_s + self.cnn_x 
            
            Y = self.Dy[i_s : i_e ]
            X = self.Dx[i_s : i_e ]
            X = self.data_norm(X)
            
            self.Dy_set.append(Y)
            self.Dx_set.append(X)
                        
    # =========================================================================
    # 2. list to ndarray
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
# 02.Test
# =============================================================================
class sim_testing(Test_Dataset_Maker):
 
    
    def __init__(self, overlap, symbol_N=10000, seedi=0):
        
        # class : Train_Dataset_Maker
        super().__init__()
        
        self.overlap     = overlap
        self.SNR         = np.arange(0,16,5) 
        self.symbol_N    = symbol_N
        
        self.weight_name = "simulation_overlap_{0:03}_per_".format(100-int(100 * self.overlap))
        self.save_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/result/"
            
        self.sim = Smaker.simulation_signal(self.overlap, self.symbol_N, seedi)
        
        # saving
        self.Mlabel  = ["SNR"]
        self.F_s     = [self.SNR] # f score
        self.E_R     = [self.SNR]
        
        
    def __call__(self, model_class, model_name = ""):
        
        
        # Prepare wave
        F_s = []
        E_R = []
        
        self.Mlabel.append(model_name)
            
        for snr in self.SNR:
            
            self.make_snr(snr)
            self.test_z = model_class(self.test_x)
            
            
            F_s.append(func1.cal_Error_rate(self.test_y, self.test_z))
            E_R.append(func1.cal_F_score(   self.test_y, self.test_z))
                
        self.F_s.append(F_s)
        self.E_R.append(E_R)    
        
        
    def make_snr(self, snr):
        wave, event = self.sim(snr)
        self.test_set(wave, event)
        self.test_x, self.test_y = self.get_dataset(False)
        
    
        
        
        
    def show_result(self):
        
        # ==============
        # make table
        # ==============
        FS_df = pd.DataFrame(np.array(self.F_s), index = self.Mlabel)
        ER_df = pd.DataFrame(np.array(self.E_R), index = self.Mlabel)
        
        
        # ==============
        # show graph
        # ==============
        def show_graph(DataFrame, y_label = ""):
            
            plt.figure()
            for i, label in enumerate(self.Mlabel[1:]):
                plt.plot(DataFrame.loc["SNR"], DataFrame.iloc[1+i], label = label )
                
            plt.yscale('log')
            plt.ylim(0.001,100)
            plt.ylabel(y_label)
            plt.xlabel("SNR[dB]")
            
            plt.legend()
            plt.tight_layout()

        show_graph(FS_df,"F-score")
        plt.savefig(self.save_folder + self.weight_name+"Fscore.png")
        
        show_graph(ER_df,"ER")
        plt.savefig(self.save_folder + self.weight_name+"ER.png")
        
        
        # ==============
        # saving            
        # ==============
        
        FS_df.to_csv(self.save_folder + self.weight_name+"Fscore.csv")
        ER_df.to_csv(self.save_folder + self.weight_name+"ER.csv")
        
          
    
def test():
    
    for lap in [1, 1/2, 1/4]:
        
        test  = sim_testing(lap, 10000)
        
        CNN  = cnn.CNN(  test.weight_name,  param_init = True)    
        RNN  = rnn.RNN(  test.weight_name,  param_init = True)    
        CRNN = crnn.CRNN(test.weight_name,  param_init = True)    
        
        test(CNN,"CNN")
        test(RNN,"RNN")
        test(CRNN,"CRNN")
            
        test.show_result()
        
        
        
if __name__ == '__main__':
    
    
    test()
    
    