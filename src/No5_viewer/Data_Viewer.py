# -*- coding: utf-8 -*-
"""
<vol_03_CNN_Data_Viewer>
@author: oze_tech

 - This program is showing dataset
 
 1. input data  & predict data
 2. feature map & filter
 
"""

import os
import sys  


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from No1_train_data  import Simulation_Wave_Maker  as Smaker
from No2_model       import vol_01_CNN             as cnn
from No2_model       import vol_02_RNN             as rnn
from No2_model       import vol_03_CRNN            as crnn
from No4_test        import Test
from No5_viewer.func import vol_01_show_dataset    as n5v1
from No5_viewer.func import vol_02_show_FeatureMap as n5v2

if __name__ == '__main__':
    
    overlap = 1
    SNR     = 15
    seedi   = 0
    batch   = 128
    show_i  = 2
    
    # make a wave
    sim = Smaker.simulation_signal(overlap)
    wave, event = sim(SNR)
    

    # make a dataset
    tester = Test.sim_testing(overlap, 100, seedi)
    tester.make_snr(SNR)
    
    x_test = tester.test_x
    y_test = tester.test_y
    
    
    CNN   = cnn.CNN(  tester.weight_name)
    RNN   = rnn.RNN(  tester.weight_name)
    CRNN  = crnn.CRNN(tester.weight_name)
    
    CNN.show_model()
    RNN.show_model()
    CRNN.show_model()
    
        
    
    # show only predict data
    if False:
        #n5v1.show_spec_data(  x_test[show_i,:,:,0].T, "")
        #n5v1.show_event_label(y_test[show_i].T, "")
    
        n5v1.show_spec_data(  x_test[show_i,:,:,0].T, "input spectrogram : SNR "+str(SNR))
        n5v1.show_event_label(y_test[show_i].T, "ideal hopping event label")
        
        n5v1.show_event_label(n5v1.pred_label(CNN.model , x_test)[show_i].T,"event label predicted by CNN")
        n5v1.show_event_label(n5v1.pred_label(RNN.model , x_test)[show_i].T,"event label predicted by RNN")
        n5v1.show_event_label(n5v1.pred_label(CRNN.model, x_test)[show_i].T,"event label predicted by CRNN")
        
    
    # show input data and predict data 
    if False:
        z_test = n5v1.pred_model(CNN.model, x_test) 
        n5v1.show_predict_data(x_test, y_test, z_test, show_i)
        
        z_test = n5v1.pred_model(RNN.model, x_test) 
        n5v1.show_predict_data(x_test, y_test, z_test, show_i)
        
        z_test = n5v1.pred_model(CRNN.model, x_test) 
        n5v1.show_predict_data(x_test, y_test, z_test, show_i)
    
    # show feature map
    if False:
        n5v2.show_feature_map(CNN.model , x_test, show_idx = show_i)
        n5v2.show_feature_map(CRNN.model, x_test, show_idx = show_i)
                
    # show filter
    if False:
        n5v2.show_filter(CNN.model)
        n5v2.show_filter(CRNN.model)
    