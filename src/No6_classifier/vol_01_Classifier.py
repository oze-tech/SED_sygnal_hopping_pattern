# -*- coding: utf-8 -*-
"""
@author: oze_tech
vol_02_ML_Classifier

This program defines ML Classifier for predicting BER
    
event data  : time stamp by ch, hopping_pattern, start, end information
event label : strong label (x:time, y:ch)

"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from No2_model import Dataset_Maker as Dmaker


class Classifier():
    
    
    def __init__(self, model_class, input_shape  = (64, 64,  1), output_shape = (64, 20)):
        
        # data parameter
        maker = Dmaker.ML_Dataset_Maker()
        
        # model
        self.Model = model_class
        
        
    
    def __call__(self, wave):
        """
        input  : signal wave
        output : predict event
        """
        
        
        
        
    def test(self, wave, event):
        
        # 1. data processing
        self.dataset_maker(wave, event)
        Dx, Dy = self.get_dataset(show = False)
        
        # 2. predict event label
        Dz = self.model_pred(Dx)  #(batch, x, y,)
        
        