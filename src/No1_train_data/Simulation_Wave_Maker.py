# -*- coding: utf-8 -*-
"""
<Simulation_Wave_Maker>
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
import matplotlib.pyplot as plt


# Get the absolute path of the file
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signal_class import vol_01_Downsampling_Processer as vol1
from signal_class import vol_02_Filter_Maker           as vol2
from signal_class import vol_03_Symbol_setting         as vol3


class simulation_signal(vol3.signal_making):
    """
    make wave class
    
    input : 
        overlap  :  symbol overlap
        data_len :  signal length
        seedi    :  random seed
    output : 
        Wave adjusted for SNR
    """
    
    
    def __init__(self, overlap, data_len = 1000, seedi = 0):
        
        
        # class : symbol_making
        super().__init__(overlap, data_len, seedi )
        
        # =====================================================================
        # !low sampling (signal , Event data) 
        # =====================================================================
        
        # Class
        self.Downer  = vol1.downsampling_processer()        # Class : Downer 
        self.filter  = vol2.filter_maker()                  # Class : Filter
        
        # low symbol
        self.low_signal_wave = self.Downer(self.signal_wave)        # low signal
        self.wave_size       = self.low_signal_wave.size            # signal size
        self.low_signal_wave = self.normalize_power(self.low_signal_wave)

        # Event data (ch, hopping_ch, time_start, time_end)
        self.low_event_data = self.event_data.copy()                # resampling Event data
        self.low_event_data[:,2] //= self.re_ratio                  # resampling start
        self.low_event_data[:,3] //= self.re_ratio                  # resampling end
        
        # wihte noise 
        np.random.seed(seed = seedi)
        self.white_noise = np.random.random(self.wave_size) - 0.5
        self.white_noise = self.normalize_power(self.white_noise)

        # normalize
        self.signal_1ch = self.filter(self.low_signal_wave)[self.low_event_data[0,0], :self.sre_len]
        self.noise_2ch  = self.filter(self.white_noise)[:2, :self.sre_len]
        
        self.power_rate = self.get_rms(self.noise_2ch) / self.get_rms(self.signal_1ch)
        
        # normalize power (signal power = noise power)
        self.low_signal_wave *= self.power_rate
        
        
        
    def __call__(self, SNR):
        """
        input : SNR(INT)
        output: wave, event
        """
        
        def db2amplitude(db_):
            """noise's amp is adjusted (dB -> amp)"""
            return 10.0**(db_/20.0)
        
        # =====================================================================
        # !mix signal and noise
        # =====================================================================
        self.signal_SNR   = self.low_signal_wave * db2amplitude(SNR)
        self.signal_noise = self.white_noise     + self.signal_SNR
        
        
        self.signal_noise = self.normalize_power(self.signal_noise)
        
        return self.signal_noise , self.low_event_data
        
        
    
    #==========================================================================
    # !normalize
    #==========================================================================
    def get_rms(self, in_array):
            power = in_array * np.conj(in_array)
            return np.sqrt(np.real(np.average(power)))
    
    
    def normalize_power(self,in_array):
        """
        normalize power to 1
        """
        
        return in_array / self.get_rms(in_array)
    
    
    
    
    
if __name__ == '__main__':
    
    
    sim = simulation_signal(1,5)
    
    wave , event_data = sim(10)
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(sim.white_noise)
    
    plt.subplot(3,1,2)
    plt.plot(sim.signal_SNR)
        
    plt.subplot(3,1,3)
    plt.plot(sim.signal_noise)
        