# -*- coding: utf-8 -*-
"""
<vol_03_Symbol_setting>
@author: oze_tech

this cord prepare wave for train and reputation

===============
1. set symbol parameter 
2. make signal
==============="""


import numpy as np
from scipy import signal


class symbol_setting:
    
    def __init__(self, overlap, data_len = 10000, seedi = 0, hop_pattern = None):
        """
        hop_pattern : (default is None) 
           hop_pattern is rooped 
        """
        self.seedi = seedi 
        np.random.seed(seed = self.seedi)
                
        
        # =====================================================================
        # !symbol param
        # =====================================================================
        
        # frequency
        self.fs       = 48000                                        # [Hz] sampling
        self.fc       = 18000                                        # [Hz] carry
        
        # time
        self.s_t      = 0.0415                                       # [s] symbol time
        self.s_len    = int( self.s_t * self.fs )                    # [sample] a symbol length
        
        # ch setting
        self.fg       = 50                                           # [Hz] guard band frecency of symbols    
        self.fhop     = 100                                          # [Hz] guard band frecency of hopping pattern
        self.fch      = 40                                           # [ch] frecency channel
        self.hop      = 20                                           # [ch] hopping  channel of signals (20ch)
        self.fch_rng  = self.fc + np.arange(self.fch) * self.fg      # [Hz] symbol frecency
        
        
        # resampling
        self.re_fs    = 8000                                         # [Hz] after resampling 
        self.re_ratio = self.fs // self.re_fs                        # resampling ratio
        self.sre_len  = int( self.s_t * self.re_fs )                 # [sample] a symbol length after resampling
        
        
        # =====================================================================
        # !signal param
        # =====================================================================

        # signal pattern
        self.si_lap   = overlap                                      # [%] overlap rate
        self.si_len   = data_len                                     # [n] total symbol
        self.si_rng   = np.arange(self.si_len)
        self.si_bin   = np.random.randint(0, 2, self.si_len)         # binary  value of signal
        
        if hop_pattern is None:                                      # hopping pattern of signal
            hop_pattern = np.arange(self.hop)
            np.random.shuffle(hop_pattern)        
            self.si_hop = self.roop_hopping_pattern(hop_pattern)
        else:
            self.si_hop = self.roop_hopping_pattern(hop_pattern)
            
        
        # =====================================================================
        # !high sampling (signal , Event data) 
        # =====================================================================

        self.si_ch    = self.si_hop * 2 + self.si_bin                # Even of 0~40 + Odd
        self.si_strat = self.si_rng * int(self.s_len * self.si_lap)  # start timing of symbol 
        self.si_end   = self.si_strat + self.s_len                   # end   timing of symbol 
        
        # Event data (ch, hopping_ch, time_start, time_end)
        self.event_data  = np.stack([self.si_ch, self.si_hop, self.si_strat, self.si_end], 1)
        
        
        
    #==========================================================================
    # 1. check hopping pattern
    #==========================================================================
    def roop_hopping_pattern(self, hop_pattern):
        
        si_hop = []
        while ( len(si_hop) < self.si_len ):
            si_hop.extend(hop_pattern)
        return np.array(si_hop[:self.si_len])
        
    
    def self_hopping_checker(self, hop_pattern):
        """
        check hopping pattern
        if symbol have no two guard band frecency next to symbols, 
        it needs to change fch
        """

        # candidate      
        np.random.seed(seed = self.seedi)
        next_hop = np.arange(self.hop)
        np.random.shuffle(next_hop)
        
        # overlap
        step = int(1 / self.si_lap) + 1   
        
        # about symbol one by one
        for i in self.si_rng:
            
            # if symbol has not get guard band frecency,
            # change the hopping pattern from candidate.
            j = 0
            while ( np.any(self.si_hop[i-step:i] == self.si_hop[i])):
                self.si_hop[i] = next_hop[j]
                j +=1
                
                
  
class signal_making(symbol_setting):  
    
    
    def __init__(self, overlap, data_len = 10000, seedi = 0, hop_pattern = None):
        
        # class : symbol_setting
        super().__init__(overlap, data_len, seedi, hop_pattern)
        
        # high freqency signal
        self.signal_wave = self.signal_wave_maker()
        
        
    
    #==========================================================================
    # 1. make symbol
    #==========================================================================
                    
    def symbol_maker(self, f_, Square = True):
        """ make a f_[Hz] symbol """
        
        # make a f_ symbol
        x = np.linspace(0,2 * np.pi * self.s_len *( f_ / self.fs), self.s_len)
        Y = np.sin(x)
        
        # make a window
        if Square: # Square wave window
            window = signal.hann(self.s_len//2)
            half   = window.size // 2
            W      = np.hstack((window[:half],np.ones(self.s_len - window.size),window[half:]))    

        else : # normal window  
            W = signal.hann(self.s_len)

        return Y * W 
    
    
    #==========================================================================
    # 2. make signal
    #==========================================================================
    
    def signal_wave_maker(self, Square = True):
        """ make signal wave """
        
        out_wave = np.zeros(self.event_data[-1,-1])
        
        # hop is 0 ~ 20
        for s_ch,_,start,end in self.event_data:
            
            f_ = self.fch_rng[s_ch]
            out_wave[start:end] += self.symbol_maker(f_,Square)
        
        return out_wave / np.abs(out_wave).max()
    
    
    #==========================================================================
    # 3. show signal
    #==========================================================================
    
    def show_signal_paramater(self):
        
        print("total", self.si_len, "symbol")
        print("symbol  ch = ",self.si_ch[:10])
        print("hopping ch = ",self.si_hop[:10])
        print("bin     ch = ",self.si_bin[:10])
        
        
if __name__ == '__main__':
    
    
    maker = signal_making(1,30,0)
    maker.show_signal_paramater()
    