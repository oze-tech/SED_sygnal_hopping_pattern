# -*- coding: utf-8 -*-
"""
@author: oze_tech
vol_02_Filter_Maker

making a band pass filter by 40ch

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class filter_maker():
    
    # =============================
    # initial defines
    # =============================
    def __init__(self, fs=8000, fc=1000, fch=40, fg=50):
            
        # param
        self.fs  = fs  #[sampling rate]
        self.fc  = fc  #[Hz] frequency of carryer 
        self.fg  = fg  #[Hz] frequency of symbol's guard band
        self.fch = fch #[ch] channel count (default : 40ch)
        
        
        self.fhop = 100  #[Hz] guard band frecency of hopping pattern
        self.hop  = 20   #[ch] hopping  channel of signals (20ch)
       
        
        # ==================
        # making Nch filter
        # ==================
        
        # frequency of band pass filters
        self.fil_ch   = []
        self.fil_idx  = np.arange(self.fch)
        self.fil_freq = self.fg * self.fil_idx + self.fc
        
        
        for fil_f in self.fil_freq:
            
            # define of band pass freqency
            self.cutoff = (np.array([-self.fg//2, self.fg//2]) + fil_f)/ self.fs * 2
            
            # making BPF
            self.fil_ch.append(signal.firwin(401, self.cutoff, pass_zero = False))
        
                        
    def __call__(self, in_wave):
        """
        inwave -> Through BPF
        """
        self.pass_fil(in_wave)
        return self.bpf_wave
        
    

    #==========================
    # 1 . filter processing
    #==========================      
        
    def conv_fil(self, in_wave):
        """
        convolute wave : target wave
        """
        
        begin = self.conv_filter.size//2
        
        end = begin + in_wave.size
        
        wave = np.convolve(in_wave, self.conv_filter)[begin:end] #wave × filter
        
        
        return wave
    
    
    
    #==========================
    # 2 . band pass by each ch
    #==========================      
        
        
    def pass_fil(self, in_wave):
        """
        in_wave is passed through (N)ch BPF, 
            default： N = 40ch (01ch:1000Hz(975~1025Hz), 40ch:3950Hz(3925~3975Hz))       
        bpf_wave : (40,n)
        """
        
        # filter processing by (N)ch
        passed_wave = []
        
        for self.conv_filter in self.fil_ch:   
            
            passed_wave.append(self.conv_fil(in_wave))
            
        self.bpf_wave = np.array(passed_wave)
    
    
   
        
        
        
    #=============
    # showing
    #=============

    def show_filter(self, show_fil_idx = np.arange(40)):
        
        """
        decide self.conv_filter before doing 
        """
        show_fil = np.array(self.fil_ch)[show_fil_idx]
        show_fil = np.sum(show_fil, axis = 0)
        
        w, h = signal.freqz(show_fil)
        a = 20 * np.log10(abs(h));
        f = w / (2 * np.pi) * self.fs
        
        plt.figure()  # (width, height)        
        plt.plot(f,a)
        plt.title('Digital filter frequency response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.xlim(0, 4000)
        plt.ylim([-15,0])
        plt.grid(which='both', axis='both')
        plt.show()
        
        
    
    def show_ch_wave(self, in_wave):
        """
        in_wave : wave through BPF
        """
        max_y = np.abs(in_wave).max()
        
        plt.figure()
        for i, wave in enumerate(in_wave):
            plt.subplot(len(in_wave),1,i+1)
            plt.plot(wave)
            plt.title("ch :"+str(i))
            plt.ylim([-max_y,max_y])
        plt.tight_layout()
        plt.show()
            
           
if __name__ == '__main__':
    
     FM = filter_maker()
     
     FM.show_filter()
     FM.show_filter([1])