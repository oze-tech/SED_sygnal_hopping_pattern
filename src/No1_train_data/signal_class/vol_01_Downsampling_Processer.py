# -*- coding: utf-8 -*-
"""
@author: zekio2779
vol_01_Downsampling_Processer

Downsampling the high frequency signal
     <Processing flow>
     1. BPF processing
     2. Mixer treatment
     3. LPF processing
     4. Resampling
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy import signal


class downsampling_processer():

    def __init__(self):  
        
        
        self.fs         = 48000  #[Hz] sampling
        self.fc         = 17000  #[Hz] carrier
        
        # 1st bpf setting
        self.bpf_band   = 4000   #[Hz] band of BPF 
        self.cutoff     = np.array( [self.fc, self.fc + self.bpf_band] ) / self.fs * 2
        self.bpf_filter = signal.firwin(401, self.cutoff, pass_zero = False)
        
        # 3rd lpf setting
        self.low_band   = 5000   #[Hz] LPF
        self.lpf_filter = signal.firwin(401, self.low_band / self.fs * 2)
        
        # resampling setting
        self.re_fs      = 8000   #[Hz] after resampling 
        self.step       = self.fs // self.re_fs # Ratio before resampling processing
        
        
    def __call__(self,in_wave):
        
        self.bpf_1st(in_wave)
        self.mixer_2nd(self.wave)
        self.lpf_3rd(self.wave)
        self.resample_4th(self.wave)
        return self.wave
    
        
    #==========================
    # filter processing
    #==========================
        
    def conv_fil(self,in_wave):
        """
        convolute wave : target wave
        """
        begin = self.conv_filter.size//2
        end = begin + in_wave.size
        self.wave = np.convolve(in_wave, self.conv_filter)[begin:end] #wave × filter 
        
    

    def bpf_1st(self, in_wave):
        """
        1st BPF( pass 17kHz-21kHz)
        """ 
        self.conv_filter = self.bpf_filter
        self.conv_fil(in_wave)
        
        
    def mixer_2nd(self, in_wave):
        """
        2nd mixer (conv wave × fc)
        """
        carrier = np.sin(np.arange(in_wave.size) * 2 * np.pi * self.fc / self.fs )
        self.wave = carrier * in_wave

    def lpf_3rd(self, in_wave):
        """
        3rd LPF( pass under 5kHz)
        """ 
        self.conv_filter = self.lpf_filter
        self.conv_fil(in_wave)
    
    def resample_4th(self, in_wave):
        """
        4th resampling
        """
        resample_size = in_wave.size // self.step
        
        resample_wave = []
        
        for i in range(resample_size):
            resample_wave.append(in_wave[self.step * i])
        
        self.wave = np.array(resample_wave)
           

    
    
    
    #==========================
    # show filter result
    #==========================
    
    def show_filter(self, low = 0 , high = 24000):
        """
        decide self.conv_filter before doing 
        """
        w, h = signal.freqz(self.conv_filter)
        a = 20 * np.log10(abs(h));
        f = w / (2 * np.pi) * self.fs
        
        plt.figure()  # (width, height)        
        plt.plot(f,a)
        plt.title('Digital filter frequency response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.xlim(low, high)
        plt.ylim([-15,0])
        plt.grid(which='both', axis='both')
        plt.show()
    
    def show_wave(self,fs_, n_fft, title = ""):

        plt.figure()
        plt.title(title)
        plt.plot(self.wave)
        plt.show()
    
        D = np.abs(librosa.stft(self.wave, n_fft))
        plt.figure()
        plt.title(title)    
        librosa.display.specshow(D, hop_length = n_fft//4,
                                 y_axis='linear',x_axis='time',sr = fs_)
        plt.show()            
    
        
    def show_all(self, in_wave):
        self.wave = in_wave
        
        self.show_wave(self.fs, 256,"before process")
        self.bpf_1st(in_wave)
        self.show_filter(16000, 22000)
        self.show_wave(self.fs, 256,"after BPF")
        
        self.mixer_2nd(self.wave)
        self.show_wave(self.fs, 256,"after Mixer")
        
        self.lpf_3rd(self.wave)
        self.show_wave(self.fs, 256,"after LPF")
        self.show_filter(0,8000)
        
        self.resample_4th(self.wave)        
        self.show_wave(self.re_fs, 64,"after Resampling")
        
        
if __name__ == '__main__':
    dper = downsampling_processer()
    dper.conv_filter = dper.bpf_filter
    dper.show_filter(16000, 22000)
    dper.conv_filter = dper.lpf_filter
    dper.show_filter(0,8000)
    
    