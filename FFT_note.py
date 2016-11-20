#!/usr/bin/env python
import sys

# http://stackoverflow.com/questions/18625085/how-to-plot-a-wav-file
from scipy.io.wavfile import read
import matplotlib.pyplot as pyplot

# https://plot.ly/matplotlib/fft/
#import matplotlib.pyplot as plt
#import plotly.plotly as py
import numpy as np


# Expects a .wav file with 0.5 seconds of a sound sampled at 44100Hz
# Use a hard coded path or provide one at runtime
if(len(sys.argv) == 1):
    input_path = ""
    input_name = input_path+"PW8-A220.wav"
else:
    input_name = sys.argv[1]


# Read and convert .wav to a list
input_wav = read(input_name)
input_audio = input_wav[1]
input_rate = input_wav[0]
input_sample = input_audio[1:22050]


# Parse filename without extension for output and chart titles
input_name = input_name.split('/')[-1]
input_name = input_name.split('.')
input_name = '.'.join(input_name[0:-1])


# Example code from:
# http://glowingpython.blogspot.com/2011/08/how-to-plot-frequency-spectrum-with.html
# 
# pyplot.plot(input_sample)
# pyplot.ylabel("Amplitude")
# pyplot.xlabel("Time")
# pyplot.title("Triangle A220")
# 
# pyplot.show()
# 
# Fs = 150.0;  # sampling rate
# Ts = 1.0/Fs; # sampling interval
# t = np.arange(0,1,Ts) # time vector
# 
# ff = 5.5;   # frequency of the signal
# y = np.sin(2*np.pi*ff*t)
# print(y)


# Compute FFT on input .wav file
# Modified example code from:
# https://plot.ly/matplotlib/fft/
# 
y = input_sample
Fs = input_rate

n = len(y) # length of the signal
k = np.arange(n)
T = float(n)/Fs
frq = k/T # two sides frequency range
frq = frq[range(n/2)] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(n/2)]

Ynorm = (1000*Y) / max(abs(Y))

#fig, ax = plt.subplots(2, 1)
#ax[0].plot(t,y)
#ax[0].set_xlabel('Time')
#ax[0].set_ylabel('Amplitude')


# Plot with matplotlib
fundamental = 220
xmin = 0
xmax = 10000

pyplot.figure(figsize=(4,2), dpi=300, facecolor='w', edgecolor='w')

# Plot light vertical lines on even harmonics
for harmonic in range(0, xmax, fundamental*2): 
    pyplot.axvline(harmonic, color='0.9')

# Plot dark vertical lines on odd harmonics
for harmonic in range(fundamental, xmax, fundamental*2): 
    pyplot.axvline(harmonic, color='0.8')

pyplot.plot(frq,abs(Ynorm),'k') # plotting the spectrum
pyplot.title(input_name)
#pyplot.xlabel('Freq (Hz)')
#pyplot.ylabel('|Y (freq)|')
pyplot.axis([xmin, xmax, 0, 1000])
pyplot.savefig(input_name+".png", dpi=300, bbox_inches='tight')
#pyplot.show()