#!/usr/bin/env python
###########################################################
# 
# Script for spectrum analysis of a mono .wav file.
# 
# Code inspired on example code from:
# http://stackoverflow.com/questions/18625085/how-to-plot-a-wav-file
# http://plot.ly/matplotlib/fft/
# http://stackoverflow.com/questions/23507217/python-plotting-2d-data-on-to-3d-axes/23968448#23968448
# http://glowingpython.blogspot.com/2011/08/how-to-plot-frequency-spectrum-with.html
# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.specgram
# 
# Includes a sample clip:
# https://www.freesound.org/people/Kyster/sounds/117719/
# 

import sys
from scipy.io.wavfile import read
import matplotlib as mpl
import matplotlib.pyplot as pyplot
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


###########################################################
# Expects a mono .wav file
# Use a hard coded path or provide one at runtime
if(len(sys.argv) == 1):
    input_path = ""
    input_name = input_path+"117719__kyster__low-d.wav"
else:
    input_name = sys.argv[1]


# Read and convert .wav to a list
input_wav = read(input_name)
input_audio = input_wav[1]
input_rate = input_wav[0]
#input_sample = input_audio[1:22050]


# Parse filename without extension for output and chart titles
input_name = input_name.split('/')[-1]
input_name = input_name.split('.')
input_name = '.'.join(input_name[0:-1])


###########################################################
# Compute FFT on entire input .wav file
fundamental = 313
xmin = 0
xmax = 2000
zmin = 0
zmax = 2000

y = input_audio
Fs = input_rate

n = len(y) # length of the signal
k = np.arange(n)
T = float(n)/Fs
frq = k/T # two sides frequency range
frq = frq[range(n/2)] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(n/2)]

Ynorm = (1000*Y) / max(abs(Y))


###########################################################
# Plot entire FFT with matplotlib
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
pyplot.xticks(np.arange(xmin, xmax, fundamental))
pyplot.savefig(input_name+".png", dpi=300, bbox_inches='tight')
pyplot.close()
#pyplot.show()


###########################################################
# Plot a spectrogram with matplotlib
hot_norm = mpl.colors.Normalize(vmin=-1.,vmax=1.)
pyplot.specgram(input_audio, mode='psd', scale='linear', detrend='none', 
                cmap='gist_heat', NFFT=4096, Fs=44100, noverlap=2048, 
                norm=mpl.colors.Normalize(vmin=0.,vmax=2000.))
pyplot.axis([0, T, xmin, xmax])
pyplot.yticks(np.arange(xmin, xmax, fundamental))
pyplot.savefig(input_name+"_spec.png", dpi=300, bbox_inches='tight')
#pyplot.show()
pyplot.close()


###########################################################
# Plot a 3D diagram of FFTs
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

# Number of slices
y = np.linspace(1, len(input_audio), 5, endpoint=False, dtype=int)

# X axis is frequency range
Fs = input_rate
n = y[1] - y[0] # length of the first sample
k = np.arange(n)
T = float(n)/Fs
x = k/T # two sides frequency range
x = x[range(n/2)] # one side frequency range

xmax_index = next((i for i, x_enum in enumerate(x) if x_enum >= xmax), -1)
#fundamental_index = next((i for i, x_enum in enumerate(x) if x_enum >= fundamental), -1)
x = x[range(xmax_index)]

# Set up 3d plot
X,Y = np.meshgrid(x,y)
Z = np.zeros((len(y),len(x)))

for i in range(len(y-1)):
    # Compute FFT on input .wav file
    # Modified example code from:
    # https://plot.ly/matplotlib/fft/
    # 
    current_sample = input_audio[y[i]:y[i]+n]
    
    current_Z = np.fft.fft(current_sample)/n # fft computing and normalization
    #current_Z = current_Z[range(n/2)]
    current_Z = current_Z[range(xmax_index)]
    
    Z[i] = abs(current_Z)
    
    # Z[i] = (1000*Z) / max(abs(Z))
    #    damp = (i/float(len(y)))**2
    #    Z[i] = 5*damp*(1 - np.sqrt(np.abs(x/50)))
    #    Z[i] += np.random.uniform(0,.1,len(Z[i]))
    
#print(np.argmax(Z[0])) 
#fundamental = int(x[np.argmax(Z[0])])
    
ax.plot_wireframe(X, Y, Z, rstride=1, cstride=30000, color='k', lw=.5)

ax.set_zlim(zmin, zmax)
ax.set_xlim(xmin, xmax)
pyplot.xticks(np.arange(xmin, xmax, fundamental))
#ax.set_zlabel("Intensity")
ax.view_init(41,-59)
#pyplot.show()
pyplot.savefig(input_name+"_3D.png", dpi=300, bbox_inches='tight')
pyplot.close()
