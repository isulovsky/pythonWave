# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:17:19 2024

@author: Ivan Sulovsky
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:52:00 2024

@author: user
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib import rc

#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
from scipy import signal
from scipy.integrate import simpson

#pierson-moskowitz Spectrum coefficients-----------------#

Hsig = 2.35 #2.35

U = 10.414 
f_period = 1.08
Tpeak = 7.59 *f_period 
g = 9.8067
gamma = 3.3
twoPi = 2*np.pi


Fs = 5

dt = 1/Fs

StopTime    = 3600

t = np.arange(0, StopTime-dt, dt)

low_w = 0.003

high_w = 3

w = np.linspace(low_w,high_w,2000)

dw = w[1]-w[0]

w = w = w + dw * np.random.rand(len(w))

phi = 2 * np.pi * (np.random.rand(len(w)))

alpha = 4*np.pi**3*(Hsig/(g*Tpeak**2))**2

beta = 16*np.pi**3*(U/(g*Tpeak))**4

S = (alpha*g**2)/w**5*np.exp(-beta*(g/(U*w))**4)

#synthetic wave trace-----------------------------------#

Zeta = np.sqrt(2*S*dw)

wave_ittc=np.empty(np.size(t))

for i in range(np.size(t)):
      wave_ittc[i]=np.sum(Zeta*np.cos(w*t[i]+phi))
      
#Fourier reconstruction using welch method--------#      

coeff = 35 #tuning coefficient for the welch method. 

sfreq = np.size(wave_ittc)/StopTime #optimal sampling frequency

win = int((np.size(wave_ittc)/coeff)) #length of each segment 

f1,px1  = signal.welch(wave_ittc,sfreq,nperseg=win,scaling='density',return_onesided=True)

fig, (ax1,ax2) = plt.subplots(2, dpi=150,figsize=(10,18))
fig.tight_layout(pad=3.5)
ax1.grid(linewidth=0.55)
ax1.plot(t,wave_ittc, color='black', linewidth=0.5,label='wave trace in time domain')
ax1.set_xlabel("$time, s$", fontsize=14)
ax1.set_ylabel("$\eta, m$", fontsize=14)
ax1.legend(fontsize=12)
ax1.set_xlim(0,StopTime)
ax2.grid()
ax2.plot(f1*twoPi,px1/twoPi,label='Fourier reconstruction ', color='black', linewidth=1.0)
ax2.plot(w,S,label='JONSWAP Theoretical ',color='black', linewidth=0.6,ls="--")
ax2.set_xlabel("$\omega, rad/s$", fontsize=14)
ax2.set_ylabel("$S_(\omega), m^2s$", fontsize=14)
ax2.legend(fontsize=16)
ax2.set_xlim(0,3)

'''
Calcuation of spectral moments if needed:
from scipy.integrate import simpson
m0_t = simpson(S,w)
m1_t = simpson(S*w,w)
m2_t = simpson(S*w*w,w)
'''
