# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:11:45 2020

@author: yumou
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import hbtepLib as hbt

import warnings
warnings.filterwarnings("ignore")

##############################################################################
def applyButterHighpass(raw, time, lowcut=1e3, N=1):
    """
    Copied from fastcamLib._fastcamProcessData
    """
    fs = 1/(time[1]-time[0])
    nyq = 0.5*fs
    low = lowcut / nyq
    b, a = signal.butter(N, low, btype='highpass')
    filt = signal.filtfilt(b,a,raw)
    return filt

def getBreakdownTime(shotno, lowcut=1000, N=1, plot=True, return_index=False):
    """
    Find breakdown time. Return False if time point can't be found.
    Return False if %TREE-E-NODATA, No data available for this node
    """
    try:
        ip_hbt = hbt.get.ipData(shotno, findDisruption=False)
    except:
        return False
    y = ip_hbt.ip[0:1000]
    t = ip_hbt.time[0:1000]
    y_filt = applyButterHighpass(y, t, lowcut=lowcut, N=N)
                                 
    i_start = np.argmax(y_filt)
    t_start = t[i_start]
    
    if plot == True:
        plt.plot(t, y)
        plt.plot(t, y_filt)
        plt.plot(t_start, y[i_start], 'x')
        plt.grid()
        plt.show()
    
    if return_index == True:
        return i_start
    else:
        return round(t_start, 6)

def getDisruptionTime(shotno, lowcut=1000, N=1, i_start=500, threshold=300, plot=True, return_index=False):
    """
    Find disruption time by:
        1.Find peaks in ip;
        2.Find the first peak whose value in y_filt is higher than a threshold value (currently 300);
    Return False if time point can't be found.
    Return False if %TREE-E-NODATA, No data available for this node
    This code works better at identifying spike and discarding wrong ones than John's
    """
    try:
        ip_hbt = hbt.get.ipData(shotno, tStop=0.02, findDisruption=False)
    except:
        return False
    y = ip_hbt.ip[i_start:]
    t = ip_hbt.time[i_start:]
    y_filt = applyButterHighpass(y, t, lowcut=lowcut, N=N)
    
    p = signal.find_peaks(y, prominence=150, distance=50)  #Find peaks in ip
    i_p = p[0]
    
    i_disrupt = -1
    for i in i_p:
        if y_filt[i] > threshold:   #Step 2
            i_disrupt = i
            break
        
    if i_disrupt == -1:
        return False
        
    t_disrupt = t[i_disrupt]
    
    if plot == True:
        plt.plot(t*1e3, y/1e3)
        plt.plot(t*1e3, y_filt/1e3)
        plt.plot(t_disrupt*1e3, y[i_disrupt]/1e3, 'x')
        plt.grid()
        plt.show()
    
    if return_index == True:
        return i_disrupt + i_start
    else:
        return round(t_disrupt,6)