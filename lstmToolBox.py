# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 09:35:14 2020

@author: yumou

Functions for preprocessing data for LSTM
"""
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def getLookback(data_name, t_lookback):
    """
    Calculate n_lookback
    """
    DATA_DIR = os.path.join(os.getcwd(), 'Data')
    DATA_FILE = os.path.join(DATA_DIR, data_name)
    data_df = pd.read_csv(DATA_FILE, nrows=2)
    fs = 1/round(data_df['t'].iloc[1] - data_df['t'].iloc[0],6)
    n_lookback = int(round(t_lookback * fs,0))
    
    return n_lookback


def temporalizeShot(data_df_slice, lookback, divide_set=True):
    """
    Takes in data_df_slice dataframe; returns 3 data_np_slice_3d 3d array
    Array has the shape (samples x timesteps x 4+features(10))
    """
    if divide_set == True:
        data_np_slice_3d_breakdown = np.empty((0, lookback, data_df_slice.shape[1]))
        data_np_slice_3d_fit = np.empty((0, lookback, data_df_slice.shape[1]))
        data_np_slice_3d_disrupt = np.empty((0, lookback, data_df_slice.shape[1]))
        
        for i in np.arange(lookback, data_df_slice.shape[0]):
            #Check if row i belong to breakdown/fit/disrupt group
            if data_df_slice.iloc[i]['label'] == 1:
                data_np_slice_3d_fit = np.append(data_np_slice_3d_fit, [data_df_slice.iloc[i:i-lookback:-1,:].to_numpy()], axis=0)
            elif data_df_slice.iloc[i]['label'] == 0:
                data_np_slice_3d_breakdown = np.append(data_np_slice_3d_breakdown, [data_df_slice.iloc[i:i-lookback:-1,:].to_numpy()], axis=0)
            elif data_df_slice.iloc[i]['label'] == 2:
                data_np_slice_3d_disrupt = np.append(data_np_slice_3d_disrupt, [data_df_slice.iloc[i:i-lookback:-1,:].to_numpy()], axis=0)
        return data_np_slice_3d_breakdown, data_np_slice_3d_fit, data_np_slice_3d_disrupt
    else:
        data_np_slice_3d = np.empty((0, lookback, data_df_slice.shape[1]))
        
        for i in np.arange(lookback, data_df_slice.shape[0]):
            data_np_slice_3d = np.append(data_np_slice_3d, [data_df_slice.iloc[i:i-lookback:-1,:].to_numpy()], axis=0)
        return data_np_slice_3d

def getScaler(data_np_3d):
    """
    Calculate normalization scaler using data_np_3d 
    data_np_3d has the shape (samples x timesteps x features(10))
    """
    shape = data_np_3d.shape
    data_np_2d = data_np_3d.reshape(shape[0]*shape[1], shape[2])
    scaler = StandardScaler().fit(data_np_2d)
    return scaler


def rescale(data_np_3d, scaler):
    """
    Flatten data_np_3d, rescale according to scaler, then reshape back to 3d
    3d array has the shape (samples x timesteps x features(10))
    Flattened 2d array has the shape (sample*timesteps x features(10))
    """
    shape = data_np_3d.shape
    data_np_2d = data_np_3d.reshape(shape[0]*shape[1], shape[2])
    data_np_2d_rescaled = scaler.transform(data_np_2d)
    data_np_3d_rescaled = data_np_2d_rescaled.reshape(shape[0], shape[1], shape[2])
    return data_np_3d_rescaled


def unscale(data_np_3d_rescaled, scaler):
    """
    Inverse transform of rescale() input 3d array.
    3d array should have the shape (samples x timesteps x features(10)).
    Output of scaler.inverse_transform() will have some rounding issue, so 
    np.array_equal(original, unscaled) will return False if not specifying .round(n).
    """
    shape = data_np_3d_rescaled.shape
    data_np_2d_rescaled = data_np_3d_rescaled.reshape(shape[0]*shape[1], shape[2])
    data_np_2d_unscaled = scaler.inverse_transform(data_np_2d_rescaled)
    data_np_3d_unscaled = data_np_2d_unscaled.reshape(shape[0], shape[1], shape[2])
    return data_np_3d_unscaled


'''
def flatten(data_np_3d):    
    """
    Flatten a 3d np array to a 2d np array for scaling
    3d array has the shape (samples x timesteps x features(10))
    Flattened 2d array has the shape (sample*timesteps x features(10))
    THIS FUNCTION WORKS SLOWER THAN ARRAY.RESHAPE()
    """
    shape = data_np_3d.shape
    data_np_2d = np.empty((shape[0]*shape[1], shape[2]))
    for i in range(shape[0]):
        print(i)
        data_np_3d_islice = data_np_3d[i,:,:]
        data_np_2d = np.append(data_np_2d, data_np_3d_islice, axis=0)
    return data_np_2d
'''