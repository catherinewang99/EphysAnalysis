# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:41:04 2024

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from ephysSession import Session
import behavior
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 

path = r'H:\ephys_data\CW47\python\2024_10_25'
path = r'J:\ephys_data\CW48\python\2024_10_30'
path = r'J:\ephys_data\CW48\python\2024_11_05'
path = r'J:\ephys_data\CW49\python\2024_12_12'
path = r'J:\ephys_data\CW49\python\2024_12_13'
path = r'J:\ephys_data\CW53\python\2025_01_29'
path = r'J:\ephys_data\CW54\python\2025_02_03'
s1 = Session(path, passive=False)#, side='R')

#%% Plot sig neurons
s1.plot_number_of_sig_neurons(window=100)

#%%
s1.plot_raster_and_PSTH(215, opto=True, stimside='R', binsize=75, timestep=5)

for i in range(12,250,35):
    s1.plot_raster_and_PSTH(i, opto=False, stimside='R', binsize=75, timestep=5)

#%% 
#Delay seelective
for n in s1.get_epoch_selective((s1.delay, s1.response), p=0.01):
    if s1.unit_side[n] == 'R':
        
        s1.plot_raster_and_PSTH(n, opto=True, stimside = 'L', binsize=150, timestep=5)

        
    # s1.plot_raster_and_PSTH(n, binsize=75, timestep=5)
    # stimside = 'R' if s1.unit_side[n] == 'L' else 'L'
    # s1.plot_raster_and_PSTH(n, opto=True, stimside = stimside, binsize=75, timestep=5)
    
#%% 
#Sample seelective
for n in s1.get_epoch_selective((s1.sample, s1.delay)):

    s1.plot_raster_and_PSTH(n, binsize=75, timestep=5)
    
#%%
s1 = Session(path, passive=False, side='R')
s1.plot_number_of_sig_neurons(window=100)

s1 = Session(path, passive=False, side='L')
s1.plot_number_of_sig_neurons(window=100)