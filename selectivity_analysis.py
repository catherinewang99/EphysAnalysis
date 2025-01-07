# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 09:38:49 2025

Recreate elements from Chen et al 2021 Fig 1

@author: catherinewang

"""


import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from session import Session
import behavior
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 

#%% Aggregate over all FOVs for this analysis

paths = [
            r'J:\ephys_data\CW49\python\2024_12_11',
            r'J:\ephys_data\CW49\python\2024_12_12',
            r'J:\ephys_data\CW49\python\2024_12_13',
            r'J:\ephys_data\CW49\python\2024_12_14',
            r'J:\ephys_data\CW49\python\2024_12_15',
            r'J:\ephys_data\CW49\python\2024_12_16',
        
        ]


# Get selectivity (spikes/s) Chen et al 2021 Fig 1C
leftpref, leftnonpref = [], []
rightpref, rightnonpref = [], []

left_sample_sel, right_sample_sel = [],[]
left_choice_sel, right_choice_sel = [],[]



for path in paths:
    s1 = Session(path, passive=False, side='L')
    delay_neurons = s1.get_epoch_selective(epoch=(s1.response-1.5, s1.response), p=0.05)
    
    pref,nonpref,_ = s1.plot_selectivity(delay_neurons, plot=False, binsize=200, timestep=50)
    
    leftpref += [pref]
    leftnonpref += [nonpref]

    windows = np.arange(-0.2, s1.time_cutoff, 0.2)
    
    sample_sel, delay_sel = [],[]
    for t in range(len(windows)-1):
        sample_sel += [s1.get_number_selective((windows[t],windows[t+1]), mode='stimulus')]
        delay_sel += [s1.get_number_selective((windows[t],windows[t+1]), mode='choice')]
        
    left_sample_sel += [np.array(sample_sel) / len(s1.good_neurons)]
    left_choice_sel += [np.array(delay_sel) / len(s1.good_neurons)]
    
    s1 = Session(path, passive=False, side='R')
    delay_neurons = s1.get_epoch_selective(epoch=(s1.response-1.5, s1.response), p=0.05)
    
    pref,nonpref,time = s1.plot_selectivity(delay_neurons, plot=False, binsize=200, timestep=50)
    
    rightpref += [pref]
    rightnonpref += [nonpref]
    
    sample_sel, delay_sel = [],[]
    for t in range(len(windows)-1):
        sample_sel += [s1.get_number_selective((windows[t],windows[t+1]), mode='stimulus')]
        delay_sel += [s1.get_number_selective((windows[t],windows[t+1]), mode='choice')]
        
    right_sample_sel += [np.array(sample_sel) / len(s1.good_neurons)]
    right_choice_sel += [np.array(delay_sel) / len(s1.good_neurons)]

leftpref, leftnonpref = cat(leftpref), cat(leftnonpref)
rightpref, rightnonpref = cat(rightpref), cat(rightnonpref)
leftsel = np.mean(np.array(leftpref)-np.array(leftnonpref), axis=0)
leftsel_error = np.std(np.array(leftpref)-np.array(leftnonpref), axis=0) / np.sqrt(leftsel.shape[0])

rightsel = np.mean(np.array(rightpref)-np.array(rightnonpref), axis=0)
rightsel_error = np.std(np.array(rightpref)-np.array(rightnonpref), axis=0) / np.sqrt(rightsel.shape[0])

# Proportion selective: 200 ms time bins, p < 0.05
right_sample_sel = np.mean(right_sample_sel, axis=0)
left_sample_sel = np.mean(left_sample_sel, axis=0)
right_choice_sel = np.mean(right_choice_sel, axis=0)
left_choice_sel = np.mean(left_choice_sel, axis=0)
windows = windows[:-1]

#Plot all
f, axarr = plt.subplots(2,2, figsize=(15,10), sharey='row', sharex='col')

axarr[0,0].plot(time, leftsel, color='black')
axarr[0,0].fill_between(time, leftsel - leftsel_error, 
          leftsel + leftsel_error,
          color=['darkgray'])
axarr[0,0].set_ylabel('Selectivity (spikes/s)')

axarr[0,1].plot(time, rightsel, color='black')
axarr[0,1].fill_between(time, rightsel - rightsel_error, 
          rightsel + rightsel_error,
          color=['darkgray'])

axarr[0,0].set_title('Left ALM')
axarr[0,1].set_title('Right ALM')
axarr[0,0].axhline(0, color='black', ls='--')
axarr[0,1].axhline(0, color='black', ls='--')

axarr[1,0].plot(windows, left_sample_sel, color='green', label='Stimulus selective')
axarr[1,0].plot(windows, left_choice_sel, color='purple', label='Choice selective')

axarr[1,1].plot(windows, right_sample_sel, color='green')
axarr[1,1].plot(windows, right_choice_sel, color='purple')

axarr[1,0].set_ylabel('Frac. of neurons')

for i in range(2):
    for j in range(2):
        axarr[i, j].axvline(s1.sample, color = 'grey', alpha=0.5, ls = '--')
        axarr[i, j].axvline(s1.delay, color = 'grey', alpha=0.5, ls = '--')
        axarr[i, j].axvline(s1.response, color = 'grey', alpha=0.5, ls = '--')
        axarr[i, j].axhline(0, color = 'grey', alpha=0.5, ls = '--')
        
axarr[1,0].legend()
