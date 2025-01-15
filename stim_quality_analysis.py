# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:27:09 2024

@author: catherinewang
"""


import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from ephysSession import Session
from matplotlib.pyplot import figure
# import decon
from scipy.stats import chisquare
import pandas as pd
plt.rcParams['pdf.fonttype'] = '42' 
import random
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

## Paths

path = r'J:\ephys_data\CW49\python\2024_12_14'



paths = [
            r'J:\ephys_data\CW49\python\2024_12_11',
            r'J:\ephys_data\CW49\python\2024_12_12',
            r'J:\ephys_data\CW49\python\2024_12_13',
            r'J:\ephys_data\CW49\python\2024_12_14',
            r'J:\ephys_data\CW49\python\2024_12_15',
            r'J:\ephys_data\CW49\python\2024_12_16',
        
        ]
s1 = Session(path, passive=False)


#%% Plot distribution of waveform withs

values = []
for path in paths:
    s1 = Session(path, passive=False)
    for n in range(s1.num_neurons):
        
        # if s1.celltype[n] == 3:# or s1.celltype[n] == 3:
        
        values += [s1.get_waveform_width(n)] # Plot in ms

f = plt.figure()
plt.hist(values)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.axvline(0.35, color='grey', ls = '--')
plt.axvline(0.45, color='grey', ls = '--')
plt.ylabel('Number of neurons')
plt.xlabel('Spike trough-to-peak (ms)')
#%% Plot waveforms by cell type: FS and ppyr
celltypes = [3,1] # FS and ppyr

f = plt.figure()
for path in paths:
    s1 = Session(path, passive=False)
    for c in celltypes:
        neuron_cell_type = np.where(s1.celltype == c)[0]
        color = 'red' if c == 1 else 'black'
        for n in neuron_cell_type:
            
            plt.plot(np.arange(len(s1.get_single_waveform(n))),
                     s1.get_single_waveform(n),
                     color=color, alpha = 0.2)
            
        
#%% Plot baseline spike rate vs spike width
f = plt.figure()

widths = []
spkrt_baseline = []

for n in range(s1.num_neurons):
    
    width = s1.get_waveform_width(n)
    widths += [width]
    all_stable_trials = [t for t in range(s1.num_trials) if t in s1.stable_trials[n]]
    baseline = s1.get_spike_rate(n, (0.07, 0.57), all_stable_trials)
    spkrt_baseline += baseline
    
    if width < 0.35:
        plt.scatter(width, baseline, color='red')
    elif width > 0.45:
        plt.scatter(width, baseline, color='black')
    else: 
        plt.scatter(width, baseline, facecolors='none', edgecolors='brown')

# plt.yscale('log')
plt.ylabel('Spike rate, baseline (spks/s)')
plt.xlabel('Spike trough-to-peak (ms)')

#%% Get avg spk count for stim  vs condition over cell types per neuron
# seperate left vs right ALM recordings, only consider the ipsi stim condition





f = plt.figure()

FS_ppyr_idx = []
side = 'R'
window = (0.570 + 1.3, 0.570 + 1.3 + 1) # First second of delay




allmeans, allerrs = [], []
for path in paths:
    s1 = Session(path, passive=False, side=side)
    
    neuron_cell_type = np.where(s1.celltype == 3)[0]
    neuron_cell_type = [n for n in neuron_cell_type if n in s1.good_neurons]
    old_spk_rate = []
    exclude_n = []
    means, errs = [], []
    # for i in range(len(s1.all_stim_levels)): 
    #     level = s1.all_stim_levels[i]
    for i in range(2):
        level = [0.0, 0.6][i]
        
        avg_spk_rate = []
        for n in neuron_cell_type:
            stim_trials = np.where(s1.stim_level == level)[0]
            stim_trial_side = np.where(s1.stim_side==side)[0] if level != 0 else stim_trials
            stim_trials = [c for c in stim_trials if c in s1.stable_trials[n] and c in stim_trial_side]
            
            baseline = s1.get_spike_rate(n, (0.07, 0.57), stim_trials)
            
            if baseline < 1 or n in exclude_n:
                # avg_spk_rate += [0]
                exclude_n+=[n]
                continue
    
                    
            avg_spk_rate += [s1.get_spike_rate(n, window, stim_trials) / baseline]
        
        plt.scatter(np.ones(len(avg_spk_rate)) * i, avg_spk_rate, alpha=0.5)            
        # ax[j].plot
        # ax[j].set_yscale('log')
        # ax[j].set_ylim(0,1.2)
        if len(old_spk_rate) != 0:
            for k in range(len(avg_spk_rate)):
                plt.plot([i-1, i], [old_spk_rate[k], avg_spk_rate[k]], color='grey', alpha=0.2)        
        old_spk_rate = avg_spk_rate
        
        means += [np.mean(avg_spk_rate)]
        errs += [np.std(avg_spk_rate) / np.sqrt(len(avg_spk_rate))]
        
    allmeans += [means]
    allerrs += [errs]
    
plt.ylabel('Normalized spike rate (spks/s)')            
plt.xlabel('Stim power (mW)')
plt.xticks([0,1], [0, 1.5])  
plt.title('ppyr')          
plt.show()

catallmeans = np.mean(allmeans, axis=0)
catallerrs = np.sum(allerrs, axis=0)

f = plt.figure()
for i in range(len(allmeans)):
    plt.plot([0,1], allmeans[i], color='blue', alpha=0.3)
plt.errorbar([0,1], catallmeans, yerr=catallerrs)
plt.axhline(0.2, ls='--')
plt.ylabel('Normalized spike rate (spks/s)')            
plt.xlabel('Stim power (mW)')
plt.xticks([0,1], [0, 1.5])  
plt.title('ppyr')          
plt.show()



