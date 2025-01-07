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
from session import Session
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

s1 = Session(path, passive=False)

#%% Plot distribution of waveform withs

values = []
# for n in s1.L_alm_idx:
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

window = (0.570 + 1.3, 0.570 + 1.3 + 1)
control_trials = np.where(s1.stim_ON == False)[0] 


f, ax = plt.subplots(2,len(set(s1.celltype)), figsize=(13,4))

FS_ppyr_idx = []

s1 = Session(path, passive=False, side='L')

for j in range(len(set(s1.celltype))):
    neuron_cell_type = np.where(s1.celltype == j+1)[0]
    neuron_cell_type = [n for n in neuron_cell_type if n in s1.L_alm_idx]
    old_spk_rate = []
    for i in range(len(s1.all_stim_levels)): 
        level = s1.all_stim_levels[i]
        avg_spk_rate = []
        for n in neuron_cell_type:
            stim_trials = np.where(s1.stim_level == level)[0]
            stim_trials = [c for c in stim_trials if c in s1.stable_trials[n]]
            
            baseline = s1.get_spike_rate(n, (0.07, 0.57), stim_trials)
            if baseline < 1:
                avg_spk_rate += [0]
                continue
            if j == 2 and i == 4 and s1.get_spike_rate(n, window, stim_trials) / baseline > 2:
                FS_ppyr_idx += [n]                

            
            avg_spk_rate += [s1.get_spike_rate(n, window, stim_trials) / baseline]

        ax[j].scatter(np.ones(len(avg_spk_rate)) * i, avg_spk_rate, alpha=0.5)            
        # ax[j].plot
        # ax[j].set_yscale('log')
        # ax[j].set_ylim(0,1.2)
        if len(old_spk_rate) != 0:
            for k in range(len(neuron_cell_type)):
                ax[j].plot([i-1, i], [old_spk_rate[k], avg_spk_rate[k]], color='grey', alpha=0.2)        
        old_spk_rate = avg_spk_rate
        
            
            

    # ax[j].set_xticks(range(len(s1.all_stim_levels)), ['Ctl', '1.5', '3', '5', '10'])

ax[0].set_xlabel('Stim power (mW)')
ax[0].set_ylabel('Normalized spike rate (spks/s)')
ax[0].set_title('Cell type: FS')
ax[1].set_title('Cell type: intermediate')
ax[2].set_title('Cell type: ppyr')
ax[3].set_title('Cell type: pDS')
