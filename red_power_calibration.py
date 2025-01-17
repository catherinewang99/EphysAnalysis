# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:12:16 2024

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
# from multisession import multiSession

## Paths
allpaths = [
    r'H:\ephys_data\CW47\python\2024_10_19',
    r'H:\ephys_data\CW47\python\2024_10_20',
    r'H:\ephys_data\CW47\python\2024_10_21',
    r'H:\ephys_data\CW47\python\2024_10_22',
    r'H:\ephys_data\CW47\python\2024_10_23',
    r'H:\ephys_data\CW47\python\2024_10_24'
    ]


path = r'H:\ephys_data\CW47\python\2024_10_19'

s1 = Session(path, passive=True, laser = 'red')

# allsess = multiSession(allpaths, passive=True, laser = 'red')

#%% Plot distribution of waveform withs

values = []
# for n in s1.L_alm_idx:
for n in range(s1.num_neurons):
    
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
                 color=color, alpha = 0.3)
        
        
#%% Get avg spk count for diff stim condition over cell types per neuron
window = (0.570, 0.570 + 1.3 + 0.1)
control_trials = np.where(s1.stim_ON == False)[0] 


f, ax = plt.subplots(1,len(set(s1.celltype)), figsize=(13,4))

for j in range(len(set(s1.celltype))):
    neuron_cell_type = np.where(s1.celltype == j+1)[0]
    neuron_cell_type = [n for n in neuron_cell_type if n in s1.L_alm_idx]
    
    for i in range(len(s1.all_stim_levels)): 
        level = s1.all_stim_levels[i]
        avg_spk_rate = []
        for n in neuron_cell_type:
            stim_trials = np.where(s1.stim_level == level)[0]
            stim_trials = [c for c in stim_trials if c in s1.stable_trials[n]]
            
            baseline = s1.get_spike_rate(n, (0.07, 0.57), stim_trials)
            
            avg_spk_rate += [s1.get_spike_rate(n, window, stim_trials) / baseline]

        ax[j].scatter(np.ones(len(avg_spk_rate)) * i, avg_spk_rate, alpha=0.5)
        # ax[j].set_yscale('log')
    
    ax[j].set_xticks(range(len(s1.all_stim_levels)), ['Ctl', '1.5', '3', '5', '10'])

ax[0].set_xlabel('Stim power (mW)')
ax[0].set_ylabel('Normalized spike rate (spks/s)')
ax[0].set_title('Cell type: FS')
ax[1].set_title('Cell type: intermediate')
ax[2].set_title('Cell type: ppyr')
ax[3].set_title('Cell type: pDS')
#%% Per neuron, connected
window = (0.570, 0.570 + 1.3)
control_trials = np.where(s1.stim_ON == False)[0] 


f, ax = plt.subplots(1,len(set(s1.celltype)), figsize=(13,4))

FS_ppyr_idx = []

for j in range(len(set(s1.celltype))):
    neuron_cell_type = np.where(s1.celltype == j+1)[0]
    neuron_cell_type = [n for n in neuron_cell_type if n in s1.L_alm_idx]
    old_spk_rate = []

    for i in range(len(s1.all_stim_levels)): 
        level = s1.all_stim_levels[i]
        avg_spk_rate = []
        exclude_n = []

        for n in neuron_cell_type:
            stim_trials = np.where(s1.stim_level == level)[0]
            stim_trials = [c for c in stim_trials if c in s1.stable_trials[n]]
            
            baseline = s1.get_spike_rate(n, (0.07, 0.57), stim_trials)
            # if baseline < 1:
            #     avg_spk_rate += [0]
            #     continue
            
            if baseline < 1 or n in exclude_n:
                avg_spk_rate += [0]
                exclude_n+=[n]
                continue


            if j == 2 and i == 4 and s1.get_spike_rate(n, window, stim_trials) / baseline > 1.2:
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
        
            
            

    ax[j].set_xticks(range(len(s1.all_stim_levels)), ['Ctl', '1.5', '3', '5', '10'])

ax[0].set_xlabel('Stim power (mW)')
ax[0].set_ylabel('Normalized spike rate (spks/s)')
ax[0].set_title('Cell type: FS')
ax[1].set_title('Cell type: intermediate')
ax[2].set_title('Cell type: ppyr')
ax[3].set_title('Cell type: pDS')

#%% Get avg spk count for diff stim condition over cell types population level
window = (0.570, 0.570 + 1.3 + 0.1)
control_trials = np.where(s1.stim_ON == False)[0] 


f, ax = plt.subplots(1,len(set(s1.celltype)), figsize=(13,4))

for j in range(len(set(s1.celltype))):
    neuron_cell_type = np.where(s1.celltype == j+1)[0]
    neuron_cell_type = [n for n in neuron_cell_type if n in s1.L_alm_idx]
    
    allspkrt = []
    for i in range(len(s1.all_stim_levels)): 
        level = s1.all_stim_levels[i]
        avg_spk_rate = []
        allbaseline = []
        for n in neuron_cell_type:
            stim_trials = np.where(s1.stim_level == level)[0]
            stim_trials = [c for c in stim_trials if c in s1.stable_trials[n]]
            
            baseline = s1.get_spike_rate(n, (0.07, 0.57), stim_trials)
            
            avg_spk_rate += [s1.get_spike_rate(n, window, stim_trials)]
            allbaseline += [baseline]
        
        allspkrt += [np.mean(avg_spk_rate) / np.mean(allbaseline)]
            

    ax[j].plot([0, 1.5, 3, 5, 10], allspkrt, marker ='o')
    ax[j].set_yscale('log')
    ax[j].axhline(1, color='grey', ls='--')
    ax[j].axhline(0, color='grey', ls='--')
    # ax[j].set_xscale('log')
    

ax[0].set_xlabel('Stim power (mW)')
ax[0].set_ylabel('Normalized spike rate (spks/s)')
ax[0].set_title('Cell type: FS')
ax[1].set_title('Cell type: intermediate')
ax[2].set_title('Cell type: ppyr')
ax[3].set_title('Cell type: pDS')


#%% Get the non supressed pyramidal cells at the highest stim condition
# Plot the ppyr cells excluding the pFS neurons

window = (0.570, 0.570 + 1.3)

neuron_cell_type = np.where(s1.celltype == 3)[0] # ppyr
neuron_cell_type = [n for n in neuron_cell_type if n in s1.L_alm_idx]

level = s1.all_stim_levels[-1]
all_avg_spk_rt = []
drop_idx = []
for level in s1.all_stim_levels:
    avg_spk_rate = []
    for n in neuron_cell_type:
        stim_trials = np.where(s1.stim_level == level)[0]
        stim_trials = [c for c in stim_trials if c in s1.stable_trials[n]]
        
        baseline = s1.get_spike_rate(n, (0.07, 0.57), stim_trials)
        if baseline < 1:
            avg_spk_rate += [0]
            drop_idx += [np.where(neuron_cell_type == n)[0][0]]
            continue
        
        if s1.get_spike_rate(n, window, stim_trials) / baseline > 3:
            drop_idx += [np.where(neuron_cell_type == n)[0][0]]

        avg_spk_rate += [s1.get_spike_rate(n, window, stim_trials) / baseline]
   
    all_avg_spk_rt += [avg_spk_rate]
#
# Filter list, dropping neurons
for i in range(len(all_avg_spk_rt)):
    all_avg_spk_rt[i] = [item for i, item in enumerate(all_avg_spk_rt[i]) if i not in drop_idx]

# Plot filtered list
for i in range(len(all_avg_spk_rt)):

    if i != 0:
        for j in range(len(all_avg_spk_rt[i])):    
            plt.plot([i-1, i], [all_avg_spk_rt[i-1][j], all_avg_spk_rt[i][j]], color='grey', alpha =0.3)
            
    plt.scatter(np.ones(len(all_avg_spk_rt[i])) * i, all_avg_spk_rt[i])
    
plt.xticks(range(len(s1.all_stim_levels)), ['Ctl', '1.5', '3', '5', '10'])
plt.axhline(0.2, ls='--')
plt.title('ppyr cells')
plt.ylabel('Normalized spike rate (spks/s)')
plt.xlabel('Stim power (mW)')
plt.show()

#Plot population level plot

plt.errorbar([0, 1.5, 3, 5, 10], np.mean(all_avg_spk_rt, axis=1), np.std(all_avg_spk_rt, axis=1))
plt.title('ppyr cells')
plt.ylabel('Normalized spike rate (spks/s)')
plt.xlabel('Stim power (mW)')
plt.xscale('symlog')
plt.axhline(0.2, ls='--')

plt.show()

# plt.scatter(range(len(avg_spk_rate)), avg_spk_rate)
# plt.show()
# plt.hist(avg_spk_rate)
# # filt = [1 if i > 0.2 and i < 1 else 0 for i in avg_spk_rate ]
# filt = [1 if i > 0.2 else 0 for i in avg_spk_rate ]
# idx = np.where(filt)[0]
# # idx = np.where(np.array(avg_spk_rate) > 0.2 and np.array(avg_spk_rate) < 1)[0]
# # idx = [i for i in idx]
# # act_neurons = neuron_cell_type[np.array(idx)]



#%% Look at rasters of significantly excited/inhibited units
n=neuron_cell_type[75]

for stim_level in s1.all_stim_levels: 
# for stim_level in [s1.all_stim_levels[-1]]:
    stim_trials = np.where(s1.stim_level == stim_level)[0]
    stim_trials = [c for c in stim_trials if c in s1.stable_trials[n]]
                
    f = s1.plot_raster(n, window = (0.3, 0.5 + 1.8), trials=stim_trials)
    ax = f.gca()
    for x in np.arange(0.57+0.0125, 0.57+1.3125, 0.025): # Plot at the peak
        ax.axvline(x, color='lightblue', linewidth=0.5)
    ax.set_title('Stim power: {}'.format(stim_level))
    
    ax.set_xlim(0.57, 0.57+0.4)