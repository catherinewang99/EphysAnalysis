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
cat = np.concatenate
## Paths

all_learning_paths = [[r'G:\ephys_data\CW63\python\2025_03_19',
                      r'G:\ephys_data\CW63\python\2025_03_20',         
                      r'G:\ephys_data\CW63\python\2025_03_22',         
                      r'G:\ephys_data\CW63\python\2025_03_23',         
                      r'G:\ephys_data\CW63\python\2025_03_25',],
                      
                      [r'G:\ephys_data\CW61\python\2025_03_08',
                       r'G:\ephys_data\CW61\python\2025_03_09', 
                       r'G:\ephys_data\CW61\python\2025_03_10', 
                       r'G:\ephys_data\CW61\python\2025_03_11', 
                       r'G:\ephys_data\CW61\python\2025_03_12', 
                       r'G:\ephys_data\CW61\python\2025_03_14', 
                       r'G:\ephys_data\CW61\python\2025_03_17', 
                       r'G:\ephys_data\CW61\python\2025_03_18', 
                       ],
                      [r'J:\ephys_data\CW54\python\2025_02_01',
                       r'J:\ephys_data\CW54\python\2025_02_03']
                      ]


all_expert_paths = [[
                        # r'J:\ephys_data\CW49\python\2024_12_11',
                        # r'J:\ephys_data\CW49\python\2024_12_12',
                        r'J:\ephys_data\CW49\python\2024_12_13',
                        r'J:\ephys_data\CW49\python\2024_12_14',
                        r'J:\ephys_data\CW49\python\2024_12_15',
                        r'J:\ephys_data\CW49\python\2024_12_16',
                
                          ],
                    [
                        r'J:\ephys_data\CW53\python\2025_01_27',
                        r'J:\ephys_data\CW53\python\2025_01_28',
                        r'J:\ephys_data\CW53\python\2025_01_29',
                        r'J:\ephys_data\CW53\python\2025_01_30',
                        r'J:\ephys_data\CW53\python\2025_02_01',
                        r'J:\ephys_data\CW53\python\2025_02_02',
                          ],
                    
                    [r'G:\ephys_data\CW59\python\2025_02_22',
                     r'G:\ephys_data\CW59\python\2025_02_24',
                     r'G:\ephys_data\CW59\python\2025_02_25',
                     r'G:\ephys_data\CW59\python\2025_02_26',
                     r'G:\ephys_data\CW59\python\2025_02_28',
                     ]]

all_naive_paths = [
        [r'J:\ephys_data\CW48\python\2024_10_29',
        r'J:\ephys_data\CW48\python\2024_10_30',
        r'J:\ephys_data\CW48\python\2024_10_31',
        r'J:\ephys_data\CW48\python\2024_11_01',
        r'J:\ephys_data\CW48\python\2024_11_02',
        r'J:\ephys_data\CW48\python\2024_11_03',
        r'J:\ephys_data\CW48\python\2024_11_04',
        r'J:\ephys_data\CW48\python\2024_11_05',
        r'J:\ephys_data\CW48\python\2024_11_06',],
        
                   [r'H:\ephys_data\CW47\python\2024_10_17',
          r'H:\ephys_data\CW47\python\2024_10_18',
          # r'H:\ephys_data\CW47\python\2024_10_19',
          r'H:\ephys_data\CW47\python\2024_10_20',
          r'H:\ephys_data\CW47\python\2024_10_21',
          r'H:\ephys_data\CW47\python\2024_10_22',
          # r'H:\ephys_data\CW47\python\2024_10_23',
          r'H:\ephys_data\CW47\python\2024_10_24',
          r'H:\ephys_data\CW47\python\2024_10_25',],
                   
                   [r'G:\ephys_data\CW65\python\2025_02_25',],
                    ]


#%% Plot distribution of waveform withs

values = []
for path in cat(all_learning_paths):
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
for path in cat(all_learning_paths):
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


#%% get avk spk rate on stim conditions

stim_spk, ctl_spk = [],[]

for path in cat(all_learning_paths):
    s1 = Session(path, passive=False)#, side='R')
    sided_neurons = s1.L_alm_idx
    
    stim_trials = s1.i_good_L_stim_trials
    window = (0.570 + 1.3 + 0.5, 0.570 + 1.3 + 1) # First second of delay
    
    for n in sided_neurons:
        ctl_rate = s1.get_spike_rate(n, window, s1.i_good_non_stim_trials)
        if ctl_rate < 1:
            continue
        stim_spk += [s1.get_spike_rate(n, window, stim_trials)]
        ctl_spk += [ctl_rate]
        
        
f = plt.figure()
plt.scatter(ctl_spk, stim_spk, color='blue')
plt.plot([0, max(ctl_spk)], [0, max(ctl_spk)], ls='--', color='black')
plt.ylabel('Photoinhibition (spks/s)')
plt.xlabel('Control (spks/s)')
plt.show()








#%% Get avg spk count for stim  vs condition over cell types per neuron
# seperate left vs right ALM recordings, only consider the ipsi stim condition

# Take out and save neurons that are activated by stim (likely sst interneurons)



f = plt.figure()

FS_ppyr_idx = []
side = 'L'
window = (0.570 + 1.3, 0.570 + 1.3 + 1) # First second of delay
# window = (0.570 + 1.3, 0.570 + 1.3 + 0.5) # Use shorter window




allmeans, allerrs = [], []
for path in paths:
    s1 = Session(path, passive=False, side=side)
    
    neuron_cell_type = np.where(s1.celltype == 3)[0]
    neuron_cell_type = [n for n in neuron_cell_type if n in s1.good_neurons]
    old_spk_rate = []
    exclude_n = []
    drop_idx = []
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
                # continue
            
            if s1.get_spike_rate(n, window, stim_trials) / baseline > 2 and i == 1:
                drop_idx += [n]
                    
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
    print(len(drop_idx))
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

#%% Plot waveforms and rasters of activated neurons

# waveforms


# Raster

n=neuron_cell_type[94]

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



#%% Get the non supressed pyramidal cells at the highest stim condition
# Plot the ppyr cells excluding the pFS neurons

window = (0.570 + 1.3, 0.570 + 1.3 + 1) # First second of delay
side = 'R'
drop_idx = []
catall_avg_spk_rt = []
ctlcatall_avg_spk_rt = []
for path in paths:
    s1 = Session(path, passive=False, side=side)
    
    # neuron_cell_type = np.where(s1.celltype == 3)[0] # ppyr
    # neuron_cell_type = [n for n in neuron_cell_type if n in s1.good_neurons]
    neuron_cell_type = s1.good_neurons
    
    level = s1.all_stim_levels[-1]
    all_avg_spk_rt = []

    for level in [0.0, 0.6]:
        avg_spk_rate = []
        for n in neuron_cell_type:
            # stim_trials = np.where(s1.stim_level == level)[0]
            # stim_trials = [c for c in stim_trials if c in s1.stable_trials[n]]
            
            stim_trials = np.where(s1.stim_level == level)[0]
            stim_trial_side = np.where(s1.stim_side==side)[0] if level != 0 else stim_trials
            stim_trials = [c for c in stim_trials if c in s1.stable_trials[n] and c in stim_trial_side]
            
            baseline = s1.get_spike_rate(n, (0.07, 0.57), stim_trials)
            if baseline < 1:
                avg_spk_rate += [0]
                drop_idx += [np.where(neuron_cell_type == n)[0][0]]
                continue
            
            if s1.get_spike_rate(n, window, stim_trials) / baseline > 3:
                drop_idx += [np.where(neuron_cell_type == n)[0][0]]
    
            avg_spk_rate += [s1.get_spike_rate(n, window, stim_trials) / baseline]
       
        all_avg_spk_rt += [avg_spk_rate]
    
    # Filter list, dropping neurons
    for i in range(len(all_avg_spk_rt)):
        all_avg_spk_rt[i] = [item for i, item in enumerate(all_avg_spk_rt[i]) if i not in drop_idx]
    
    # Plot filtered list
    for i in range(len(all_avg_spk_rt)):
    
        if i != 0:
            for j in range(len(all_avg_spk_rt[i])):    
                plt.plot([i-1, i], [all_avg_spk_rt[i-1][j], all_avg_spk_rt[i][j]], color='grey', alpha =0.3)
                
        plt.scatter(np.ones(len(all_avg_spk_rt[i])) * i, all_avg_spk_rt[i])
    
    ctlcatall_avg_spk_rt += [all_avg_spk_rt[0]]
    catall_avg_spk_rt += [all_avg_spk_rt[1]]
    
plt.xticks(range(len(s1.all_stim_levels)), ['Ctl', '1.5'])
plt.axhline(0.2, ls='--')
plt.title('ppyr cells')
plt.ylabel('Normalized spike rate (spks/s)')
plt.xlabel('Stim power (mW)')
plt.show()

#Plot population level plot

plt.errorbar([0, 1.5], [np.mean(cat(ctlcatall_avg_spk_rt)), np.mean(cat(catall_avg_spk_rt))], 
                         [np.std(cat(ctlcatall_avg_spk_rt)), np.std(cat(catall_avg_spk_rt))])
plt.title('ppyr cells')
plt.ylabel('Normalized spike rate (spks/s)')
plt.xlabel('Stim power (mW)')
plt.xscale('symlog')
plt.axhline(0.2, ls='--')

plt.show()

