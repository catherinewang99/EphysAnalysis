# -*- coding: utf-8 -*-
"""
Created on Wed May  7 10:33:21 2025

@author: catherinewang
"""

import probeinterface as pif
from probeinterface.plotting import plot_probe, plot_probe_group
import numpy as np

path = r'G:\data_tmp\CW59\20250222\catgt_CW59_20250222_g0\CW59_20250222_g0_imec0\CW59_20250222_g0_tcat.imec0.ap.meta'


probe = pif.read_spikeglx(path)

# pif.plotting.plot_probe(probe)

shanks = probe.shank_ids
path = r'G:\data_tmp\CW59\20250222\catgt_CW59_20250222_g0\CW59_20250222_g0_imec0\imec0_ks2_trimmed\\'


channel_map = np.load(path + r'channel_map.npy')
channel_positions = np.load(path + r'channel_positions.npy')

#%% Analysis across probes



import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from ephysSession import Session
import behavior
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 

# paths:
    
    
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
all_learning_paths_stimcorrected = [[r'G:\ephys_data\CW63\python\2025_03_19',
                          r'G:\ephys_data\CW63\python\2025_03_20',         
                          r'G:\ephys_data\CW63\python\2025_03_22',         
                           # r'G:\ephys_data\CW63\python\2025_03_23',         
                          r'G:\ephys_data\CW63\python\2025_03_25',],
                          
                          [r'G:\ephys_data\CW61\python\2025_03_08',
                           # r'G:\ephys_data\CW61\python\2025_03_09', 
                           r'G:\ephys_data\CW61\python\2025_03_10', 
                           r'G:\ephys_data\CW61\python\2025_03_11', 
                           # r'G:\ephys_data\CW61\python\2025_03_12', 
                           # r'G:\ephys_data\CW61\python\2025_03_14', 
                           r'G:\ephys_data\CW61\python\2025_03_17', 
                           # r'G:\ephys_data\CW61\python\2025_03_18', 
                           ],
                          [r'J:\ephys_data\CW54\python\2025_02_01',
                           r'J:\ephys_data\CW54\python\2025_02_03']
                          ]


#%% look at probe info, figure out which is anterior vs posterior


path = r'G:\ephys_data\CW59\python\2025_02_26'
s1 = Session(path, passive=False, filter_low_perf=True, filter_by_stim=True) # blue laser session

p=0.01
L_side_shank = s1.shank[s1.L_alm_idx]
R_side_shank = s1.shank[s1.R_alm_idx]
sample_sel = s1.get_epoch_selective((s1.sample, s1.delay), p=p)
delay_sel = s1.get_epoch_selective((s1.delay, s1.response), p=p)
action_sel = s1.get_epoch_selective((s1.response, s1.time_cutoff), p=p)

l_sample_sel = [i for i in sample_sel if i in s1.L_alm_idx]
r_sample_sel = [i for i in sample_sel if i in s1.R_alm_idx]

l_delay_sel = [i for i in delay_sel if i in s1.L_alm_idx]
r_delay_sel = [i for i in delay_sel if i in s1.R_alm_idx]

l_action_sel = [i for i in action_sel if i in s1.L_alm_idx]
r_action_sel = [i for i in action_sel if i in s1.R_alm_idx]

# Number of units per shank

for i in range(1,5):
    plt.bar([i], [sum(R_side_shank == i)])
plt.xticks(range(1,5))
plt.xlabel('Shank #')
plt.title('R ALM npxl: units per shank')
plt.show()


for i in range(1,5):
    plt.bar([i], [sum(L_side_shank == i)])
plt.xticks(range(1,5))
plt.xlabel('Shank #')
plt.title('L ALM npxl: units per shank')
plt.show()

# Proportion of selective units per shank
for i in range(1,5):
    plt.bar([i-0.3], [sum(s1.shank[l_sample_sel] == i) / sum(L_side_shank == i)], 0.15, color='yellow', label='Sample')
    
    plt.bar([i], [sum(s1.shank[l_delay_sel] == i) / sum(L_side_shank == i)], 0.15, color='red', label='Delay')
    
    plt.bar([i+0.3], [sum(s1.shank[l_action_sel] == i) / sum(L_side_shank == i)], 0.15, color='blue', label='Response')
    
plt.xticks(range(1,5))
plt.xlabel('Shank #')
plt.ylabel('Proportion of units')
plt.title('L ALM npxl: selective units per shank')
# plt.legend()
plt.show()

# Proportion of selective units per shank
for i in range(1,5):
    plt.bar([i-0.3], [sum(s1.shank[r_sample_sel] == i) / sum(R_side_shank == i)], 0.15, color='yellow', label='Sample')
    
    plt.bar([i], [sum(s1.shank[r_delay_sel] == i) / sum(R_side_shank == i)], 0.15, color='red', label='Delay')
    
    plt.bar([i+0.3], [sum(s1.shank[r_action_sel] == i) / sum(R_side_shank == i)], 0.15, color='blue', label='Response')
    
plt.xticks(range(1,5))
plt.xlabel('Shank #')
plt.ylabel('Proportion of units')
plt.title('R ALM npxl: selective units per shank')
# plt.legend()
plt.show()


#%% Shanks prop selective over all FOVs

l_sample_p, l_delay_p, l_action_p = [],[],[]
r_sample_p, r_delay_p, r_action_p = [],[],[]

for path in cat(all_expert_paths[1:]):

    
    s1 = Session(path, passive=False, filter_low_perf=True, filter_by_stim=True) # blue laser session
    
    p=0.01
    L_side_shank = s1.shank[s1.L_alm_idx]
    R_side_shank = s1.shank[s1.R_alm_idx]
    sample_sel = s1.get_epoch_selective((s1.sample, s1.delay), p=p)
    delay_sel = s1.get_epoch_selective((s1.delay, s1.response), p=p)
    action_sel = s1.get_epoch_selective((s1.response, s1.time_cutoff), p=p)
    
    l_sample_sel = [i for i in sample_sel if i in s1.L_alm_idx]
    r_sample_sel = [i for i in sample_sel if i in s1.R_alm_idx]
    
    l_delay_sel = [i for i in delay_sel if i in s1.L_alm_idx]
    r_delay_sel = [i for i in delay_sel if i in s1.R_alm_idx]
    
    l_action_sel = [i for i in action_sel if i in s1.L_alm_idx]
    r_action_sel = [i for i in action_sel if i in s1.R_alm_idx]
    
    
    l_sample_p += [[sum(s1.shank[l_sample_sel] == i) / sum(L_side_shank == i) for i in range(1,5)]] 
    l_delay_p += [[sum(s1.shank[l_delay_sel] == i) / sum(L_side_shank == i) for i in range(1,5)]] 
    l_action_p += [[sum(s1.shank[l_action_sel] == i) / sum(L_side_shank == i) for i in range(1,5)]] 
    
    r_sample_p += [[sum(s1.shank[r_sample_sel] == i) / sum(R_side_shank == i) for i in range(1,5)]] 
    r_delay_p += [[sum(s1.shank[r_delay_sel] == i) / sum(R_side_shank == i) for i in range(1,5)]] 
    r_action_p += [[sum(s1.shank[r_action_sel] == i) / sum(R_side_shank == i) for i in range(1,5)]] 
    # Number of units per shank
    
    # for i in range(1,5):
    #     plt.bar([i], [sum(R_side_shank == i)])
    # plt.xticks(range(1,5))
    # plt.xlabel('Shank #')
    # plt.title('R ALM npxl: units per shank')
    # plt.show()
    
    
    # for i in range(1,5):
    #     plt.bar([i], [sum(L_side_shank == i)])
    # plt.xticks(range(1,5))
    # plt.xlabel('Shank #')
    # plt.title('L ALM npxl: units per shank')
    # plt.show()
    
    # Proportion of selective units per shank
    
    # l_sample_p, l_delay_p, l_action_p = [],[],[]
    # r_sample_p, r_delay_p, r_action_p = [],[],[]
    
for i in range(4):
    plt.bar([i-0.3], [np.mean(l_sample_p, axis=0)[i]], 0.3, color='yellow', label='Sample')
    plt.scatter(np.ones(len(l_sample_p)) * (i-0.3), np.array(l_sample_p)[:, i], color='grey')
    
    plt.bar([i], [np.mean(l_delay_p, axis=0)[i]], 0.3, color='red', label='Delay')
    plt.scatter(np.ones(len(l_delay_p)) * (i), np.array(l_delay_p)[:, i], color='grey')

    plt.bar([i+0.3], [np.mean(l_action_p, axis=0)[i]], 0.3, color='blue', label='Response')
    plt.scatter(np.ones(len(l_action_p)) * (i+0.3), np.array(l_action_p)[:, i], color='grey')

plt.xticks(range(4), range(1,5))
plt.xlabel('Shank #')
plt.ylabel('Proportion of units')
plt.title('L ALM npxl: selective units per shank')
# plt.legend()
plt.show()

for i in range(4):
    plt.bar([i-0.3], [np.mean(r_sample_p, axis=0)[i]], 0.3, color='yellow', label='Sample')
    plt.scatter(np.ones(len(r_sample_p)) * (i-0.3), np.array(r_sample_p)[:, i], color='grey')
    
    plt.bar([i], [np.mean(r_delay_p, axis=0)[i]], 0.3, color='red', label='Delay')
    plt.scatter(np.ones(len(r_delay_p)) * (i), np.array(r_delay_p)[:, i], color='grey')

    plt.bar([i+0.3], [np.mean(r_action_p, axis=0)[i]], 0.3, color='blue', label='Response')
    plt.scatter(np.ones(len(r_action_p)) * (i+0.3), np.array(r_action_p)[:, i], color='grey')

plt.xticks(range(4), range(1,5))
plt.xlabel('Shank #')
plt.ylabel('Proportion of units')
plt.title('R ALM npxl: selective units per shank')
# plt.legend()
plt.show()

#%% Selectivity trace per shank AP axis
# Do left and right separately


left_sel = [[], [], [], []]

for path in cat(all_expert_paths[1:]):
    
    s1 = Session(path, passive=False, side='R')

    for i in range(4):
        p=0.05/len(s1.good_neurons)
        if len([n for n in s1.good_neurons if s1.shank[n] == i+1]) == 0:
            continue
        p=0.05/len([n for n in s1.good_neurons if s1.shank[n] == i+1])
        
        all_neurons = s1.get_epoch_selective(epoch=(s1.sample, s1.response + 2.5), p=p)
        all_neurons = [a for a in all_neurons if s1.shank[a] == i+1]
        if len(all_neurons) == 0:
            continue
        
        epochs = [(s1.sample, s1.response + 2.5) for _ in range(len(all_neurons))]
        epochs = [(s1.delay, s1.response) for _ in range(len(all_neurons))]
        sel, time = s1.plot_selectivity(all_neurons, binsize=200, timestep=50, return_pref_np=False, epoch=epochs)
        
        if len(left_sel[i]) == 0:
            left_sel[i] = np.array(sel)
        else:
            left_sel[i] = np.vstack((left_sel[i], sel))
    
f, axarr = plt.subplots(1,4, figsize=(16,4), sharey='row', sharex='col')
   
for j in range(4):     
    windows = np.arange(-0.2, s1.time_cutoff, 0.2)
    
    leftsel = np.mean(left_sel[j], axis=0)
    # rightsel = np.mean(right_sel, axis=0)
    leftsel_error = np.std(left_sel[j], axis=0) / np.sqrt(len(left_sel[j]))
    # rightsel_error = np.std(right_sel, axis=0) / np.sqrt(len(right_sel))
    
    windows = windows[:-1]
    
    #Plot all
    
    axarr[j].plot(time, leftsel, color='black')
    axarr[j].fill_between(time, leftsel - leftsel_error, 
              leftsel + leftsel_error,
              color=['darkgray'])
    
    axarr[j].axhline(0, color='grey', ls='--')

    axarr[j].axvline(s1.sample, color = 'grey', alpha=0.5, ls = '--')
    axarr[j].axvline(s1.delay, color = 'grey', alpha=0.5, ls = '--')
    axarr[j].axvline(s1.response, color = 'grey', alpha=0.5, ls = '--')
    
    axarr[j].set_title('Shank {} (n={})'.format(j, left_sel[j].shape[0]))
    
axarr[0].set_ylabel('Selectivity (spikes/s)')

f.suptitle('Right ALM across shanks')




#%% proportion stim vs delay selective by shank (takes a long time)


# FIXME: take out selectiivty portion
# Get selectivity (spikes/s) Chen et al 2021 Fig 1C


left_sample_sel, right_sample_sel = [[], [], [], []], [[], [], [], []]
left_choice_sel, right_choice_sel = [[], [], [], []], [[], [], [], []]



for path in cat(all_expert_paths[1:]):
    s1 = Session(path, passive=False, side='L')
    
    for i in range(4):
        shank_neurons = [n for n in s1.good_neurons if s1.shank[n] == i+1]

        sample_sel = s1.count_significant_neurons_by_time(shank_neurons, mode='stimulus')
        delay_sel = s1.count_significant_neurons_by_time(shank_neurons, mode='choice')
    
        if len(sample_sel) != 0:
            left_sample_sel[i] += [np.array(sample_sel) / len(shank_neurons)]
        if len(delay_sel) != 0:
            left_choice_sel[i] += [np.array(delay_sel) / len(shank_neurons)]
    
    s1 = Session(path, passive=False, side='R')
    for i in range(4):
        shank_neurons = [n for n in s1.good_neurons if s1.shank[n] == i+1]

       
        sample_sel = s1.count_significant_neurons_by_time(shank_neurons, mode='stimulus')
        delay_sel = s1.count_significant_neurons_by_time(shank_neurons, mode='choice')  
        
        if len(sample_sel) != 0:
            right_sample_sel[i] += [np.array(sample_sel) / len(shank_neurons)]
        if len(delay_sel) != 0:
            right_choice_sel[i] += [np.array(delay_sel) / len(shank_neurons)]
    
#%%
right_sample_sel_mean, left_sample_sel_mean, right_choice_sel_mean,left_choice_sel_mean = [],[],[],[]

# Proportion selective: 200 ms time bins, p < 0.05
for i in range(4):
    right_sample_sel_mean += [np.mean(right_sample_sel[i], axis=0)]
    left_sample_sel_mean += [np.mean(left_sample_sel[i], axis=0)]
    right_choice_sel_mean += [np.mean(right_choice_sel[i], axis=0)]
    left_choice_sel_mean += [np.mean(left_choice_sel[i], axis=0)]
# windows = windows[:-1]

#Plot all
f, axarr = plt.subplots(2,4, figsize=(16,8), sharey='row', sharex='col')



windows = np.arange(-0.2, s1.time_cutoff, 0.2)
for i in range(4):
    axarr[0,i].plot(windows, left_sample_sel_mean[i], color='green', label='Stimulus selective')
    axarr[0,i].plot(windows, left_choice_sel_mean[i], color='purple', label='Choice selective')
    
    axarr[1,i].plot(windows, right_sample_sel_mean[i], color='green')
    axarr[1,i].plot(windows, right_choice_sel_mean[i], color='purple')
    
    
    axarr[1,i].axhline(0.05, color = 'grey', alpha=0.5, ls = '--')
    axarr[1,i].axhline(0.05, color = 'grey', alpha=0.5, ls = '--')

for i in range(2):
    for j in range(4):
        axarr[i, j].axvline(s1.sample, color = 'grey', alpha=0.5, ls = '--')
        axarr[i, j].axvline(s1.delay, color = 'grey', alpha=0.5, ls = '--')
        axarr[i, j].axvline(s1.response, color = 'grey', alpha=0.5, ls = '--')
axarr[1,0].legend()
axarr[1,0].set_ylabel('Frac. of neurons')
plt.show()
