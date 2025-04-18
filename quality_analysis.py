# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 16:11:07 2025

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from ephysSession import Session
# import behavior
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 


#%% PATHS

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

#%% Plot behavior curves for all mice


#%% Plot number of neurons for l and r hemispheres as a scatter bar plot
all_numl, all_numr = [],[]
all_samplel, all_sampler = [],[]
all_delayl, all_delayr = [],[]
all_actionl, all_actionr = [],[]
    
for paths in all_learning_paths:
    
    numl, numr = [],[]
    samplel, sampler = [],[]
    delayl, delayr = [],[]
    actionl, actionr = [],[]
    
    for path in paths:
        
        s1 = Session(path, passive=False)#, side='R')
    
        numl += [len(s1.L_alm_idx)]    
        numr += [len(s1.R_alm_idx)]    
        
        sample_sel = s1.get_epoch_selective((s1.sample, s1.delay), p=0.05)
        delay_sel = s1.get_epoch_selective((s1.delay, s1.response), p=0.05)
        action_sel = s1.get_epoch_selective((s1.response, s1.time_cutoff), p=0.05)
    
        samplel += [len([i for i in sample_sel if i in s1.L_alm_idx])]
        sampler += [len([i for i in sample_sel if i in s1.R_alm_idx])]
        delayl += [len([i for i in delay_sel if i in s1.L_alm_idx])] 
        delayr += [len([i for i in delay_sel if i in s1.R_alm_idx])]
        actionl += [len([i for i in action_sel if i in s1.L_alm_idx])] 
        actionr += [len([i for i in action_sel if i in s1.R_alm_idx])]
        
    all_numl += [numl]
    all_numr += [numr]
    all_samplel += [samplel]
    all_sampler += [sampler]
    all_delayl += [delayl]
    all_delayr += [delayr]
    all_actionl += [actionl]
    all_actionr += [actionr]

#%% PLOT

f=plt.figure()
plt.bar([0,1], [np.mean(cat(all_numl)), np.mean(cat(all_numr))])

for idx in range(len(all_numl)):
    plt.scatter(np.zeros(len(all_numl[idx])), all_numl[idx])
    plt.scatter(np.ones(len(all_numr[idx])), all_numr[idx])
    
plt.xticks([0,1],['Left ALM', 'Right ALM'])
plt.ylabel('Number of neurons')
plt.title('Expert sessions (n=3 mice)')
    

f, ax = plt.subplots(1,2, sharey='row', figsize=(10,5))
for idx in range(len(all_numl)):

    ax[0].scatter(np.ones(len(all_actionl[idx]))*2, all_actionl[idx], color='brown', marker='o', label='Action')
    ax[0].scatter(np.ones(len(all_delayl[idx])), all_delayl[idx], color='purple', marker='o', label='Delay')
    ax[0].scatter(np.zeros(len(all_samplel[idx])) , all_samplel[idx], color='green', marker='o', label='Sample')
    
    ax[1].scatter(np.ones(len(all_actionr[idx]))*2, all_actionr[idx], color='brown', marker='o', label='Action')
    ax[1].scatter(np.ones(len(all_delayr[idx])), all_delayr[idx], color='purple', marker='o', label='Delay')
    ax[1].scatter(np.zeros(len(all_sampler[idx])) , all_sampler[idx], color='green', marker='o', label='Sample')
    
    
ax[0].set_xticks([0,1,2],['Sample', 'Delay', 'Action'])
ax[0].set_xlabel('Selectivity')
ax[0].set_ylabel('Number of selective neurons')
ax[0].set_title('L ALM selective neurons (p=0.05)')


# Plot R Selective neurons

ax[1].set_xticks([0,1,2],['Sample', 'Delay', 'Action'])
# ax[1].xlabel('Day of recording')
# ax[1].ylabel('Number of selective neurons')
ax[1].set_title('R ALM selective neurons (p=0.05)')
for i in range(2):
    ax[i].axhline(50, ls='--', color='grey')
    ax[i].axhline(100, ls='--', color='grey')
# plt.legend()
plt.show()

    
# Fraction of selective neurons

f, ax = plt.subplots(1,2, sharey='row', figsize=(10,5))
for idx in range(len(all_numl)):

    ax[0].scatter(np.ones(len(all_actionl[idx]))*2, np.array(all_actionl[idx]) / np.array(all_numl[idx]), color='brown', marker='o', label='Action')
    ax[0].scatter(np.ones(len(all_delayl[idx])), np.array(all_delayl[idx]) / np.array(all_numl[idx]), color='purple', marker='o', label='Delay')
    ax[0].scatter(np.zeros(len(all_samplel[idx])) ,np.array(all_samplel[idx]) / np.array(all_numl[idx]), color='green', marker='o', label='Sample')
    
    ax[1].scatter(np.ones(len(all_actionr[idx]))*2, np.array(all_actionr[idx]) / np.array(all_numr[idx]), color='brown', marker='o', label='Action')
    ax[1].scatter(np.ones(len(all_delayr[idx])), np.array(all_delayr[idx]) / np.array(all_numr[idx]), color='purple', marker='o', label='Delay')
    ax[1].scatter(np.zeros(len(all_sampler[idx])) ,  np.array(all_sampler[idx]) / np.array(all_numr[idx]), color='green', marker='o', label='Sample')
    
    
ax[0].set_xticks([0,1,2],['Sample', 'Delay', 'Action'])
ax[0].set_xlabel('Selectivity')
ax[0].set_ylabel('Proportion of neurons selective')
ax[0].set_title('L ALM selective neurons (p=0.05)')


# Plot R Selective neurons

ax[1].set_xticks([0,1,2],['Sample', 'Delay', 'Action'])
# ax[1].xlabel('Day of recording')
# ax[1].ylabel('Number of selective neurons')
ax[1].set_title('R ALM selective neurons (p=0.05)')
for i in range(2):
    ax[i].axhline(0.5, ls='--', color='grey')
    ax[i].axhline(0.25, ls='--', color='grey')
# plt.legend()
plt.show()

#%% Effect of stim per neuron, scatter control vs stim
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


#%% Variable effect of stim over trials?

# For one session, plot effect of stim over trials per neuron?
s1 = Session(path, passive=False)#, side='R')
sided_neurons = s1.L_alm_idx
stim_trials = s1.i_good_L_stim_trials
# stim_trials = s1.i_good_non_stim_trials
window = (0.570 + 1.3 + 0.5, 0.570 + 1.3 + 1) # First second of delay

stim_spk = []
for n in sided_neurons:
    n_spk = []
    for t in stim_trials:
        n_spk += [s1.get_spike_rate(n, window, [t])]
    stim_spk += [n_spk]

f=plt.figure(figsize=(10,10))
for i in range(len(sided_neurons)):
    plt.scatter(np.ones(len(stim_trials)) * i, stim_spk[i])
plt.ylabel('spk rate across trials')
plt.xlabel('neurons')
plt.show()


f=plt.figure(figsize=(10,10))
for i in range(len(stim_trials)):
    plt.scatter(np.ones(len(sided_neurons)) * i, np.array(stim_spk)[:,i])
plt.ylabel('spk rate across neurons')
plt.xlabel('trials')
plt.show()


# alternatively, look at the variance across trials vs across neurons
neuron_variance = np.var(stim_spk, axis=1)
trial_variance = np.var(stim_spk, axis=0)
    
plt.scatter(np.zeros(len(neuron_variance)), neuron_variance)
plt.scatter(np.ones(len(trial_variance)), trial_variance)
plt.xticks([0,1],['Neuron variance', 'Trial variance'])

#%% Effect of stim olds

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




