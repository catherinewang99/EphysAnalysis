# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:48:12 2024

Analyze the quality of collected data

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

path = r'H:\ephys_data\CW47\python\2024_10_17'

paths = [r'H:\ephys_data\CW47\python\2024_10_17',
          r'H:\ephys_data\CW47\python\2024_10_20',
          r'H:\ephys_data\CW47\python\2024_10_21',
          r'H:\ephys_data\CW47\python\2024_10_22',
          r'H:\ephys_data\CW47\python\2024_10_24',
          r'H:\ephys_data\CW47\python\2024_10_25',
          ]

# paths = [
#         r'J:\ephys_data\CW48\python\2024_10_29',
#         r'J:\ephys_data\CW48\python\2024_10_30',
#         r'J:\ephys_data\CW48\python\2024_10_31',
#         r'J:\ephys_data\CW48\python\2024_11_01',
#         r'J:\ephys_data\CW48\python\2024_11_02',
#         r'J:\ephys_data\CW48\python\2024_11_03',
#         r'J:\ephys_data\CW48\python\2024_11_04',
#         r'J:\ephys_data\CW48\python\2024_11_05',
#         r'J:\ephys_data\CW48\python\2024_11_06',
#           ]


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

#%% Plots    
## Plot nums

f = plt.figure()
plt.plot(range(len(paths)), numl, color='blue', marker='o', label='Left ALM')
plt.plot(range(len(paths)), numr, color='red', marker='o', label='Right ALM')
plt.xticks(np.arange(len(paths)), np.arange(len(paths)) + 1)
plt.xlabel('Day of recording')
plt.ylabel('Number of neurons')
plt.legend()
plt.show()


## Plot sample selective neurons
f = plt.figure()
plt.plot(range(len(paths)), samplel, color='blue', marker='o', label='Left ALM')
plt.plot(range(len(paths)), sampler, color='red', marker='o', label='Right ALM')
plt.xticks(np.arange(len(paths)), np.arange(len(paths)) + 1)
plt.xlabel('Day of recording')
plt.ylabel('Number of sample selective neurons')
plt.legend()
plt.show()

## Plot delay selective neurons
f = plt.figure()
plt.plot(range(len(paths)), delayl, color='blue', marker='o', label='Left ALM')
plt.plot(range(len(paths)), delayr, color='red', marker='o', label='Right ALM')
plt.xticks(np.arange(len(paths)), np.arange(len(paths)) + 1)
plt.xlabel('Day of recording')
plt.ylabel('Number of delay selective neurons')
plt.legend()
plt.show()


## Plot action selective neurons
f = plt.figure()
plt.plot(range(len(paths)), actionl, color='blue', marker='o', label='Left ALM')
plt.plot(range(len(paths)), actionr, color='red', marker='o', label='Right ALM')
plt.xticks(np.arange(len(paths)), np.arange(len(paths)) + 1)
plt.xlabel('Day of recording')
plt.ylabel('Number of action selective neurons')
plt.legend()
plt.show()


#%% Plot L Selective neurons
f, ax = plt.subplots(1,2, sharey='row', figsize=(10,5))
ax[0].plot(range(len(paths)), actionl, color='brown', marker='o', label='Action')
ax[0].plot(range(len(paths)), delayl, color='purple', marker='o', label='Delay')
ax[0].plot(range(len(paths)), samplel, color='green', marker='o', label='Sample')
ax[0].set_xticks(np.arange(len(paths)), np.arange(len(paths)) + 1)
ax[0].set_xlabel('Day of recording')
ax[0].set_ylabel('Number of selective neurons')
ax[0].set_title('L ALM selective neurons (p=0.05)')


# Plot R Selective neurons
ax[1].plot(range(len(paths)), actionr, color='brown', marker='o', label='Action')
ax[1].plot(range(len(paths)), delayr, color='purple', marker='o', label='Delay')
ax[1].plot(range(len(paths)), sampler, color='green', marker='o', label='Sample')
ax[1].set_xticks(np.arange(len(paths)), np.arange(len(paths)) + 1)
# ax[1].xlabel('Day of recording')
# ax[1].ylabel('Number of selective neurons')
ax[1].set_title('R ALM selective neurons (p=0.05)')
for i in range(2):
    ax[i].axhline(50, ls='--', color='grey')
    ax[i].axhline(100, ls='--', color='grey')
plt.legend()
plt.show()

#%% Look at the cell types

fs, ppyr = [],[]

samplefs, samplep = [],[]
delayfs, delayp = [],[]
actionfs, actionp = [],[]

for path in paths:
    
    s1 = Session(path, passive=False)#, side='R')

    fs += [len(s1.fs_idx)]    
    ppyr += [len(s1.pyr_idx)]    
    
    sample_sel = s1.get_epoch_selective((s1.sample, s1.delay), p=0.05)
    delay_sel = s1.get_epoch_selective((s1.delay, s1.response), p=0.05)
    action_sel = s1.get_epoch_selective((s1.response, s1.time_cutoff), p=0.05)

    samplefs += [len([i for i in sample_sel if i in s1.fs_idx])]
    samplep += [len([i for i in sample_sel if i in s1.pyr_idx])]
    delayfs += [len([i for i in delay_sel if i in s1.fs_idx])] 
    delayp += [len([i for i in delay_sel if i in s1.pyr_idx])]
    actionfs += [len([i for i in action_sel if i in s1.fs_idx])] 
    actionp += [len([i for i in action_sel if i in s1.pyr_idx])]
    
#%% Plot celltypes

## Plot nums

f = plt.figure()
plt.bar([0,1], [np.mean(fs), np.mean(ppyr)], color='grey')
plt.scatter(np.zeros(len(paths)), fs, color='blue',alpha=0.5)
plt.scatter(np.ones(len(paths)), ppyr, color='red', alpha=0.5)
plt.xticks([0,1],['FS', 'Ppyr'])
plt.xlabel('Cell type')
plt.ylabel('Number of neurons')
plt.legend()
plt.show()

# Plot FS Selective neurons
f, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].plot(range(len(paths)), actionfs, color='brown', marker='o', label='Action')
ax[0].plot(range(len(paths)), delayfs, color='purple', marker='o', label='Delay')
ax[0].plot(range(len(paths)), samplefs, color='green', marker='o', label='Sample')
ax[0].set_xticks(np.arange(len(paths)), np.arange(len(paths)) + 1)
ax[0].set_xlabel('Day of recording')
ax[0].set_ylabel('Number of selective neurons')
ax[0].set_title('FS ALM selective neurons (p=0.05)')


# Plot Pyr Selective neurons
ax[1].plot(range(len(paths)), actionp, color='brown', marker='o', label='Action')
ax[1].plot(range(len(paths)), delayp, color='purple', marker='o', label='Delay')
ax[1].plot(range(len(paths)), samplep, color='green', marker='o', label='Sample')
ax[1].set_xticks(np.arange(len(paths)), np.arange(len(paths)) + 1)
# ax[1].xlabel('Day of recording')
# ax[1].ylabel('Number of selective neurons')
ax[1].set_title('Ppyr selective neurons (p=0.05)')
# for i in range(2):
#     ax[i].axhline(50, ls='--', color='grey')
#     ax[i].axhline(100, ls='--', color='grey')
plt.legend()
plt.show()


#%% Plot waveforms

celltypes = [3,1] # FS and ppyr
for path in paths:
    
    s1 = Session(path, passive=False)#, side='R')
    f = plt.figure()
    for c in celltypes:
        neuron_cell_type = np.where(s1.celltype == c)[0]
        color = 'red' if c == 1 else 'black'
        for n in neuron_cell_type:
            
            plt.plot(np.arange(len(s1.get_single_waveform(n))),
                     s1.get_single_waveform(n),
                     color=color, alpha = 0.2)
            
    plt.show()

