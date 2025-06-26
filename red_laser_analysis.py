# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 12:16:20 2025

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
# from multisession import multiSession
import behavior


#%% Proportion of neurons significantly affected by laser -- ppyr
path = r'G:\ephys_data\CW63\python\2025_03_21'
# path = r'G:\ephys_data\CW61\python\2025_03_12'
path = r'J:\ephys_data\CW53\python\2025_01_31'
# path = r'G:\ephys_data\CW59\python\2025_02_27'
passive=False
s1 = Session(path, passive=passive, filter_low_perf=False, filter_by_stim=False, laser = 'red')

window = (0.57, 0.57+1.3) if passive else (0.57, 0.57+2.3) 


stim_trials = np.where(s1.stim_level == 2.7)[0]
control_trials = np.where(s1.stim_level == 0)[0]

sig_effect = []
for n in s1.L_alm_idx:
    val = s1.stim_effect_per_neuron(n, stim_trials, window=window, p=0.0001)
    sig_effect += [val]    
    
frac_supr, frac_exc, frac_none = sum(np.array(sig_effect) == -1), sum(np.array(sig_effect) == 1), sum(np.array(sig_effect) == 0)
l_info = np.array([frac_supr, frac_exc, frac_none]) / len(s1.L_alm_idx)

sig_effect_R = []
for n in s1.R_alm_idx:
    val = s1.stim_effect_per_neuron(n, stim_trials, window=window, p=0.0001)
    sig_effect_R += [val]    
    
frac_supr, frac_exc, frac_none = sum(np.array(sig_effect_R) == -1), sum(np.array(sig_effect_R) == 1), sum(np.array(sig_effect_R) == 0)
r_info = np.array([frac_supr, frac_exc, frac_none]) / len(s1.R_alm_idx)

plt.bar(np.arange(3) - 0.2, l_info, 0.4, color='red', label='Left')
plt.bar(np.arange(3) + 0.2, r_info, 0.4, color='blue', label='Right')
plt.xticks(np.arange(3), ['Supr.', 'Exc.', 'None'])
plt.legend()
plt.title(path)
plt.ylabel('Proportion of ppyr neurons')
plt.ylim(0, 0.75)
plt.show()

#%% Proportion of neurons significantly affected by laser -- fs
path = r'G:\ephys_data\CW63\python\2025_03_21'
# path = r'G:\ephys_data\CW61\python\2025_03_12'
path = r'J:\ephys_data\CW53\python\2025_01_31'
path = r'G:\ephys_data\CW59\python\2025_02_27'
passive=False

s1 = Session(path, passive=passive, filter_low_perf=False, filter_by_stim=False, laser = 'red', only_ppyr=False)

window = (0.57, 0.57+1.3) if passive else (0.57, 0.57+2.3) 
window = (0.57, 0.57+0.1)

stim_trials = np.where(s1.stim_level == 2.7)[0]
control_trials = np.where(s1.stim_level == 0)[0]

L_alm_fs = [n for n in s1.L_alm_idx if s1.celltype[n] == 1]
R_alm_fs = [n for n in s1.R_alm_idx if s1.celltype[n] == 1]


sig_effect = []
for n in L_alm_fs:
    val = s1.stim_effect_per_neuron(n, stim_trials, window=window, p=0.01)
    sig_effect += [val]    
    
frac_supr, frac_exc, frac_none = sum(np.array(sig_effect) == -1), sum(np.array(sig_effect) == 1), sum(np.array(sig_effect) == 0)
l_info = np.array([frac_supr, frac_exc, frac_none]) / len(L_alm_fs)

# sig_effect_R = []
# for n in R_alm_fs:
#     val = s1.stim_effect_per_neuron(n, stim_trials, window=window, p=0.01)
#     sig_effect_R += [val]    
    
# frac_supr, frac_exc, frac_none = sum(np.array(sig_effect_R) == -1), sum(np.array(sig_effect_R) == 1), sum(np.array(sig_effect_R) == 0)
# r_info = np.array([frac_supr, frac_exc, frac_none]) / len(R_alm_fs)

# plt.bar(np.arange(3) - 0.2, l_info, 0.4, color='red', label='Left')
# plt.bar(np.arange(3) + 0.2, r_info, 0.4, color='blue', label='Right')
# plt.xticks(np.arange(3), ['Supr.', 'Exc.', 'None'])
# plt.legend()
# plt.title(path)
# plt.ylabel('Proportion of ppyr neurons')
# # plt.ylim(0, 0.75)
# plt.show()



stim_trials = np.where(s1.stim_level == 2.7)[0]
control_trials = np.where(s1.stim_level == 0)[0]
sided_neurons = L_alm_fs

# Plot control vs stim scatter
    
# window = (0.57, 0.57+1.3) if passive else (0.57, 0.57+2.3) 
# window = (0.57, 0.57+2.3) # Behavior
stim_spk, ctl_spk = [],[]
middle_neurons, pint = [], []
norm_rate = []
f = plt.figure()

for i in range(len(sided_neurons)):
    n = sided_neurons[i]
    ctl_rate = s1.get_spike_rate(n, window, control_trials)
    # if ctl_rate < 1:
    #     continue
    stim_rate = s1.get_spike_rate(n, window, stim_trials)
    # if stim_rate/ctl_rate > 0.55: 
    #     continue
    # if stim_rate > ctl_rate: 
    #     pint = [n]
    #     continue
    stim_spk += [stim_rate]
    ctl_spk += [ctl_rate]
    
    norm_rate += [stim_rate / ctl_rate]
    
    # if stim_rate/ctl_rate > 0.4 and stim_rate/ctl_rate < 1:
    #     middle_neurons += [n]
    color='orange' if sig_effect[i] == 1 else 'blue'
    if sig_effect[i] == 0:
        color = 'grey'
    plt.scatter(ctl_rate, stim_rate, color=color)

# plt.scatter(ctl_spk, stim_spk, color=color)
    
plt.plot([0, max(ctl_spk)], [0, max(ctl_spk)], ls='--', color='black')
plt.ylabel('Photoinhibition (spks/s)')
plt.xlabel('Control (spks/s)')
plt.title('{}'.format(s1.path))
plt.show()

#%% plot single neurons
idx = np.where(np.array(sig_effect) == 1)[0]

s1.plot_raster_and_PSTH(s1.R_alm_idx[idx[3]], opto=True, stimside = 'L')

#%% Scatter analysis - plot overall effect on ppyr neurons
path = r'G:\ephys_data\CW63\python\2025_03_21'
# path = r'G:\ephys_data\CW61\python\2025_03_12'
path = r'J:\ephys_data\CW53\python\2025_01_31'
# path = r'G:\ephys_data\CW59\python\2025_02_27'

passive=False
s1 = Session(path, passive=passive, filter_low_perf=False, filter_by_stim=False, laser = 'red')


stim_trials = np.where(s1.stim_level == 2.7)[0]
control_trials = np.where(s1.stim_level == 0)[0]
sided_neurons = s1.L_alm_idx

# Plot control vs stim scatter
    
window = (0.57, 0.57+1.3) if passive else (0.57, 0.57+2.3) 
# window = (0.57, 0.57+2.3) # Behavior
stim_spk, ctl_spk = [],[]
middle_neurons, pint = [], []
norm_rate = []
f = plt.figure()

for i in range(len(sided_neurons)):
    n = sided_neurons[i]
    ctl_rate = s1.get_spike_rate(n, window, control_trials)
    # if ctl_rate < 1:
    #     continue
    stim_rate = s1.get_spike_rate(n, window, stim_trials)
    # if stim_rate/ctl_rate > 0.55: 
    #     continue
    # if stim_rate > ctl_rate: 
    #     pint = [n]
    #     continue
    stim_spk += [stim_rate]
    ctl_spk += [ctl_rate]
    
    norm_rate += [stim_rate / ctl_rate]
    
    # if stim_rate/ctl_rate > 0.4 and stim_rate/ctl_rate < 1:
    #     middle_neurons += [n]
    color='orange' if sig_effect[i] == 1 else 'blue'
    if sig_effect[i] == 0:
        color = 'grey'
    plt.scatter(ctl_rate, stim_rate, color=color)

# plt.scatter(ctl_spk, stim_spk, color=color)
    
plt.plot([0, max(ctl_spk)], [0, max(ctl_spk)], ls='--', color='black')
plt.ylabel('Photoinhibition (spks/s)')
plt.xlabel('Control (spks/s)')
plt.title('{}'.format(s1.path))
plt.show()





    
f = plt.figure()
plt.hist(stim_spk, color='blue', bins=50)
plt.xlabel('Photoinhibition (spks/s)')
plt.ylabel('Count')
plt.title('{}'.format(s1.path))
plt.show()

f = plt.figure()
plt.hist(norm_rate, color='blue', bins=50)
plt.xlabel('Normalized rate (spks/s)')
plt.ylabel('Count')
plt.title('{}'.format(s1.path))
plt.show()



#%% Plot neurons with laser peaks
n=42
n=141
n=s1.L_alm_idx[7]
# n=sided_neurons[np.argsort(norm_rate)[-10]]
# f = s1.plot_raster(n, window = (0.3, 0.5 + 1.8), trials=stim_trials)
f = s1.plot_raster(n, window = (0, 3), trials=stim_trials)
# f = s1.plot_raster(n, window = (0.3, 3), trials=control_trials)
ax = f.gca()
for x in np.arange(0.57+0.0125, 0.57+1.3125, 0.025): # Plot at the peak
    ax.axvline(x, color='lightblue', linewidth=0.5)
ax.set_title('Stim power: {}'.format(2.7))

ax.set_xlim(0.57+0.9, 0.57+1.6)

#%% Correlate # of significantly affected neurons with learning speed
paths = [ r'G:\ephys_data\CW63\python\2025_03_21',
         r'G:\ephys_data\CW61\python\2025_03_12',
         r'J:\ephys_data\CW53\python\2025_01_31',
         r'G:\ephys_data\CW59\python\2025_02_27',]

passives=[True, True, False, False]
l_frac, r_frac = [], []

for i in range(4):
    path = paths[i]
    passive = passives[i]
    s1 = Session(path, passive=passive, filter_low_perf=False, filter_by_stim=False, laser = 'red')
    
    window = (0.57, 0.57+1.3) if passive else (0.57, 0.57+2.3) 
    
    
    stim_trials = np.where(s1.stim_level == 2.7)[0]
    control_trials = np.where(s1.stim_level == 0)[0]

    sig_effect = []
    for n in s1.L_alm_idx:
        val = s1.stim_effect_per_neuron(n, stim_trials, window=window)
        sig_effect += [val]    
        
    frac_none = sum(np.array(sig_effect) == 0)
    l_frac += [(len(s1.L_alm_idx) - frac_none) / len(s1.L_alm_idx)]
    
    sig_effect = []
    for n in s1.R_alm_idx:
        val = s1.stim_effect_per_neuron(n, stim_trials, window=window)
        sig_effect += [val]    
        
    frac_none = sum(np.array(sig_effect) == 0)
    r_frac += [(len(s1.R_alm_idx) - frac_none) / len(s1.R_alm_idx)]


ephys_paths = [
    
    r'J:\ephys_data\Behavior data\CW63\python_behavior',
    r'J:\ephys_data\Behavior data\CW61\python_behavior',
    r'J:\ephys_data\Behavior data\CW53\python_behavior',
    r'J:\ephys_data\Behavior data\CW59\python_behavior',
    
    ]

delay_length = 1.3
performance = 0.75
window = 25

ephys_trials = []
for path in ephys_paths:
    b = behavior.Behavior(path, behavior_only=True)
    ephys_trials += [b.time_to_reach_perf(performance, delay_length, window=window)]
    
    
f=plt.figure()
plt.scatter(ephys_trials, l_frac, color='red')
plt.scatter(ephys_trials, r_frac, color='blue')


#%% Proportion of suppr/exc neurons that are delay/sample selective

paths = [r'J:\ephys_data\CW53\python\2025_01_31',
         r'G:\ephys_data\CW59\python\2025_02_27',]

l_frac_delay, r_frac_delay = [], []
l_frac_sample, r_frac_sample = [], []

for i in range(4):
    path = paths[i]
    s1 = Session(path, passive=False, filter_low_perf=False, filter_by_stim=False, laser = 'red')
    
    window = (0.57, 0.57+2.3) 
    
    stim_trials = np.where(s1.stim_level == 2.7)[0]
    control_trials = np.where(s1.stim_level == 0)[0]

    delay_sel = s1.get_epoch_selective((s1.delay, s1.response), p=0.01)
    sample_sel = s1.get_epoch_selective((s1.sample, s1.delay), p=0.01)

    sig_effect_delay, sig_effect_sample = [], []
    for n in s1.L_alm_idx: 
        if n in delay_sel:
            val = s1.stim_effect_per_neuron(n, stim_trials, window=window)
            sig_effect_delay += [val]
        elif n in sample_sel:
            val = s1.stim_effect_per_neuron(n, stim_trials, window=window)
            sig_effect_sample += [val]
        
    frac_supr, frac_exc, frac_none = sum(np.array(sig_effect_delay) == -1), sum(np.array(sig_effect_delay) == 1), sum(np.array(sig_effect_delay) == 0)
    l_frac_delay +=[ np.array([frac_supr, frac_exc, frac_none])] # / len(s1.R_alm_idx)
    frac_supr, frac_exc, frac_none = sum(np.array(sig_effect_sample) == -1), sum(np.array(sig_effect_sample) == 1), sum(np.array(sig_effect_sample) == 0)
    l_frac_sample += [np.array([frac_supr, frac_exc, frac_none])] # / len(s1.R_alm_idx)
    
    sig_effect_delay, sig_effect_sample = [], []
    for n in s1.R_alm_idx:
        if n in delay_sel:
            val = s1.stim_effect_per_neuron(n, stim_trials, window=window)
            sig_effect_delay += [val]
        elif n in sample_sel:
            val = s1.stim_effect_per_neuron(n, stim_trials, window=window)
            sig_effect_sample += [val]
        
    frac_supr, frac_exc, frac_none = sum(np.array(sig_effect_delay) == -1), sum(np.array(sig_effect_delay) == 1), sum(np.array(sig_effect_delay) == 0)
    r_frac_delay += [np.array([frac_supr, frac_exc, frac_none])] # / len(s1.R_alm_idx)
    frac_supr, frac_exc, frac_none = sum(np.array(sig_effect_sample) == -1), sum(np.array(sig_effect_sample) == 1), sum(np.array(sig_effect_sample) == 0)
    r_frac_sample += [np.array([frac_supr, frac_exc, frac_none])] # / len(s1.R_alm_idx)
    
#%% Selectivity recovery for red laser behavior sessions

path = r'J:\ephys_data\CW53\python\2025_01_31'
# path = r'G:\ephys_data\CW59\python\2025_02_27'

s1 = Session(path, passive=False, filter_low_perf=True, filter_by_stim=False, laser='red')
left_info, right_info = s1.selectivity_optogenetics(epoch = (s1.sample , s1.delay),
                                                    p=0.05                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         , 
                                                    binsize=150, 
                                                    timestep=50,
                                                    bootstrap=False)
sel, selo_stimleft, selo_stimright, err, erro_stimleft, erro_stimright, time = left_info
sel_R, selo_stimleft_R, selo_stimright_R, err_R, erro_stimleft_R, erro_stimright_R, time = right_info

f, axarr = plt.subplots(1,2, sharey='row', figsize=(10,5))  

axarr[0].plot(time, sel, 'black')
        
axarr[0].fill_between(time, sel - err, 
          sel + err,
          color=['darkgray'])

axarr[1].plot(time, sel_R, 'black')
        
axarr[1].fill_between(time, sel_R - err_R, 
          sel_R + err_R,
          color=['darkgray'])
    
axarr[0].plot(time, selo_stimleft, 'red')
        
axarr[0].fill_between(time, selo_stimleft - erro_stimleft, 
          selo_stimleft + erro_stimleft,
          color=['pink'])       
axarr[0].hlines(y=max(cat((selo_stimleft, sel))), xmin=s1.sample, xmax=s1.delay+1, linewidth=10, color='red')



axarr[1].plot(time, selo_stimleft_R, 'red')
        
axarr[1].fill_between(time, selo_stimleft_R - erro_stimleft_R, 
          selo_stimleft_R + erro_stimleft_R,
          color=['pink'])       
axarr[1].hlines(y=max(cat((selo_stimleft_R, sel_R))), xmin=s1.sample, xmax=s1.delay+1, linewidth=10, color='red')


for i in range(2):
    axarr[i].axvline(s1.sample, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(s1.delay, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(s1.response, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axhline(0, color = 'grey', alpha=0.5, ls = '--')

left_sel_n, right_sel_n = len([n for n in s1.selective_neurons if n in s1.L_alm_idx]), len([n for n in s1.selective_neurons if n in s1.R_alm_idx])
axarr[0].set_title('Left ALM (n={})'.format(left_sel_n)) # (n = {} neurons)'.format(num_neurons))                  
axarr[1].set_title('Right ALM (n={})'.format(right_sel_n)) # (n = {} neurons)'.format(num_neurons))                  
axarr[0].set_xlabel('Time from Go cue (s)')
axarr[0].set_ylabel('Selectivity')

# plt.suptitle('{} ALM recording ({} neurons)'.format(s1.side, len(all_control_sel)))

plt.show()

