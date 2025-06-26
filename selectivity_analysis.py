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
from ephysSession import Session
import behavior
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
                        r'J:\ephys_data\CW49\python\2024_12_11',
                        r'J:\ephys_data\CW49\python\2024_12_12',
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


#%% Just selectivity plot whole trial epoch

left_sel, right_sel = [], []
for path in cat(all_expert_paths):
    s1 = Session(path, passive=False, side='L', anterior_shank=True)

    p=0.05/len(s1.good_neurons)
    all_neurons = s1.get_epoch_selective(epoch=(s1.sample, s1.response + 2.5), p=p)
    epochs = [(s1.sample, s1.response + 2.5) for _ in range(len(all_neurons))]
    epochs = [(s1.delay, s1.response) for _ in range(len(all_neurons))]
    sel, _ = s1.plot_selectivity(all_neurons, binsize=200, timestep=50, return_pref_np=False, epoch=epochs)
    
    if len(left_sel) == 0:
        left_sel = np.array(sel)
    else:
        left_sel = np.vstack((left_sel, sel))

    s1 = Session(path, passive=False, side='R', anterior_shank=True)

    p=0.05/len(s1.good_neurons)
    all_neurons = s1.get_epoch_selective(epoch=(s1.sample, s1.response + 2.5), p=p)
    epochs = [(s1.sample, s1.response + 2.5) for _ in range(len(all_neurons))]
    epochs = [(s1.delay, s1.response) for _ in range(len(all_neurons))]
    sel, time = s1.plot_selectivity(all_neurons, binsize=200, timestep=50, return_pref_np=False, epoch=epochs)
    
    if len(right_sel) == 0:
        right_sel = np.array(sel)
    else:
        right_sel = np.vstack((right_sel, sel)) 
        
windows = np.arange(-0.2, s1.time_cutoff, 0.2)

leftsel = np.mean(left_sel, axis=0)
rightsel = np.mean(right_sel, axis=0)
leftsel_error = np.std(left_sel, axis=0) / np.sqrt(len(left_sel))
rightsel_error = np.std(right_sel, axis=0) / np.sqrt(len(right_sel))

windows = windows[:-1]

#Plot all
f, axarr = plt.subplots(1,2, figsize=(10,5), sharey='row', sharex='col')

axarr[0].plot(time, leftsel, color='black')
axarr[0].fill_between(time, leftsel - leftsel_error, 
          leftsel + leftsel_error,
          color=['darkgray'])
axarr[0].set_ylabel('Selectivity (spikes/s)')


axarr[1].plot(time, rightsel, color='black')
axarr[1].fill_between(time, rightsel - rightsel_error, 
          rightsel + rightsel_error,
          color=['darkgray'])

axarr[0].set_title('Left ALM (n={})'.format(left_sel.shape[0]))
axarr[1].set_title('Right ALM (n={})'.format(right_sel.shape[0]))
axarr[0].axhline(0, color='grey', ls='--')
axarr[1].axhline(0, color='grey', ls='--')
for j in range(2):
    axarr[j].axvline(s1.sample, color = 'grey', alpha=0.5, ls = '--')
    axarr[j].axvline(s1.delay, color = 'grey', alpha=0.5, ls = '--')
    axarr[j].axvline(s1.response, color = 'grey', alpha=0.5, ls = '--')


#%% Just selectivity plot delay epoch

left_sel, right_sel = [], []
for path in (all_learning_paths[2]):
    s1 = Session(path, passive=False, side='L', anterior_shank=True)

    p=0.05/len(s1.good_neurons)
    p=0.05
    all_neurons = s1.get_epoch_selective(epoch=(s1.delay, s1.response), p=p)
    epochs = [(s1.sample, s1.response + 2.5) for _ in range(len(all_neurons))]
    epochs = [(s1.delay, s1.response) for _ in range(len(all_neurons))]
    sel, _ = s1.plot_selectivity(all_neurons, binsize=200, timestep=50, return_pref_np=False, epoch=epochs)
    
    if len(left_sel) == 0:
        left_sel = np.array(sel)
    else:
        left_sel = np.vstack((left_sel, sel))

    s1 = Session(path, passive=False, side='R', anterior_shank=True)

    p=0.05/len(s1.good_neurons)
    p=0.05
    all_neurons = s1.get_epoch_selective(epoch=(s1.delay, s1.response), p=p)
    epochs = [(s1.sample, s1.response + 2.5) for _ in range(len(all_neurons))]
    epochs = [(s1.delay, s1.response) for _ in range(len(all_neurons))]
    sel, time = s1.plot_selectivity(all_neurons, binsize=200, timestep=50, return_pref_np=False, epoch=epochs)
    
    if len(right_sel) == 0:
        right_sel = np.array(sel)
    else:
        right_sel = np.vstack((right_sel, sel)) 
        
windows = np.arange(-0.2, s1.time_cutoff, 0.2)

leftsel = np.mean(left_sel, axis=0)
rightsel = np.mean(right_sel, axis=0)
leftsel_error = np.std(left_sel, axis=0) / np.sqrt(len(left_sel))
rightsel_error = np.std(right_sel, axis=0) / np.sqrt(len(right_sel))

windows = windows[:-1]

#Plot all
f, axarr = plt.subplots(1,2, figsize=(10,5), sharey='row', sharex='col')

axarr[0].plot(time, leftsel, color='black')
axarr[0].fill_between(time, leftsel - leftsel_error, 
          leftsel + leftsel_error,
          color=['darkgray'])
axarr[0].set_ylabel('Selectivity (spikes/s)')


axarr[1].plot(time, rightsel, color='black')
axarr[1].fill_between(time, rightsel - rightsel_error, 
          rightsel + rightsel_error,
          color=['darkgray'])

axarr[0].set_title('Left ALM (n={})'.format(left_sel.shape[0]))
axarr[1].set_title('Right ALM (n={})'.format(right_sel.shape[0]))
axarr[0].axhline(0, color='grey', ls='--')
axarr[1].axhline(0, color='grey', ls='--')
for j in range(2):
    axarr[j].axvline(s1.sample, color = 'grey', alpha=0.5, ls = '--')
    axarr[j].axvline(s1.delay, color = 'grey', alpha=0.5, ls = '--')
    axarr[j].axvline(s1.response, color = 'grey', alpha=0.5, ls = '--')
    
    
    
#%% Aggregate over all FOVs for both analyses - TAKES LONG TIME :^(



# Get selectivity (spikes/s) Chen et al 2021 Fig 1C
leftpref, leftnonpref = [], []
rightpref, rightnonpref = [], []
left_sel, right_sel = [], []

left_sample_sel, right_sample_sel = [],[]
left_choice_sel, right_choice_sel = [],[]



for path in cat(all_expert_paths):
    s1 = Session(path, passive=False, side='L', anterior_shank=True)

    p=0.05/len(s1.good_neurons)

    delay_neurons = s1.get_epoch_selective(epoch=(s1.delay, s1.response), p=p)
    sample_neurons = s1.get_epoch_selective(epoch=(s1.sample, s1.delay), p=p)
    response_neurons = s1.get_epoch_selective(epoch=(s1.response, s1.response + 2.5), p=p)
    
    epochs = [[(s1.delay, s1.response) for _ in range(len(delay_neurons))],
                [(s1.sample, s1.delay) for _ in range(len(sample_neurons))],
                [(s1.response, s1.response + 2.5) for _ in range(len(response_neurons))]]
    epochs = cat([t for t in epochs if len(t) != 0])
    
    sel, _ = s1.plot_selectivity(cat([delay_neurons, sample_neurons, response_neurons]).astype(int), 
                                 binsize=200, timestep=50, return_pref_np=False, epoch=epochs)
    
    if len(left_sel) == 0:
        left_sel = np.array(sel)
    else:
        left_sel = np.vstack((left_sel, sel))

    # leftpref += [pref]
    # leftnonpref += [nonpref]

    windows = np.arange(-0.2, s1.time_cutoff, 0.2)
    
    # sample_sel, delay_sel = [],[]
    # for t in range(len(windows)-1):
    #     sample_sel += [s1.get_number_selective((windows[t],windows[t+1]), mode='stimulus')]
    #     delay_sel += [s1.get_number_selective((windows[t],windows[t+1]), mode='choice')]
    sample_sel = s1.count_significant_neurons_by_time(s1.good_neurons, mode='stimulus')
    delay_sel = s1.count_significant_neurons_by_time(s1.good_neurons, mode='choice')
    
    if len(sample_sel) != 0:
        left_sample_sel += [np.array(sample_sel) / len(s1.good_neurons)]
    if len(delay_sel) != 0:
        left_choice_sel += [np.array(delay_sel) / len(s1.good_neurons)]
    
    s1 = Session(path, passive=False, side='R', anterior_shank=True)

    p=0.05/len(s1.good_neurons)

    delay_neurons = s1.get_epoch_selective(epoch=(s1.delay, s1.response), p=p)
    sample_neurons = s1.get_epoch_selective(epoch=(s1.sample, s1.delay), p=p)
    response_neurons = s1.get_epoch_selective(epoch=(s1.response, s1.response + 2.5), p=p)
    
    epochs = [[(s1.delay, s1.response) for _ in range(len(delay_neurons))],
                [(s1.sample, s1.delay) for _ in range(len(sample_neurons))],
                [(s1.response, s1.response + 2.5) for _ in range(len(response_neurons))]]
    epochs = cat([t for t in epochs if len(t) != 0])
    
    sel, time = s1.plot_selectivity(cat([delay_neurons, sample_neurons, response_neurons]).astype(int), 
                                    binsize=200, timestep=50, return_pref_np=False, epoch=epochs)
    
    if len(right_sel) == 0:
        right_sel = np.array(sel)
    else:
        right_sel = np.vstack((right_sel, sel))    # rightpref += [pref]
    # rightnonpref += [nonpref]
    
    # sample_sel, delay_sel = [],[]
    # for t in range(len(windows)-1):
    #     sample_sel += [s1.get_number_selective((windows[t],windows[t+1]), mode='stimulus')]
    #     delay_sel += [s1.get_number_selective((windows[t],windows[t+1]), mode='choice')]
    sample_sel = s1.count_significant_neurons_by_time(s1.good_neurons, mode='stimulus')
    delay_sel = s1.count_significant_neurons_by_time(s1.good_neurons, mode='choice')  
    
    if len(sample_sel) != 0:
        right_sample_sel += [np.array(sample_sel) / len(s1.good_neurons)]
    if len(delay_sel) != 0:
        right_choice_sel += [np.array(delay_sel) / len(s1.good_neurons)]

# leftpref, leftnonpref = cat(leftpref), cat(leftnonpref)
# rightpref, rightnonpref = cat(rightpref), cat(rightnonpref)
# leftsel = np.mean(np.array(leftpref)-np.array(leftnonpref), axis=0)
# leftsel_error = np.std(np.array(leftpref)-np.array(leftnonpref), axis=0) / np.sqrt(leftsel.shape[0])

# rightsel = np.mean(np.array(rightpref)-np.array(rightnonpref), axis=0)
# rightsel_error = np.std(np.array(rightpref)-np.array(rightnonpref), axis=0) / np.sqrt(rightsel.shape[0])

# Selectivity, left and right

leftsel = np.mean(left_sel, axis=0)
rightsel = np.mean(right_sel, axis=0)
leftsel_error = np.std(left_sel, axis=0) / np.sqrt(len(left_sel))
rightsel_error = np.std(right_sel, axis=0) / np.sqrt(len(right_sel))


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

axarr[0,0].set_title('Left ALM (n={})'.format(left_sel.shape[0]))
axarr[0,1].set_title('Right ALM (n={})'.format(right_sel.shape[0]))
axarr[0,0].axhline(0, color='grey', ls='--')
axarr[0,1].axhline(0, color='grey', ls='--')
windows = np.arange(-0.2, s1.time_cutoff, 0.2)

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
axarr[1,0].axhline(0.05, color = 'grey', alpha=0.5, ls = '--')
axarr[1,1].axhline(0.05, color = 'grey', alpha=0.5, ls = '--')
        
axarr[1,0].legend()

#%% Proportion of selective neurons as a bar graph (left vs right hemisphere)
paths = [
            # r'J:\ephys_data\CW49\python\2024_12_11',
            # r'J:\ephys_data\CW49\python\2024_12_12',
            r'J:\ephys_data\CW49\python\2024_12_13',
            r'J:\ephys_data\CW49\python\2024_12_14',
            r'J:\ephys_data\CW49\python\2024_12_15',
            r'J:\ephys_data\CW49\python\2024_12_16',
        
        ]


delay_right, delay_left = [], []
p=0.0001

for path in paths:
    s1 = Session(path, passive=False, side='L')
    s1.good_neurons = [n for n in s1.good_neurons if n in np.where(s1.celltype == 3)[0]]

    delay_neurons = s1.get_epoch_selective(epoch=(s1.response-1.5, s1.response), p=p)
    
    delay_left += [len(delay_neurons) / len(s1.good_neurons)]
    
    
    s1 = Session(path, passive=False, side='R')
    s1.good_neurons = [n for n in s1.good_neurons if n in np.where(s1.celltype == 3)[0]]

    delay_neurons = s1.get_epoch_selective(epoch=(s1.response-1.5, s1.response), p=p)

    delay_right += [len(delay_neurons) / len(s1.good_neurons)]

    
f=plt.figure(figsize=(5,7))

plt.bar([0,1], [np.mean(delay_left), np.mean(delay_right)])
plt.scatter(np.zeros(len(delay_left)), delay_left)
plt.scatter(np.ones(len(delay_right)), delay_right)
plt.xticks([0,1], ['Left', 'Right'])
plt.ylabel('Proportion of selective neurons')





