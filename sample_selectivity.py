# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:07:14 2025

@author: catherinewang
"""


import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from ephysSession import Session
import behavior
from activitymode import Mode

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

allpaths = [[
    r'J:\ephys_data\CW53\python\2025_01_27',
    r'J:\ephys_data\CW53\python\2025_01_28',
    r'J:\ephys_data\CW53\python\2025_01_29',
    # r'J:\ephys_data\CW53\python\2025_01_30',
    r'J:\ephys_data\CW53\python\2025_02_01',
    r'J:\ephys_data\CW53\python\2025_02_02',
      ]]

# allpaths = [[r'G:\ephys_data\CW59\python\2025_02_22',
#   r'G:\ephys_data\CW59\python\2025_02_24',
#   r'G:\ephys_data\CW59\python\2025_02_25',
#   r'G:\ephys_data\CW59\python\2025_02_26',
#   r'G:\ephys_data\CW59\python\2025_02_28',
  
#   ]]

#%% Sample mode decoding accuracy
path =  r'G:\ephys_data\CW59\python\2025_02_24'
all_rda, all_lda = [], []
for paths in all_expert_paths:
    
    r_da, l_da = [],[]
    
    for path in paths:

        s1 = Mode(path, side='R')#, timestep=1)#, passive=False)
        # s1.plot_CD(mode_input='stimulus')
        _, _, db, choice = s1.decision_boundary(mode_input='stimulus')
        r_da += [np.mean(choice)]
        
        s1 = Mode(path, side='L')#, timestep=1)#, passive=False)
        # s1.plot_CD(mode_input='stimulus')
        _, _, db, choice = s1.decision_boundary(mode_input='stimulus')
        l_da += [np.mean(choice)]
        
    all_rda += [r_da]
    all_lda += [l_da]
    
f = plt.figure()
plt.bar(np.arange(len(all_lda)) - 0.2, [np.mean(a) for a in all_lda], 0.4, label='L ALM')
plt.bar(np.arange(len(all_rda)) + 0.2, [np.mean(a) for a in all_rda], 0.4, label='R ALM')

for i in range(len(all_rda)):
    plt.scatter(np.ones(len(all_rda[i])) * i + 0.2, all_rda[i])
    plt.scatter(np.ones(len(all_lda[i])) * i - 0.2, all_lda[i])
    
plt.xticks([0,1,2], ['CW49', 'CW53', 'CW59'])
plt.ylabel('Decoding accuracy')
plt.axhline(0.5, ls='--', color='grey')
plt.legend()
#%% Proportion of sample selective neurons per animal
f, ax =plt.subplots(1,2, sharey='row')
allpropl, allpropr = [],[]
counter = 0
for paths in all_expert_paths:
    numl, numr, samplel, sampler = [],[],[],[]
    propl, propr = [],[]
    for path in paths:
        
        s1 = Session(path, passive=False, filter_low_perf=True)#, side='R')
    
        numl += [len(s1.L_alm_idx)]    
        numr += [len(s1.R_alm_idx)]    
        
        sample_sel = s1.get_epoch_selective((s1.sample, s1.delay), p=0.05)
        delay_sel = s1.get_epoch_selective((s1.delay, s1.response), p=0.05)
        action_sel = s1.get_epoch_selective((s1.response, s1.time_cutoff), p=0.05)
    
        samplel += [len([i for i in sample_sel if i in s1.L_alm_idx])]
        sampler += [len([i for i in sample_sel if i in s1.R_alm_idx])]
        
        propl += [len([i for i in sample_sel if i in s1.L_alm_idx]) / len(s1.L_alm_idx)]
        propr += [len([i for i in sample_sel if i in s1.R_alm_idx]) / len(s1.R_alm_idx)]
        
        # delayl += [len([i for i in delay_sel if i in s1.L_alm_idx])] 
        # delayr += [len([i for i in delay_sel if i in s1.R_alm_idx])]
        # actionl += [len([i for i in action_sel if i in s1.L_alm_idx])] 
        # actionr += [len([i for i in action_sel if i in s1.R_alm_idx])]
    allpropl += [propl]
    allpropr += [propr]
    
    ax[0].bar([counter], np.mean(propl), fill=False)
    ax[1].bar([counter], np.mean(propr), fill=False)
    
    ax[0].scatter(np.ones(len(propl)) * counter, propl)
    ax[1].scatter(np.ones(len(propr)) * counter, propr)
    
    counter += 1
    
for i in range(2):
    ax[i].set_xticks(range(3), ['CW49', 'CW53', 'CW59'])
    # ax[i].set_xticks(range(3), ['CW63', 'CW61', 'CW54'])
ax[0].set_title('L ALM')
ax[1].set_title('R ALM')
#%% Sample selectivity - get proportion selective (SLOW)
allpaths = [[
    r'J:\ephys_data\CW53\python\2025_01_27',
    r'J:\ephys_data\CW53\python\2025_01_28',
    r'J:\ephys_data\CW53\python\2025_01_29',
    # r'J:\ephys_data\CW53\python\2025_01_30',
    r'J:\ephys_data\CW53\python\2025_02_01',
    r'J:\ephys_data\CW53\python\2025_02_02',
      ]]

# allpaths = [[r'G:\ephys_data\CW59\python\2025_02_22',
#   r'G:\ephys_data\CW59\python\2025_02_24',
#   r'G:\ephys_data\CW59\python\2025_02_25',
#   r'G:\ephys_data\CW59\python\2025_02_26',
#   r'G:\ephys_data\CW59\python\2025_02_28',
  
#   ]]

leftpref, leftnonpref = [], []
rightpref, rightnonpref = [], []
left_sel, right_sel = [], []

left_sample_sel, right_sample_sel = [],[]
left_choice_sel, right_choice_sel = [],[]



for path in cat(allpaths):
    s1 = Session(path, passive=False, side='L')


    windows = np.arange(-0.2, s1.time_cutoff, 0.2)
    

    sample_sel = s1.count_significant_neurons_by_time(s1.good_neurons, mode='stimulus')
    delay_sel = s1.count_significant_neurons_by_time(s1.good_neurons, mode='choice')
    
    if len(sample_sel) != 0:
        left_sample_sel += [np.array(sample_sel) / len(s1.good_neurons)]
    if len(delay_sel) != 0:
        left_choice_sel += [np.array(delay_sel) / len(s1.good_neurons)]
    
    s1 = Session(path, passive=False, side='R')

   
   
    sample_sel = s1.count_significant_neurons_by_time(s1.good_neurons, mode='stimulus')
    delay_sel = s1.count_significant_neurons_by_time(s1.good_neurons, mode='choice')  
    
    if len(sample_sel) != 0:
        right_sample_sel += [np.array(sample_sel) / len(s1.good_neurons)]
    if len(delay_sel) != 0:
        right_choice_sel += [np.array(delay_sel) / len(s1.good_neurons)]





# Proportion selective: 200 ms time bins, p < 0.05
right_sample_sel = np.mean(right_sample_sel, axis=0)
left_sample_sel = np.mean(left_sample_sel, axis=0)
right_choice_sel = np.mean(right_choice_sel, axis=0)
left_choice_sel = np.mean(left_choice_sel, axis=0)
# windows = windows[:-1]


#Plot all
f, axarr = plt.subplots(1,2, figsize=(10,5), sharey='row', sharex='col')

# axarr[0].plot(time, leftsel, color='black')
# axarr[0].fill_between(time, leftsel - leftsel_error, 
#           leftsel + leftsel_error,
#           color=['darkgray'])
# axarr[0].set_ylabel('Selectivity (spikes/s)')


# axarr[1].plot(time, rightsel, color='black')
# axarr[1].fill_between(time, rightsel - rightsel_error, 
#           rightsel + rightsel_error,
#           color=['darkgray'])
axarr[0].plot(windows, left_sample_sel, color='green', label='Stimulus selective')
axarr[0].plot(windows, left_choice_sel, color='purple', label='Choice selective')

axarr[1].plot(windows, right_sample_sel, color='green')
axarr[1].plot(windows, right_choice_sel, color='purple')

axarr[0].set_ylabel('Frac. of neurons')

axarr[0].set_title('Left ALM') #' (n={})'.format(left_sel.shape[0]))
axarr[1].set_title('Right ALM')#' (n={})'.format(right_sel.shape[0]))
axarr[0].axhline(0.05, color='grey', ls='--')
axarr[1].axhline(0.05, color='grey', ls='--')
for j in range(2):
    axarr[j].axvline(s1.sample, color = 'grey', alpha=0.5, ls = '--')
    axarr[j].axvline(s1.delay, color = 'grey', alpha=0.5, ls = '--')
    axarr[j].axvline(s1.response, color = 'grey', alpha=0.5, ls = '--')
    
axarr[0].legend()






#%% Make figure s1E for right vs left alm ppyr sample selectivity, per FOV












