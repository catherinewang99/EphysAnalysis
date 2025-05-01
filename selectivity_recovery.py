# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:06:43 2024

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

all_expert_paths = [[
                        # r'J:\ephys_data\CW49\python\2024_12_11',
                        # r'J:\ephys_data\CW49\python\2024_12_12',
                        r'J:\ephys_data\CW49\python\2024_12_13',
                        # r'J:\ephys_data\CW49\python\2024_12_14',
                        # r'J:\ephys_data\CW49\python\2024_12_15',
                        # r'J:\ephys_data\CW49\python\2024_12_16',
                
                          ],
                    [
                        r'J:\ephys_data\CW53\python\2025_01_27',
                        r'J:\ephys_data\CW53\python\2025_01_28',
                        r'J:\ephys_data\CW53\python\2025_01_29',
                        # r'J:\ephys_data\CW53\python\2025_01_30',
                        r'J:\ephys_data\CW53\python\2025_02_01',
                        r'J:\ephys_data\CW53\python\2025_02_02',
                          ],
                    
                    [r'G:\ephys_data\CW59\python\2025_02_22',
                     r'G:\ephys_data\CW59\python\2025_02_24',
                     r'G:\ephys_data\CW59\python\2025_02_25',
                     r'G:\ephys_data\CW59\python\2025_02_26',
                     r'G:\ephys_data\CW59\python\2025_02_28',
                     ]]


all_paths = cat((cat(all_expert_paths), cat(all_learning_paths_stimcorrected)))

#%% Single FOV view OLD
path = r'H:\ephys_data\CW47\python\2024_10_25'
path = r'J:\ephys_data\CW53\python\2025_02_01'
path = r'G:\ephys_data\CW59\python\2025_02_28'
# path = r'J:\ephys_data\CW54\python\2025_02_03'


s1 = Session(path, passive=False, side='L')
s1.good_neurons = [n for n in s1.good_neurons if n in np.where(s1.celltype == 3)[0]]
sel, selo_stimleft, selo_stimright, err, erro_stimleft, erro_stimright, time = s1.selectivity_optogenetics(epoch = (s1.delay, s1.response),
                                                                                                           p=0.05, 
                                                                                                           binsize=150, 
                                                                                                           timestep=50,
                                                                                                           bootstrap=True)

f, axarr = plt.subplots(1,2, sharey='row', figsize=(10,5))  

for i in range(2):
    axarr[i].plot(time, sel, 'black')
            
    axarr[i].fill_between(time, sel - err, 
              sel + err,
              color=['darkgray'])

axarr[0].plot(time, selo_stimleft, 'blue')
        
axarr[0].fill_between(time, selo_stimleft - erro_stimleft, 
          selo_stimleft + erro_stimleft,
          color=['lightblue'])       
axarr[0].hlines(y=max(cat((selo_stimleft, sel))), xmin=s1.delay, xmax=s1.delay+1, linewidth=10, color='blue')

axarr[1].plot(time, selo_stimright, 'blue')
        
axarr[1].fill_between(time, selo_stimright - erro_stimright, 
          selo_stimright + erro_stimright,
          color=['lightblue'])      
axarr[1].hlines(y=max(cat((selo_stimright, sel))), xmin=s1.delay, xmax=s1.delay+1, linewidth=10, color='blue')

for i in range(2):
    axarr[i].axvline(s1.sample, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(s1.delay, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axvline(s1.response, color = 'grey', alpha=0.5, ls = '--')
    axarr[i].axhline(0, color = 'grey', alpha=0.5, ls = '--')

axarr[0].set_title('Left stim') # (n = {} neurons)'.format(num_neurons))                  
axarr[1].set_title('Right stim') # (n = {} neurons)'.format(num_neurons))                  
axarr[0].set_xlabel('Time from Go cue (s)')
axarr[0].set_ylabel('Selectivity')

plt.suptitle('{} ALM recording ({} neurons)'.format(s1.side, len(s1.selective_neurons)))

plt.show()

#%% Single FOV view
path = r'H:\ephys_data\CW47\python\2024_10_25'
path = r'J:\ephys_data\CW53\python\2025_01_31'
# path = r'G:\ephys_data\CW59\python\2025_02_28'
               
                    # [r'G:\ephys_data\CW59\python\2025_02_22',
                    #  r'G:\ephys_data\CW59\python\2025_02_24',
                    #  r'G:\ephys_data\CW59\python\2025_02_25',
                    #  r'G:\ephys_data\CW59\python\2025_02_26',
                    #  r'G:\ephys_data\CW59\python\2025_02_28',
                    #  ]]
# path = r'J:\ephys_data\CW54\python\2025_02_03'
# [
                        # r'J:\ephys_data\CW49\python\2024_12_11',
                        # r'J:\ephys_data\CW49\python\2024_12_12',
                        # r'J:\ephys_data\CW49\python\2024_12_13',
                        # r'J:\ephys_data\CW49\python\2024_12_14',
                        # r'J:\ephys_data\CW49\python\2024_12_15',
                        # r'J:\ephys_data\CW49\python\2024_12_16',
                
                          # ]
                          
  # [r'G:\ephys_data\CW63\python\2025_03_19',
  #       r'G:\ephys_data\CW63\python\2025_03_20',         
  #       r'G:\ephys_data\CW63\python\2025_03_22',         
  #         # r'G:\ephys_data\CW63\python\2025_03_23',         
  #       r'G:\ephys_data\CW63\python\2025_03_25',]
                          
# path = r'G:\ephys_data\CW63\python\2025_03_25'

s1 = Session(path, passive=False, filter_low_perf=True, filter_by_stim=False, laser='red')
left_info, right_info = s1.selectivity_optogenetics(epoch = (s1.delay, s1.response),
                                                    p=0.05                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         , 
                                                    binsize=150, 
                                                    timestep=50,
                                                    bootstrap=True)
sel, selo_stimleft, selo_stimright, err, erro_stimleft, erro_stimright, time = left_info
sel_R, selo_stimleft_R, selo_stimright_R, err_R, erro_stimleft_R, erro_stimright_R, time = right_info

f, axarr = plt.subplots(2,2, sharey='row', figsize=(12,10))  

for i in range(2):
    axarr[i,0].plot(time, sel, 'black')
            
    axarr[i,0].fill_between(time, sel - err, 
              sel + err,
              color=['darkgray'])

    axarr[i,1].plot(time, sel_R, 'black')
            
    axarr[i,1].fill_between(time, sel_R - err_R, 
              sel_R + err_R,
              color=['darkgray'])
    
axarr[0,0].plot(time, selo_stimleft, 'blue')
        
axarr[0,0].fill_between(time, selo_stimleft - erro_stimleft, 
          selo_stimleft + erro_stimleft,
          color=['lightblue'])       
axarr[0,0].hlines(y=max(cat((selo_stimleft, sel))), xmin=s1.delay, xmax=s1.delay+1, linewidth=10, color='blue')

axarr[1,0].plot(time, selo_stimright, 'blue')
        
axarr[1,0].fill_between(time, selo_stimright - erro_stimright, 
          selo_stimright + erro_stimright,
          color=['lightblue'])      
axarr[1,0].hlines(y=max(cat((selo_stimright, sel))), xmin=s1.delay, xmax=s1.delay+1, linewidth=10, color='blue')


axarr[0,1].plot(time, selo_stimleft_R, 'blue')
        
axarr[0,1].fill_between(time, selo_stimleft_R - erro_stimleft_R, 
          selo_stimleft_R + erro_stimleft_R,
          color=['lightblue'])       
axarr[0,1].hlines(y=max(cat((selo_stimleft_R, sel_R))), xmin=s1.delay, xmax=s1.delay+1, linewidth=10, color='blue')

axarr[1,1].plot(time, selo_stimright_R, 'blue')
        
axarr[1,1].fill_between(time, selo_stimright_R - erro_stimright_R, 
          selo_stimright_R + erro_stimright_R,
          color=['lightblue'])      
axarr[1,1].hlines(y=max(cat((selo_stimright_R, sel_R))), xmin=s1.delay, xmax=s1.delay+1, linewidth=10, color='blue')



for i in range(2):
    for j in range(2):
        axarr[i,j].axvline(s1.sample, color = 'grey', alpha=0.5, ls = '--')
        axarr[i,j].axvline(s1.delay, color = 'grey', alpha=0.5, ls = '--')
        axarr[i,j].axvline(s1.response, color = 'grey', alpha=0.5, ls = '--')
        axarr[i,j].axhline(0, color = 'grey', alpha=0.5, ls = '--')

left_sel_n, right_sel_n = len([n for n in s1.selective_neurons if n in s1.L_alm_idx]), len([n for n in s1.selective_neurons if n in s1.R_alm_idx])
axarr[0,0].set_title('Left ALM (n={})'.format(left_sel_n)) # (n = {} neurons)'.format(num_neurons))                  
axarr[0,1].set_title('Right ALM (n={})'.format(right_sel_n)) # (n = {} neurons)'.format(num_neurons))                  
axarr[1,0].set_xlabel('Time from Go cue (s)')
axarr[0,0].set_ylabel('Selectivity')

# plt.suptitle('{} ALM recording ({} neurons)'.format(s1.side, len(all_control_sel)))

plt.show()




#%% Aggregate FOV view
allpaths = [r'G:\ephys_data\CW63\python\2025_03_19',
        r'G:\ephys_data\CW63\python\2025_03_20',         
        r'G:\ephys_data\CW63\python\2025_03_22',         
         # r'G:\ephys_data\CW63\python\2025_03_23',         
        r'G:\ephys_data\CW63\python\2025_03_25',]


for paths in all_learning_paths_stimcorrected:
    
    all_control_sel, all_opto_sel_stim_left, all_opto_sel_stim_right = [],[],[]
    
    all_control_sel_R, all_opto_sel_stim_left_R, all_opto_sel_stim_right_R = [],[],[]
    
    for path in paths:
        s1 = Session(path, passive=False)
    
        left_info, right_info = s1.selectivity_optogenetics(epoch = (s1.delay, s1.response),
                                                            p=0.05, 
                                                            binsize=150, 
                                                            timestep=50,
                                                            return_traces=True)
        
        control_sel, opto_sel_stim_left, opto_sel_stim_right, time = left_info
        control_sel_R, opto_sel_stim_left_R, opto_sel_stim_right_R, _ = right_info
        
        all_control_sel += control_sel
        all_opto_sel_stim_left += opto_sel_stim_left
        all_opto_sel_stim_right += opto_sel_stim_right
        
        all_control_sel_R += control_sel_R
        all_opto_sel_stim_left_R += opto_sel_stim_left_R
        all_opto_sel_stim_right_R += opto_sel_stim_right_R
        
    # Plot
    
    sel = np.mean(all_control_sel, axis=0)
    selo_stimleft = np.mean(all_opto_sel_stim_left, axis=0)
    selo_stimright = np.mean(all_opto_sel_stim_right, axis=0)
    
    err = np.std(control_sel, axis=0) / np.sqrt(len(all_control_sel))
    erro_stimleft = np.std(opto_sel_stim_left, axis=0) / np.sqrt(len(all_control_sel))
    erro_stimright = np.std(opto_sel_stim_right, axis=0) / np.sqrt(len(all_control_sel))
    
    sel_R = np.mean(all_control_sel_R, axis=0)
    selo_stimleft_R = np.mean(all_opto_sel_stim_left_R, axis=0)
    selo_stimright_R = np.mean(all_opto_sel_stim_right_R, axis=0)
    
    err_R = np.std(control_sel_R, axis=0) / np.sqrt(len(all_control_sel_R))
    erro_stimleft_R = np.std(opto_sel_stim_left_R, axis=0) / np.sqrt(len(all_control_sel_R))
    erro_stimright_R = np.std(opto_sel_stim_right_R, axis=0) / np.sqrt(len(all_control_sel_R))
    
    
    f, axarr = plt.subplots(2,2, sharey='row', figsize=(12,10))  
    
    for i in range(2):
        axarr[i,0].plot(time, sel, 'black')
                
        axarr[i,0].fill_between(time, sel - err, 
                  sel + err,
                  color=['darkgray'])
    
        axarr[i,1].plot(time, sel_R, 'black')
                
        axarr[i,1].fill_between(time, sel_R - err_R, 
                  sel_R + err_R,
                  color=['darkgray'])
        
    axarr[0,0].plot(time, selo_stimleft, 'blue')
            
    axarr[0,0].fill_between(time, selo_stimleft - erro_stimleft, 
              selo_stimleft + erro_stimleft,
              color=['lightblue'])       
    axarr[0,0].hlines(y=max(cat((selo_stimleft, sel))), xmin=s1.delay, xmax=s1.delay+1, linewidth=10, color='blue')
    
    axarr[1,0].plot(time, selo_stimright, 'blue')
            
    axarr[1,0].fill_between(time, selo_stimright - erro_stimright, 
              selo_stimright + erro_stimright,
              color=['lightblue'])      
    axarr[1,0].hlines(y=max(cat((selo_stimright, sel))), xmin=s1.delay, xmax=s1.delay+1, linewidth=10, color='blue')
    
    
    axarr[0,1].plot(time, selo_stimleft_R, 'blue')
            
    axarr[0,1].fill_between(time, selo_stimleft_R - erro_stimleft_R, 
              selo_stimleft_R + erro_stimleft_R,
              color=['lightblue'])       
    axarr[0,1].hlines(y=max(cat((selo_stimleft_R, sel_R))), xmin=s1.delay, xmax=s1.delay+1, linewidth=10, color='blue')
    
    axarr[1,1].plot(time, selo_stimright_R, 'blue')
            
    axarr[1,1].fill_between(time, selo_stimright_R - erro_stimright_R, 
              selo_stimright_R + erro_stimright_R,
              color=['lightblue'])      
    axarr[1,1].hlines(y=max(cat((selo_stimright_R, sel_R))), xmin=s1.delay, xmax=s1.delay+1, linewidth=10, color='blue')
    
    
    
    for i in range(2):
        for j in range(2):
            axarr[i,j].axvline(s1.sample, color = 'grey', alpha=0.5, ls = '--')
            axarr[i,j].axvline(s1.delay, color = 'grey', alpha=0.5, ls = '--')
            axarr[i,j].axvline(s1.response, color = 'grey', alpha=0.5, ls = '--')
            axarr[i,j].axhline(0, color = 'grey', alpha=0.5, ls = '--')
    
    axarr[0,0].set_title('Left ALM (n={})'.format(len(all_control_sel))) # (n = {} neurons)'.format(num_neurons))                  
    axarr[0,1].set_title('Right ALM (n={})'.format(len(all_control_sel_R))) # (n = {} neurons)'.format(num_neurons))                  
    axarr[1,0].set_xlabel('Time from Go cue (s)')
    axarr[0,0].set_ylabel('Selectivity')
    
    # plt.suptitle('{} ALM recording ({} neurons)'.format(s1.side, len(all_control_sel)))
    
    plt.show()

#%% Plot modularity as a proportion


all_opto_prop_stim_left, all_opto_prop_stim_right = [],[]
paths = [r'G:\ephys_data\CW59\python\2025_02_22',
  r'G:\ephys_data\CW59\python\2025_02_24',
  r'G:\ephys_data\CW59\python\2025_02_25',
  r'G:\ephys_data\CW59\python\2025_02_26',
  r'G:\ephys_data\CW59\python\2025_02_28',
  
  ]

# paths =                           [r'J:\ephys_data\CW54\python\2025_02_01',
#                             r'J:\ephys_data\CW54\python\2025_02_03']

paths = [r'G:\ephys_data\CW63\python\2025_03_19',
        r'G:\ephys_data\CW63\python\2025_03_20',         
        r'G:\ephys_data\CW63\python\2025_03_22',         
         # r'G:\ephys_data\CW63\python\2025_03_23',         
        r'G:\ephys_data\CW63\python\2025_03_25',]
for path in paths:
    s1 = Session(path, passive=False, filter_low_perf=True)
    
    left_info, right_info = s1.selectivity_optogenetics(epoch = (s1.delay, s1.response),
                                                        p=0.05, 
                                                        binsize=150, 
                                                        timestep=50,
                                                        return_traces=False,
                                                        bootstrap=True)
    sel, selo_stimleft, selo_stimright, err, erro_stimleft, erro_stimright, time = left_info
    sel_R, selo_stimleft_R, selo_stimright_R, err_R, erro_stimleft_R, erro_stimright_R, time = right_info
    

    period = np.where((time > s1.delay) & (time < s1.delay + 1))[0] # Coupling
    
    # period = np.where((time > s1.delay + 1) & (time < s1.delay + 3))[0] # Robustness
    
    all_opto_prop_stim_left += [np.mean(selo_stimright[period]) / np.mean(sel[period])]


    all_opto_prop_stim_right += [np.mean(selo_stimleft_R[period]) / np.mean(sel_R[period])]
    
# Filtering steps 

all_opto_prop_stim_left, all_opto_prop_stim_right = np.array(all_opto_prop_stim_left), np.array(all_opto_prop_stim_right)
all_opto_prop_stim_left[all_opto_prop_stim_left > 1] = 1
all_opto_prop_stim_right[all_opto_prop_stim_right > 1] = 1
all_opto_prop_stim_left[all_opto_prop_stim_left < 0] = 0
all_opto_prop_stim_right[all_opto_prop_stim_right < 0] = 0
# Plot as a scatter like in Chen et al 
# modularity of left vs right alm


    
f=plt.figure(figsize=(7,5))

plt.scatter(all_opto_prop_stim_left, all_opto_prop_stim_right)
plt.plot([0,1], [0,1], ls='--', color='black')
plt.ylabel('Modularity, Right ALM')
plt.xlabel('Modularity, Left ALM')

# plt.bar([0,1], [np.mean(all_opto_prop_stim_left), np.mean(all_opto_prop_stim_right)])
# plt.scatter(np.zeros(len(all_opto_prop_stim_left)), all_opto_prop_stim_left)
# plt.scatter(np.ones(len(all_opto_prop_stim_right)), all_opto_prop_stim_right)
# plt.xticks([0,1], ['Left stim', 'Right stim'])
# plt.ylabel('Modularity')

#%% Plot modularity against effect of photoinhibition (Fig S3 of Chen et al 2021)


all_norm_rate, all_norm_rate_err = [], []
for path in all_paths:
    stim_spk, ctl_spk = [],[]

    s1 = Session(path, passive=False)# , filter_low_perf=False)#, side='R')
    sided_neurons = s1.L_alm_idx
    
    stim_trials = s1.i_good_L_stim_trials
    window = (s1.delay + 0 , s1.delay + 1) # First second of delay
    
    for n in sided_neurons:
        ctl_rate = s1.get_spike_rate(n, window, s1.i_good_non_stim_trials)
        stim_rate = s1.get_spike_rate(n, window, stim_trials)
        if ctl_rate < 1:
            continue
        if stim_rate > ctl_rate:
            continue
        stim_spk += [stim_rate]
        ctl_spk += [ctl_rate]
    all_norm_rate += [np.mean(np.array(stim_spk) / np.array(ctl_spk))]
    all_norm_rate_err += [np.std(np.array(stim_spk) / np.array(ctl_spk)) / np.sqrt(len(stim_spk))]









