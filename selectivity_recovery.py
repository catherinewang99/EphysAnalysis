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
from session import Session
import behavior
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 

#%% Single FOV view
path = r'H:\ephys_data\CW47\python\2024_10_25'
path = r'J:\ephys_data\CW49\python\2024_12_14'

s1 = Session(path, passive=False, side='R')
s1.good_neurons = [n for n in s1.good_neurons if n in np.where(s1.celltype == 3)[0]]
sel, selo_stimleft, selo_stimright, err, erro_stimleft, erro_stimright, time = s1.selectivity_optogenetics(epoch = (s1.delay, s1.response),
                                                                                                           p=0.05, 
                                                                                                           binsize=150, 
                                                                                                           timestep=50)

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

#%% Aggregate FOV view

paths = [
            r'J:\ephys_data\CW49\python\2024_12_11',
            r'J:\ephys_data\CW49\python\2024_12_12',
            r'J:\ephys_data\CW49\python\2024_12_13',
            r'J:\ephys_data\CW49\python\2024_12_14',
            r'J:\ephys_data\CW49\python\2024_12_15',
            r'J:\ephys_data\CW49\python\2024_12_16',
        
        ]

all_control_sel, all_opto_sel_stim_left, all_opto_sel_stim_right = [],[],[]

for path in paths:
    s1 = Session(path, passive=False, side='R')
    s1.good_neurons = [n for n in s1.good_neurons if n in np.where(s1.celltype == 3)[0]]
    control_sel, opto_sel_stim_left, opto_sel_stim_right, time = s1.selectivity_optogenetics(epoch = (s1.delay, s1.response),
                                                                                            p=0.05, 
                                                                                            binsize=150, 
                                                                                            timestep=50,
                                                                                            return_traces=True)
    
    all_control_sel += control_sel
    all_opto_sel_stim_left += opto_sel_stim_left
    all_opto_sel_stim_right += opto_sel_stim_right
# Plot

sel = np.mean(all_control_sel, axis=0)
selo_stimleft = np.mean(all_opto_sel_stim_left, axis=0)
selo_stimright = np.mean(all_opto_sel_stim_right, axis=0)

err = np.std(control_sel, axis=0) / np.sqrt(len(all_control_sel))
erro_stimleft = np.std(opto_sel_stim_left, axis=0) / np.sqrt(len(all_control_sel))
erro_stimright = np.std(opto_sel_stim_right, axis=0) / np.sqrt(len(all_control_sel))
    
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

plt.suptitle('{} ALM recording ({} neurons)'.format(s1.side, len(all_control_sel)))

plt.show()
    
