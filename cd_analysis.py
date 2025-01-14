# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:15:24 2025

@author: catherinewang
"""


import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from session import Session
from activitymode import Mode
# import activityMode
from matplotlib.pyplot import figure
# import decon
from scipy.stats import chisquare
import pandas as pd
plt.rcParams['pdf.fonttype'] = '42' 
import random
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

## Paths

path = r'J:\ephys_data\CW49\python\2024_12_13'



paths = [
            # r'J:\ephys_data\CW49\python\2024_12_11',
            # r'J:\ephys_data\CW49\python\2024_12_12',
            r'J:\ephys_data\CW49\python\2024_12_13',
            r'J:\ephys_data\CW49\python\2024_12_14',
            r'J:\ephys_data\CW49\python\2024_12_15',
            r'J:\ephys_data\CW49\python\2024_12_16',
        
        ]

s1 = Mode(path, side='R')#, timestep=1)#, passive=False)
# s1.plot_CD(mode_input='stimulus')
s1.plot_CD_opto()

#%% Aggregate over FOVs

r_traces, l_traces = [],[]
for path in paths:
    
    s1 = Mode(path, side='L')#, timestep=1)#, passive=False)

    r, l = s1.plot_CD(mode_input='stimulus', return_traces = True)
    
    period = np.where((s1.t > s1.sample) & (s1.t < s1.delay))[0] # Sample period

    if np.mean(r[period]) < np.mean(l[period]):
        
        r_traces += [r]
        l_traces += [l]

    else:

        r_traces += [-r]
        l_traces += [-l]
        
        
# Plot
x = s1.t
plt.plot(x, np.mean(r_traces, axis=0), 'b', linewidth = 2)
plt.plot(x, np.mean(l_traces, axis=0), 'r', linewidth = 2)

plt.fill_between(x, np.mean(l_traces, axis=0) - stats.sem(l_traces, axis=0), 
          np.mean(l_traces, axis=0) + stats.sem(l_traces, axis=0),
          color=['#ffaeb1'])

plt.fill_between(x, np.mean(r_traces, axis=0) - stats.sem(r_traces, axis=0), 
          np.mean(r_traces, axis=0) + stats.sem(r_traces, axis=0),
          color=['#b4b2dc'])
        
plt.axvline(s1.sample, ls='--', color='grey')
plt.axvline(s1.delay, ls='--', color='grey')
plt.axvline(s1.response, ls='--', color='grey')
        
#%% Recovery to stim









        