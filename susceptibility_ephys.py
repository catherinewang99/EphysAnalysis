# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:05:28 2025
check the number of susceptible neuronsin ephys data

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
from activitymode import Mode

#%% Plot the proportion susceptible, exc vs inh
naive_path = [
            r'H:\ephys_data\CW47\python\2024_10_17',
              # r'H:\ephys_data\CW47\python\2024_10_19',
              r'H:\ephys_data\CW47\python\2024_10_20',
              r'H:\ephys_data\CW47\python\2024_10_21',
              r'H:\ephys_data\CW47\python\2024_10_22',
            r'G:\ephys_data\CW65\python\2025_02_25',
              ]

expert_path = [
            # r'J:\ephys_data\CW49\python\2024_12_11',
            # r'J:\ephys_data\CW49\python\2024_12_12',
            r'J:\ephys_data\CW49\python\2024_12_13',
            r'J:\ephys_data\CW49\python\2024_12_14',
            r'J:\ephys_data\CW49\python\2024_12_15',
            r'J:\ephys_data\CW49\python\2024_12_16',
            
            r'J:\ephys_data\CW53\python\2025_01_29',
            r'J:\ephys_data\CW53\python\2025_01_30',
            r'J:\ephys_data\CW53\python\2025_02_02',
            r'J:\ephys_data\CW53\python\2025_01_27',
            
            r'G:\ephys_data\CW59\python\2025_02_22',
            r'G:\ephys_data\CW59\python\2025_02_24',
            r'G:\ephys_data\CW59\python\2025_02_26',

        ]

all_exp_susc = []
exp_exc, exp_inh = [], []
for path in expert_path:
    s1 = Session(path, side='R')
    
    p_s = 0.0001 / len(s1.good_neurons)
    p_s = 0.01
    stim_period = (s1.delay+0, s1.delay+1)
    
    exp_susc, tstat = s1.susceptibility(stimside= 'L', period = stim_period, p=p_s, return_n=True)
    all_exp_susc += [len(exp_susc) / len(s1.good_neurons)]
    exp_exc += [sum(np.array(tstat) < 0) / len(s1.good_neurons)]
    exp_inh += [sum(np.array(tstat) > 0) / len(s1.good_neurons)]

all_nai_susc = []
nai_exc, nai_inh = [],[]
for path in naive_path:
    s1 = Session(path, side='R')
    
    p_s = 0.0001 / len(s1.good_neurons)
    p_s = 0.01
    stim_period = (s1.delay+0, s1.delay+1)
    
    exp_susc, tstat = s1.susceptibility(stimside= 'L', period = stim_period, p=p_s, return_n=True)
    all_nai_susc += [len(exp_susc) / len(s1.good_neurons)]
    nai_exc += [sum(np.array(tstat) < 0) / len(s1.good_neurons)]
    nai_inh += [sum(np.array(tstat) > 0) / len(s1.good_neurons)]

# PLot

f=plt.figure()

plt.bar(range(2), [np.mean(all_nai_susc), np.mean(all_exp_susc)])
plt.scatter(np.zeros(len(all_nai_susc)), all_nai_susc)
plt.scatter(np.ones(len(all_exp_susc)), all_exp_susc)

plt.ylabel('Proportion of susc neurons')
plt.xticks([0,1], ['Naive', 'Expert'])
plt.title('Proportion of susceptible neurons over learning')
# plt.ylim(bottom=0.9)
plt.show()


# Plot excited vs inhibited
f=plt.figure()

plt.bar(np.arange(2)-0.2, [np.mean(nai_exc), np.mean(exp_exc)], 0.4, label='Excited')
plt.scatter(np.zeros(len(nai_exc))-0.2, nai_exc)
plt.scatter(np.ones(len(exp_exc))-0.2, exp_exc)

plt.bar(np.arange(2)+0.2, [np.mean(nai_inh), np.mean(exp_inh)],0.4, label='Inhibited')
plt.scatter(np.zeros(len(nai_inh))+0.2, nai_inh)
plt.scatter(np.ones(len(exp_inh))+0.2, exp_inh)


plt.ylabel('Proportion of susc neurons')
plt.xticks([0,1], ['Naive', 'Expert'])
plt.title('Proportion of susceptible neurons over learning')
plt.axhline(0.5, ls = '--', color='grey')
# plt.ylim(bottom=0.9)
plt.legend()
plt.show()

#%% LOok at single neuron PSTH


