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
# import behavior
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

s1.plot_raster_and_PSTH(168, opto=True, stimside='L')


#%% Within session control
p_s=0.01
p=0.01
retained_sample = []
recruited_sample = []
retained_delay = []
recruited_delay = []
dropped_delay = []
dropped_sample = []
alls1list, alld1, allr1, allns1 = [],[],[],[] # s1: susc ns: non susc

learning_SDR = []
expert_SDR = []

for paths in expert_path: # For each mouse/FOV
    ret_s = []
    recr_s = []
    ret_d, recr_d = [],[]
    drop_d, drop_s = [], []
    
    s1list, d1, r1, ns1 = np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)
    
    p_s = 0.01
    s1 = Session(paths, side='R')

    stim_period = (s1.delay+0, s1.delay+1)
    
    
    
    good_non_stim_trials_set = set(s1.i_good_non_stim_trials)
    good_stim_trials_set = set(s1.i_good_stim_trials)
        
    control_trials_left = np.random.permutation([t for t in s1.L_trials if t in good_non_stim_trials_set])
    pert_trials_left = np.random.permutation([t for t in s1.L_trials if t in good_stim_trials_set])
    
    control_trials_right = np.random.permutation([t for t in s1.R_trials if t in good_non_stim_trials_set])
    pert_trials_right = np.random.permutation([t for t in s1.R_trials if t in good_stim_trials_set])
    
    L_trials_ctl_first, L_trials_ctl_second = control_trials_left[:int(len(control_trials_left) / 2)], control_trials_left[int(len(control_trials_left) / 2):]
    L_trials_opto_first, L_trials_opto_second = pert_trials_left[:int(len(pert_trials_left) / 2)], pert_trials_left[int(len(pert_trials_left) / 2):]
    
    R_trials_ctl_first, R_trials_ctl_second = control_trials_right[:int(len(control_trials_right) / 2)], control_trials_right[int(len(control_trials_right) / 2):]
    R_trials_opto_first, R_trials_opto_second = pert_trials_right[:int(len(pert_trials_right) / 2)], pert_trials_right[int(len(pert_trials_right) / 2):]

    naive_sample_sel, tstat = s1.susceptibility(stimside= 'L', period = stim_period, p=p_s, return_n=True,
                                            provide_trials = (cat((L_trials_ctl_first, R_trials_ctl_first)),
                                                              cat((L_trials_opto_first, R_trials_opto_first))))
    
    naive_sample_sel = [naive_sample_sel[n] for n in range(len(naive_sample_sel)) if tstat[n] < 0]
    naive_nonsel = [n for n in s1.good_neurons if n not in naive_sample_sel] # non susc population

    exp_susc, tstat = s1.susceptibility(stimside= 'L', period = stim_period, p=p_s, return_n=True,
                                    provide_trials = (cat((L_trials_ctl_second, R_trials_ctl_second)),
                                                      cat((L_trials_opto_second, R_trials_opto_second))))
    exp_susc = [exp_susc[n] for n in range(len(exp_susc)) if tstat[n] < 0]

    # Get functional group info
    
    
    for n in naive_sample_sel:
        if n in exp_susc:
            s1list[0] += 1
            # ret_s += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]

        else:
            s1list[3] += 1
            # drop_s += [(n, s2.good_neurons[np.where(s1.good_neurons == n)[0][0]])]

    
    for n in naive_nonsel:
        if n in exp_susc:
            ns1[0] += 1
            # recr_s += [(n, s2.good_neurons[np.where(s1.good_neurons ==n)[0][0]])]

        else:
            ns1[3] += 1
    print(s1list)

    s1list, d1, r1, ns1 = s1list / len(s1.good_neurons), d1 / len(s1.good_neurons), r1 / len(s1.good_neurons), ns1 / len(s1.good_neurons)

    alls1list += [s1list]
    alld1 += [d1]
    allr1 += [r1] 
    allns1 += [ns1]
    
    retained_sample += [ret_s]
    recruited_sample += [recr_s]
    dropped_sample += [drop_s]
    retained_delay += [ret_d]
    recruited_delay += [recr_d]
    dropped_delay += [drop_d]

alls1list = np.mean(alls1list, axis=0) 
alld1 = np.mean(alld1, axis=0)
allr1 = np.mean(allr1, axis=0)
allns1 = np.mean(allns1, axis=0)

