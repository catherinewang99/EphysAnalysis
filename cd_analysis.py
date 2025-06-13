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
from ephysSession import Session
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


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cos_sim(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

#%% Paths

path = r'J:\ephys_data\CW49\python\2024_12_13'



paths = [
            # r'J:\ephys_data\CW49\python\2024_12_11',
            # r'J:\ephys_data\CW49\python\2024_12_12',
            r'J:\ephys_data\CW49\python\2024_12_13',
            r'J:\ephys_data\CW49\python\2024_12_14',
            r'J:\ephys_data\CW49\python\2024_12_15',
            r'J:\ephys_data\CW49\python\2024_12_16',
        
        ]
path =  r'G:\ephys_data\CW59\python\2025_02_25'

s1 = Mode(path, side='R', proportion_train=0.75)#, timestep=1)#, passive=False)
s1.plot_CD(mode_input='choice')
_, _, db, choice = s1.decision_boundary(error=False)
np.mean(choice)
# s1.plot_CD_opto()
# s1.plot_CD_opto(mode_input='stimulus', stim_side = 'L')

#%% Project for  a single trial


#%% CD projection endpoints 
path =  r'G:\ephys_data\CW59\python\2025_02_25'

s1 = Mode(path, side='R')
proj_allDimR, proj_allDimL = s1.plot_CD(mode_input='choice', auto_corr_return=True, single_trial=True)
delay_idx = np.where(s1.t < s1.response)[0][-1]
r_projR = [t[delay_idx] for t in proj_allDimR] 
r_projL = [t[delay_idx] for t in proj_allDimL]

train_test_trials = ([s1.r_train_idx, s1.l_train_idx, s1.r_test_idx, s1.l_test_idx],
                     [s1.r_train_err_idx, s1.l_train_err_idx, s1.r_test_err_idx, s1.l_test_err_idx])

s1 = Mode(path, side='L', train_test_trials=train_test_trials)
proj_allDimR, proj_allDimL = s1.plot_CD(mode_input='choice', auto_corr_return=True, single_trial=True)
l_projR = [t[delay_idx] for t in proj_allDimR] 
l_projL = [t[delay_idx] for t in proj_allDimL]



f=plt.figure()
plt.axhline(0, ls = '--', color='black')
plt.axvline(0, ls = '--', color='black')
plt.scatter(l_projL, r_projL, color='red')
plt.scatter(l_projR, r_projR, color='blue')
plt.xlabel('Left ALM')
plt.ylabel('Right ALM')
plt.title('CD projections (a.u.)')

#%% Over all FOV get rank correlation CD proj
left_trial, right_trial, both = [],[],[]
for path in cat(all_expert_paths):
    
    ## this section shuffles trials, making sure that no unstable trials are used
    s1 = Session(path)
    all_i_good = s1.i_good_trials
    
    # shuffle trials externally
    r_trials = [i for i in s1.i_good_non_stim_trials if not s1.early_lick[i] and s1.R_correct[i]]
    l_trials = [i for i in s1.i_good_non_stim_trials if not s1.early_lick[i] and s1.L_correct[i]]
    np.random.shuffle(r_trials) # shuffle the items in 
    np.random.shuffle(l_trials) # shuffle the items in 
    numr, numl = len(r_trials), len(l_trials)
    
    r_train_idx, l_train_idx = r_trials[:int(numr*0.5)], l_trials[:int(numl*0.5)]
    r_test_idx, l_test_idx = r_trials[int(numr*0.5):], l_trials[int(numl*0.5):]
                
    r_trials = [i for i in s1.i_good_non_stim_trials if not s1.early_lick[i] and s1.R_wrong[i]]
    l_trials = [i for i in s1.i_good_non_stim_trials if not s1.early_lick[i] and s1.L_wrong[i]]
    np.random.shuffle(r_trials) # shuffle the items in 
    np.random.shuffle(l_trials) # shuffle the items in 
    numr, numl = len(r_trials), len(l_trials)

    r_train_err_idx, l_train_err_idx = r_trials[:int(numr*0.5)], l_trials[:int(numl*0.5)]
    r_test_err_idx, l_test_err_idx = r_trials[int(numr*0.5):], l_trials[int(numl*0.5):]
    
    train_test_trials = ([r_train_idx, l_train_idx, r_test_idx, l_test_idx],
                         [r_train_err_idx, l_train_err_idx, r_test_err_idx, l_test_err_idx])
    
    ## this section starts CD calc

    s1 = Mode(path, side='R', train_test_trials=train_test_trials)
    
    proj_allDimR, proj_allDimL = s1.plot_CD(mode_input='choice', auto_corr_return=True, single_trial=True)
    delay_idx = np.where(s1.t < s1.response)[0][-1]
    r_projR = [t[delay_idx] for t in proj_allDimR] 
    r_projL = [t[delay_idx] for t in proj_allDimL]

    
    s1 = Mode(path, side='L', train_test_trials=train_test_trials)
    proj_allDimR, proj_allDimL = s1.plot_CD(mode_input='choice', auto_corr_return=True, single_trial=True)
    l_projR = [t[delay_idx] for t in proj_allDimR] 
    l_projL = [t[delay_idx] for t in proj_allDimL]
    
    res = stats.spearmanr(l_projL, r_projL)
    left_trial += [res.statistic]
    
    res = stats.spearmanr(l_projR, r_projR)
    right_trial += [res.statistic]
    
    res = stats.spearmanr(cat((l_projL, l_projR)), cat((r_projL, r_projR)))
    both += [res.statistic]
    
f=plt.figure(figsize=(7,8))
plt.bar(range(3), [np.mean(right_trial), np.mean(left_trial), np.mean(both)], fill=False) 
plt.scatter(np.zeros(len(right_trial)), right_trial, color='red')
plt.scatter(np.ones(len(left_trial)), left_trial, color='blue')
plt.scatter(np.ones(len(both))*2, both, color='grey')
plt.xticks(range(3), ['Lick right', 'Lick left', 'Both'])
plt.axhline(0, color='black')
plt.ylabel('Rank correlation of CD proj.')

#%% Decoding accuracies
l_acc, r_acc = [],[]
for path in cat(all_expert_paths):
    s1 = Mode(path, side='L', proportion_train=0.65)#, timestep=1)#, passive=False)

    if len(s1.l_test_idx) > 5 and len(s1.r_test_idx) > 5 and len(s1.good_neurons) > 5:
        
        _, _, db, choice = s1.decision_boundary(error=False)
    
        l_acc += [np.mean(choice)]

    s1 = Mode(path, side='R', proportion_train=0.65)# timestep=1)#, passive=False)
    
    if len(s1.l_test_idx) > 5 and len(s1.r_test_idx) > 5 and len(s1.good_neurons) > 5:

        _, _, db, choice = s1.decision_boundary(error=False)
    
        r_acc += [np.mean(choice)]

# l_acc= [l if l > 0.5 else 1-l for l in l_acc ]
# r_acc= [l if l > 0.5 else 1-l for l in r_acc ]

f=plt.figure()
plt.bar([0,1], [np.mean(l_acc), np.mean(r_acc)], 0.4, fill=False)
plt.scatter(np.zeros(len(l_acc)), l_acc)
plt.scatter(np.ones(len(r_acc)), r_acc)
plt.xticks([0,1],['left ALM','right ALM'])
plt.ylabel('Decoding accuracy')
plt.ylim(0,1)


#%% Aggregate over FOVs

r_traces, l_traces = [],[]
for path in paths:
    
    s1 = Mode(path, side='L')#, timestep=1)#, passive=False)

    r, l = s1.plot_CD(mode_input='stimulus', return_traces = True)
    
    period = np.where((s1.t > s1.sample) & (s1.t < s1.delay))[0] # Sample period
    # period = np.where((s1.t > s1.delay) & (s1.t < s1.response))[0] # Delay period

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
path = r'J:\ephys_data\CW53\python\2025_01_31'

s1 = Mode(path, side='R')#, timestep=1)#, passive=False)
s1.plot_CD_opto(stim_side = 'L')

s1 = Mode(path, side='L')#, timestep=1)#, passive=False)
s1.plot_CD_opto(stim_side = 'L')


#%% Variable or multiple CDs?
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
# Correlation of CDs calculated from non overlapping trials?
naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_31',
                    r'H:\data\BAYLORCW046\python\2024_06_11',
                  r'H:\data\BAYLORCW046\python\2024_06_26',]
# naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW044\python\2024_05_22',
#                                       r'H:\data\BAYLORCW044\python\2024_06_06',
#                                     r'H:\data\BAYLORCW044\python\2024_06_19']
# naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW044\python\2024_05_23',
#                                       r'H:\data\BAYLORCW044\python\2024_06_04',
#                                     r'H:\data\BAYLORCW044\python\2024_06_18']
# naivepath, learningpath, expertpath = [r'H:\data\BAYLORCW046\python\2024_05_29',
#                     r'H:\data\BAYLORCW046\python\2024_06_07',
#                   r'H:\data\BAYLORCW046\python\2024_06_24',]
# naivepath, learningpath, expertpath =[r'H:\data\BAYLORCW046\python\2024_05_30',
#                                       r'H:\data\BAYLORCW046\python\2024_06_10',
#                                       r'H:\data\BAYLORCW046\python\2024_06_27']
split = 1/2
splitnum = int(1/split)
ctl=True
save=False
vmin=0 
vmax=0.8
side='R'

##LEARNING



# for path in allpaths[0]:
    
#     for _ in range(10):

        # s1 = Mode(path, side='R')#, timestep=1)#, passive=False)
        # CD_choice, _ = s1.plot_CD()







l1 = Mode(path, side=side)#, timestep=1)#, passive=False)

numr = sum([l1.R_correct[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numl = sum([l1.L_correct[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numr = numr-numr%splitnum if numr%splitnum else numr
numl = numl-numl%splitnum if numl%splitnum else numl
r_trials = np.random.permutation(numr) # shuffle the indices
l_trials = np.random.permutation(numl)
numr_err = sum([l1.R_wrong[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numl_err = sum([l1.L_wrong[i] for i in l1.i_good_non_stim_trials if not l1.early_lick[i]])
numr_err = numr_err-numr_err%splitnum if numr_err%splitnum else numr_err
numl_err = numl_err-numl_err%splitnum if numl_err%splitnum else numl_err
r_trials_err = np.random.permutation(numr_err) # shuffle the indices
l_trials_err = np.random.permutation(numl_err)

# First half
r_train_idx, l_train_idx = r_trials[:int(split * numr)], l_trials[:int(split * numl)] #Take a portion of the trials for train
r_test_idx, l_test_idx = r_trials[int(split * numr):int(split * 2 * numr)], l_trials[int(split * numl):int(split * 2 * numl)]

r_train_err_idx, l_train_err_idx = r_trials_err[:int(split* numr_err)], l_trials_err[:int(split * numl_err)]
r_test_err_idx, l_test_err_idx = r_trials_err[int(split* numr_err):int(split*2* numr_err)], l_trials_err[int(split * numl_err):int(split * 2 * numl_err)]

train_test_trials = (r_train_idx, l_train_idx, r_test_idx, l_test_idx)
train_test_trials_err = (r_train_err_idx, l_train_err_idx, r_test_err_idx, l_test_err_idx)

l1 = Mode(path,  side=side, 
          train_test_trials = [train_test_trials, train_test_trials_err])
projR, projL = l1.plot_CD(mode_input = 'choice', plot=False, auto_corr_return=True, ctl=ctl)
CD_choice, _ = l1.plot_CD(mode_input = 'choice', plot=False, auto_corr_return=False, ctl=ctl)
if save:
    l1.plot_CD(ctl=ctl, save=r'H:\Fig 5\CDchoice_20perc_train_test_learning_run1.pdf')
else:
    l1.plot_CD(ctl=ctl)


#second half
r_train_idx, l_train_idx = r_trials[int((1-split) * numr):], l_trials[int((1-split) * numl):] #Take a portion of the trials for train
r_test_idx, l_test_idx = r_trials[int((1-(2*split)) * numr):int((1-split) * numr)], l_trials[int((1-(2*split)) * numl):int((1-split) * numl)]

r_train_err_idx, l_train_err_idx = r_trials_err[int((1-split) * numr_err):], l_trials_err[int((1-split) * numl_err):]
r_test_err_idx, l_test_err_idx = r_trials_err[int((1-(2*split)) * numr_err):int((1-split) * numr_err)], l_trials_err[int((1-(2*split)) * numl_err):int((1-split) * numl_err)]

train_test_trials = (r_train_idx, l_train_idx, r_test_idx, l_test_idx)
train_test_trials_err = (r_train_err_idx, l_train_err_idx, r_test_err_idx, l_test_err_idx)

l1 = Mode(path,  side=side, 
          train_test_trials = [train_test_trials, train_test_trials_err])
projR1, projL1 = l1.plot_CD(mode_input = 'choice', plot=False, auto_corr_return=True, ctl=ctl)
CD_choice1, _ = l1.plot_CD(mode_input = 'choice', plot=False, auto_corr_return=False, ctl=ctl)
if save:
    l1.plot_CD(ctl=ctl, save=r'H:\Fig 5\CDchoice_20perc_train_test_learning_run2.pdf')
else:
    l1.plot_CD(ctl=ctl)

#%% Plot

cos_sim(CD_choice, CD_choice1)

#Plot the autocorrelogram
f = plt.figure(figsize=(5,5))
allproj = np.vstack((projR, projL[:, 0]))
allproj1 = np.vstack((projR1, projL1[:, 0]))
corrs = np.corrcoef(allproj, allproj1, rowvar=False)
# corrs = corrs[:l1.time_cutoff, :l1.time_cutoff]
plt.imshow(corrs, vmin=vmin, vmax=vmax)
plt.axhline(l1.sample, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.sample, color = 'white', ls='--', linewidth = 0.5)

plt.axhline(l1.delay, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.delay, color = 'white', ls='--', linewidth = 0.5)

plt.axhline(l1.response, color = 'white', ls='--', linewidth = 0.5)
plt.axvline(l1.response, color = 'white', ls='--', linewidth = 0.5)

plt.xticks([l1.sample, l1.delay, l1.response], [-4.3, -3, 0])    
plt.yticks([l1.sample, l1.delay, l1.response], [-4.3, -3, 0])    
plt.colorbar()
if save:
    plt.savefig(r'H:\Fig 5\CDchoice_20perc_train_test_learning_CORR.pdf')
plt.show()




        