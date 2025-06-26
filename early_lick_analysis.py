# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:36:06 2025

@author: catherinewang
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:10:18 2025

Investigate if there is an early lick cd that predicts the final cd over learning

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Imaging analysis")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# import decon
from scipy.stats import chisquare
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from activitymode import Mode

from ephysSession import Session
import behavior
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 

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
#%% paths


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


#%% early lick behavior analysis

path = r'G:\ephys_data\CW59\python\2025_02_26'

s1 = Session(path, passive=False)

# Vanilla raster plot
f=plt.figure(figsize=(5,5))
for i in range(len(s1.early_lick_time)):
    plt.scatter(s1.early_lick_time[i][:,0], np.ones(s1.early_lick_time[i].shape[0]) * i, color='grey')
plt.axvline(s1.sample, ls = '--', color = 'black')
plt.axvline(s1.delay, ls = '--', color = 'black')
plt.axvline(s1.response, ls = '--', color = 'black')
plt.ylabel('Trials')
plt.xlabel('Time (s)')
plt.title('Lick raster for early lick trials')

# plot with direction information (L/r)
f=plt.figure(figsize=(5,5))
for i in range(len(s1.early_lick_time)):
    for j in range(s1.early_lick_time[i].shape[0]):
        if s1.early_lick_side[i][j] == 'l':
            plt.scatter(s1.early_lick_time[i][j,0], i, color='red')
        else:
            plt.scatter(s1.early_lick_time[i][j,0], i, color='blue')

plt.axvline(s1.sample, ls = '--', color = 'black')
plt.axvline(s1.delay, ls = '--', color = 'black')
plt.axvline(s1.response, ls = '--', color = 'black')
plt.ylabel('Trials')
plt.xlabel('Time (s)')
plt.title('Lick raster for early lick trials')

# Plot with direciton info but correct error
instr_trials = s1.instructed_direction()
all_instr_dir = np.array(instr_trials)[np.where(s1.early_lick)[0]]

f=plt.figure(figsize=(5,5))
for i in range(len(s1.early_lick_time)):
    instr_dir = all_instr_dir[i]
    for j in range(s1.early_lick_time[i].shape[0]):
        if s1.early_lick_side[i][j] == instr_dir:
            plt.scatter(s1.early_lick_time[i][j,0], i, color='green')
        else:
            plt.scatter(s1.early_lick_time[i][j,0], i, color='red')

plt.axvline(s1.sample, ls = '--', color = 'black')
plt.axvline(s1.delay, ls = '--', color = 'black')
plt.axvline(s1.response, ls = '--', color = 'black')
plt.ylabel('Trials')
plt.xlabel('Time (s)')
plt.title('Lick raster for early lick trials')

#%% Plot as a histogram over time

paths = [
    r'J:\ephys_data\CW53\python\2025_01_27',
    r'J:\ephys_data\CW53\python\2025_01_28',
    r'J:\ephys_data\CW53\python\2025_01_29',
    # r'J:\ephys_data\CW53\python\2025_01_30',
    r'J:\ephys_data\CW53\python\2025_02_01',
    r'J:\ephys_data\CW53\python\2025_02_02',
      ]
# paths = [r'G:\ephys_data\CW59\python\2025_02_22',
#  r'G:\ephys_data\CW59\python\2025_02_24',
#  r'G:\ephys_data\CW59\python\2025_02_25',
#  r'G:\ephys_data\CW59\python\2025_02_26',
#  r'G:\ephys_data\CW59\python\2025_02_28',
#  ]
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
binsize = 100 # ms
buckets = np.arange(0, s1.response+binsize/1000, binsize/1000)
f = plt.figure(figsize=(8,6))
counter = 0
for path in paths:
    s1 = Session(path, passive=False)

    early_lick_count = []
    for i in range(len(s1.early_lick_time)): # every early lick trial
    
        counts, bin_edges = np.histogram(s1.early_lick_time[i][:,0], bins=buckets)
        early_lick_count += [counts]

    plt.plot(bin_edges[:-1], np.sum(early_lick_count, axis=0), label=path, alpha=0.7, color=colors[counter])
    counter += 1
plt.axvline(s1.sample, ls = '--', color = 'black')
plt.axvline(s1.delay, ls = '--', color = 'black')
plt.axvline(s1.response, ls = '--', color = 'black')
plt.ylabel('Early lick count')
plt.xlabel('Time (s)')
plt.legend()



#%% # Project regular CD choice onto early lick trials


path =  r'G:\ephys_data\CW59\python\2025_02_26'
s1 = Mode(path, side='R')

cd_choice, _ = s1.plot_CD(mode_input='choice')

early_lick_trials = np.where(s1.early_lick)[0]
early_lick_left, early_lick_right = [],[] #collect idx of el sides
for i in range(len(s1.early_lick_side)):
    if s1.early_lick_side[i][0] == 'l':
        early_lick_left += [i]
    else:
        early_lick_right += [i]
early_lick_left = [i for i in early_lick_left if i in s1.i_good_trials]
early_lick_right = [i for i in early_lick_right if i in s1.i_good_trials]
time = s1.t

# single trial view
f=plt.figure()

left_proj = []
for t in early_lick_left:
    time_adj = time[np.where(time<s1.early_lick_time[t][0][0])]
    if len(time_adj) == 0:
        continue
    
    trial = early_lick_trials[t]
    activity, _, _ = s1.get_PSTH_multiple(s1.good_neurons, [trial], binsize=s1.binsize, timestep = s1.timestep)

    # activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
    proj_allDim = np.dot(activity.T, cd_choice)
    # proj_allDimR += [proj_allDim]
    left_proj += [proj_allDim[:len(time_adj)]]
    plt.plot(time_adj - time_adj[-1], proj_allDim[:len(time_adj)], 'r', alpha = 0.5,  linewidth = 0.5)


right_proj=[]
for t in early_lick_right:
    time_adj = time[np.where(time<s1.early_lick_time[t][0][0])]
    if len(time_adj) == 0:
       continue
    
    trial = early_lick_trials[t]
    activity, _, _ = s1.get_PSTH_multiple(s1.good_neurons, [trial], binsize=s1.binsize, timestep = s1.timestep)

    # activity = activity - np.tile(np.mean(activityRL_train, axis=1)[:, None], (1, activity.shape[1]))
    proj_allDim = np.dot(activity.T, cd_choice)
    # proj_allDimR += [proj_allDim]
    right_proj += [proj_allDim[:len(time_adj)]]
    plt.plot(time_adj - time_adj[-1], proj_allDim[:len(time_adj)], 'b', alpha = 0.5,  linewidth = 0.5)
    
plt.xlabel('Time before lick (s)')
plt.ylabel('CD choice projections')
plt.show()
# population average view
#average if there an item
max_len = max(len(lst) for lst in left_proj)
time_left = time[:max_len] - time[max_len]
avg_left,std_left = [], []
for idx in range(1,max_len+1):
    
    vals = [i[-idx, 0] for i in left_proj if len(i)>idx]
    avg_left += [np.mean(vals)]
    std_left += [np.std(vals) / np.sqrt(len(vals))]
avg_left.reverse()
std_left.reverse()
max_len = max(len(lst) for lst in right_proj)
time_right = time[:max_len] - time[max_len]
avg_right,std_right = [],[]
for idx in range(1,max_len+1):
    
    vals = [i[-idx, 0] for i in right_proj if len(i)>idx]
    avg_right += [np.mean(vals)]
    std_right += [np.std(vals) / np.sqrt(len(vals))]

avg_right.reverse()
std_right.reverse()

f = plt.figure()
plt.plot(time_right, avg_right, color='b')
plt.plot(time_left, avg_left, color='r')
plt.fill_between(time_left, np.array(avg_left) - np.array(std_left), 
         np.array(avg_left) + np.array(std_left),
         color=['#ffaeb1'])
plt.fill_between(time_right, np.array(avg_right) - np.array(std_right), 
         np.array(avg_right) + np.array(std_right),
         color=['#b4b2dc'])

plt.xlabel('Time before lick (s)')
plt.ylabel('CD choice projections')
plt.show()
#%% # calculate early lick cd

path =  r'G:\ephys_data\CW59\python\2025_02_28'
s1 = Mode(path, side='L')

cd_choice, _ = s1.plot_CD(mode_input='choice')

early_lick_trials = np.where(s1.early_lick)[0]
early_lick_left, early_lick_right = [],[] #collect idx of el sides
for i in range(len(s1.early_lick_side)):
    if s1.early_lick_side[i][0] == 'l':
        early_lick_left += [i]
    else:
        early_lick_right += [i]
early_lick_left = [i for i in early_lick_left if i in s1.i_good_trials]
early_lick_right = [i for i in early_lick_right if i in s1.i_good_trials]
time = s1.t

#%% IMAGING: Work with early lick info

naivepath, learningpath, expertpath = [
            r'H:\data\BAYLORCW046\python\2024_05_29',
             r'H:\data\BAYLORCW046\python\2024_06_24',
             r'H:\data\BAYLORCW046\python\2024_06_28'
             ]

naivepath, learningpath, expertpath = [
            r'H:\data\BAYLORCW046\python\2024_05_30',
              r'H:\data\BAYLORCW046\python\2024_06_10',
              r'H:\data\BAYLORCW046\python\2024_06_27'
              ]



naivepath, learningpath, expertpath = [
            r'H:\data\BAYLORCW044\python\2024_05_22',
            r'H:\data\BAYLORCW044\python\2024_06_06',
            r'H:\data\BAYLORCW044\python\2024_06_19',
              ]

# naivepath, learningpath, expertpath = [
#             r'F:\data\BAYLORCW032\python\2023_10_05',
#             r'F:\data\BAYLORCW032\python\2023_10_19',
#             r'F:\data\BAYLORCW032\python\2023_10_24',
#               ]




allnaivepaths = [r'F:\data\BAYLORCW032\python\2023_10_05',
            # r'F:\data\BAYLORCW034\python\2023_10_12',
            r'F:\data\BAYLORCW036\python\2023_10_09',
            r'F:\data\BAYLORCW035\python\2023_10_26',
            r'F:\data\BAYLORCW037\python\2023_11_21',
            
            r'H:\data\BAYLORCW044\python\2024_05_22',
            r'H:\data\BAYLORCW044\python\2024_05_23',
            
            r'H:\data\BAYLORCW046\python\2024_05_29',
            r'H:\data\BAYLORCW046\python\2024_05_30',
            r'H:\data\BAYLORCW046\python\2024_05_31',
            ]
for naivepath in allnaivepaths:

    s2 = Mode(naivepath, 
              # use_reg = True, triple=True, 
              baseline_normalization="median_zscore")
    
    # calculate early lick CD in naive sessions - define here
    
    early_lick_trials = np.where(s2.early_lick)[0]
    early_lick_side = [i[0] for i in s2.early_lick_side] # use first lick as direction
    early_lick_time = [i[0][0] for i in s2.early_lick_time]
    
    # Filter out the early lick trials that happen too early (before end of 2.5 ITI)
    drop_idx = [i for i in range(len(early_lick_time)) if early_lick_time[i] > 2.5]
    early_lick_trials = np.array(early_lick_trials)[drop_idx]
    early_lick_side = np.array(early_lick_side)[drop_idx]
    early_lick_time = np.array(early_lick_time)[drop_idx]
    
    # project early lick CD
    r_trials = early_lick_trials[np.where(early_lick_side == 'r')[0]]
    l_trials = early_lick_trials[np.where(early_lick_side == 'l')[0]]
    
    r_trial_lick = early_lick_time[np.where(early_lick_side == 'r')[0]]
    l_trial_lick = early_lick_time[np.where(early_lick_side == 'l')[0]]
    # Get the frame where the early lick happened, per trial
    r_trial_lick = [np.round(r/s2.fs) for r in r_trial_lick] 
    l_trial_lick = [np.round(r/s2.fs) for r in l_trial_lick] 
    
    # Align every trial to shortest early lick (new go cue time)
    r_cue, l_cue = min(r_trial_lick), min(l_trial_lick)
    r_end, l_end = max(r_trial_lick), max(l_trial_lick)
    
    
    # get the train vs test indices
    all_r_idx, all_l_idx = np.arange(len(r_trials)), np.arange(len(l_trials))
    rtmp = np.random.permutation(len(all_r_idx))
    ltmp = np.random.permutation(len(all_l_idx))
    train_r_idx, test_r_idx = rtmp[:int(len(rtmp)/2)], rtmp[int(len(rtmp)/2):]
    train_l_idx, test_l_idx = ltmp[:int(len(ltmp)/2)], ltmp[int(len(ltmp)/2):]
    
    # compile train vs test psths
    PSTH_yes_correct, PSTH_no_correct = [], []
    PSTH_yes_correct_test, PSTH_no_correct_test = [], []
    PSTH_yes_correct_test_mean, PSTH_no_correct_test_mean = [], []
    for n in s2.good_neurons:
        
        r, l = s2.get_trace_matrix(n, rtrials=r_trials, ltrials=l_trials)
        
        for idx in range(len(r)): # adjust each trial to new go cue
            tmp = r[idx][int(r_trial_lick[idx] - r_cue) : ]
            r[idx] = tmp[ : int(s2.time_cutoff - (r_end-r_cue))]
        for idx in range(len(l)): # adjust each trial to new go cue
            tmp = l[idx][int(l_trial_lick[idx] - l_cue) : ]
            l[idx] = tmp[ : int(s2.time_cutoff - (l_end-l_cue))]
            
        r,l = np.array(r), np.array(l)
        
        r_train = np.mean(r[train_r_idx], axis=0)
        l_train = np.mean(l[train_l_idx], axis=0)
    
        r_test_mean = np.mean(r[test_r_idx], axis=0)
        l_test_mean = np.mean(l[test_l_idx], axis=0)
        
        r_test = r[test_r_idx]
        l_test = l[test_l_idx]
        PSTH_yes_correct_test += [r_test]
        PSTH_no_correct_test += [l_test]   
        
        if len(PSTH_yes_correct) == 0:
            PSTH_yes_correct = np.reshape(r_train, (1,-1))
            PSTH_no_correct = np.reshape(l_train, (1,-1))
            PSTH_yes_correct_test_mean = np.reshape(r_test_mean, (1,-1))
            PSTH_no_correct_test_mean = np.reshape(l_test_mean, (1,-1))
            
        else: 
            PSTH_yes_correct = np.concatenate((PSTH_yes_correct, np.reshape(r_train, (1,-1))), axis = 0)
            PSTH_no_correct = np.concatenate((PSTH_no_correct, np.reshape(l_train, (1,-1))), axis = 0)
            PSTH_yes_correct_test_mean = np.concatenate((PSTH_yes_correct_test_mean, np.reshape(r_test_mean, (1,-1))), axis = 0)
            PSTH_no_correct_test_mean = np.concatenate((PSTH_no_correct_test_mean, np.reshape(l_test_mean, (1,-1))), axis = 0)
    
    PSTH_no_correct_test, PSTH_yes_correct_test = np.array(PSTH_no_correct_test), np.array(PSTH_yes_correct_test)
    
    i_t_r = range(int(r_cue) - int(round(0.8*(1/s2.fs))), int(r_cue)+int(round(0.2*(1/s2.fs)))) # use 12 time steps before lick and 3 after lick
    i_t_l = range(int(l_cue) - int(round(0.8*(1/s2.fs))), int(l_cue)+int(round(0.2*(1/s2.fs))))
    
    
    # FIXME : use the correct slicing (don't want to slice neurons)
    wt = (PSTH_yes_correct[:, i_t_r] - PSTH_no_correct[:, i_t_l]) / 2
    CD_choice_mode = np.mean(wt, axis=1)
    
    # project onto other early lick trials - do manually with realigned trials
    # s2.plot_appliedCD(CD_choice_mode, 0)
    x = np.arange(-6.97,4, s2.fs)[:PSTH_yes_correct_test.shape[2]]
    
    proj_allDimR = []
    for t in range(PSTH_yes_correct_test.shape[1]): # every trial
        activity = PSTH_yes_correct_test[:, t, :]
        proj_allDim = np.dot(activity.T, CD_choice_mode)
        proj_allDimR += [proj_allDim[:len(s2.T_cue_aligned_sel)]]
        plt.plot(x, proj_allDim[:len(s2.T_cue_aligned_sel)], 'b', alpha = 0.5,  linewidth = 0.5)
        
    x = np.arange(-6.97,4, s2.fs)[:PSTH_no_correct_test.shape[2]]
    
    proj_allDimL = []
    # for t in range(len(PSTH_no_correct_test)):
    for t in range(PSTH_no_correct_test.shape[1]): # every trial
        activity = PSTH_no_correct_test[:, t, :]
        proj_allDim = np.dot(activity.T, CD_choice_mode)
        proj_allDimL += [proj_allDim[:len(s2.T_cue_aligned_sel)]]
        plt.plot(x, proj_allDim[:len(s2.T_cue_aligned_sel)], 'r', alpha = 0.5, linewidth = 0.5)
    
    activityRL_test = np.concatenate((PSTH_yes_correct_test_mean, PSTH_no_correct_test_mean), axis=1)
    
    # Correct trials
    proj_allDim = np.dot(activityRL_test.T, CD_choice_mode)
    
    # ax = axs.flatten()[0]
    r_cue, l_cue = int(r_cue), int(l_cue)
    plt.plot(np.arange(-6.97,4, s2.fs)[:PSTH_yes_correct_test_mean.shape[1]], proj_allDim[:PSTH_yes_correct_test_mean.shape[1]], 'b', linewidth = 2)
    plt.plot(np.arange(-6.97,4, s2.fs)[:PSTH_no_correct_test_mean.shape[1]], proj_allDim[PSTH_yes_correct_test_mean.shape[1]:], 'r', linewidth = 2)
    plt.scatter([np.arange(-6.97,4, s2.fs)[r_cue]], [proj_allDim[:PSTH_yes_correct_test_mean.shape[1]][r_cue]], color='blue', s=50)
    plt.scatter([np.arange(-6.97,4, s2.fs)[l_cue]], [proj_allDim[PSTH_yes_correct_test_mean.shape[1]:][l_cue]], color='red', s=50)
    plt.title("Applied decoder projections")
    # plt.axvline(-4.3, color = 'grey', alpha=0.5, ls = '--')
    # plt.axvline(-3, color = 'grey', alpha=0.5, ls = '--')
    # plt.axvline(0, color = 'grey', alpha=0.5, ls = '--')
    plt.ylabel('Applied projection (a.u.)')
    
        
    plt.show()

#%%
# project onto learning and expert sessions
s2 = Mode(expertpath, use_reg = True, triple=True, 
          baseline_normalization="median_zscore")
s2.plot_appliedCD(CD_choice_mode, 0)
_, mean, meantrain, meanstd = s2.plot_CD_opto(return_applied=True)
s2.plot_CD_opto_applied(CD_choice_mode, mean, meantrain, meanstd)
orthonormal_basis_learning, _ = s2.plot_CD(plot=False)


s2 = Mode(naivepath, use_reg = True, triple=True, 
          baseline_normalization="median_zscore")
s2.plot_appliedCD(CD_choice_mode, 0)
_, mean, meantrain, meanstd = s2.plot_CD_opto(return_applied=True)
s2.plot_CD_opto_applied(CD_choice_mode, mean, meantrain, meanstd)
orthonormal_basis_expert, _ = s2.plot_CD(plot=False)

# compare angle of early lick CD with final CD in expert session
np.cos(CD_choice_mode, orthonormal_basis_learning)
cosine_similarity(CD_choice_mode, orthonormal_basis_learning)

cos_sim = np.abs(cosine_similarity((CD_choice_mode, orthonormal_basis_learning)))
overall_similarity = np.mean(cos_sim[np.triu_indices_from(cos_sim, k=1)])  # Mean of upper triangle















