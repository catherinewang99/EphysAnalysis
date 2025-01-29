# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:15:56 2024

Analyze ephys behavior


@author: catherinewang
"""

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
# import session
import behavior
import sys
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")

from ephysSession import Session


#%% Plot learning progression
# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW48\python_behavior', behavior_only=True)
# b.learning_progression(window = 50)

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW49\python_behavior', behavior_only=True)
# b.learning_progression(window = 50,  color_background=range(32-6)) # All but the last 6 days

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW53\python_behavior', behavior_only=True)
# b.learning_progression(window = 75, color_background=range(36))

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW59\python_behavior', behavior_only=True)
# b.learning_progression(window = 75)#, color_background=range(31))

b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW61\python_behavior', behavior_only=True)
b.learning_progression(window = 75, color_background=range(3))

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW54t\python_behavior', behavior_only=True)
# b.learning_progression(window = 75)#, color_background=range(31))

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW54\python_behavior', behavior_only=True)
# b.learning_progression(window = 50, color_background=range(23))

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW57\python_behavior', behavior_only=True)
# b.learning_progression(window = 50, color_background=range(20))



#%% Plot behavior effect to stim

paths = [
            # r'J:\ephys_data\CW49\python\2024_12_11',
            # r'J:\ephys_data\CW49\python\2024_12_12',
            r'J:\ephys_data\CW49\python\2024_12_13',
            r'J:\ephys_data\CW49\python\2024_12_14',
            r'J:\ephys_data\CW49\python\2024_12_15',
            r'J:\ephys_data\CW49\python\2024_12_16',
        
        ]

performance_opto_left, performance_opto_right = [], []
performance_ctl = []

stim_left_performance_opto_left, stim_left_performance_opto_right = [], []
stim_right_performance_opto_left, stim_right_performance_opto_right = [], []
performance_ctl_right, performance_ctl_left = [], []

fig = plt.figure()

for path in paths:
    s1 = Session(path, passive=False)
    all_stim_trials = np.where(s1.stim_ON)[0]
    left_stim_trials = [i for i in np.where(s1.stim_side == 'L')[0] if i in all_stim_trials]
    right_stim_trials = [i for i in np.where(s1.stim_side == 'R')[0] if i in all_stim_trials]
    control_trials = np.where(~s1.stim_ON)[0]
    
    perf_right, perf_left, perf_all = s1.performance_in_trials(left_stim_trials)
    performance_opto_left += [perf_all]
    stim_left_performance_opto_left += [perf_left]
    stim_left_performance_opto_right += [perf_right]
    
    
    perf_rightctl, perf_leftctl, perf_all = s1.performance_in_trials(control_trials)
    performance_ctl += [perf_all]
    performance_ctl_right += [perf_rightctl]
    performance_ctl_left += [perf_leftctl]
    
    plt.plot([0 - 0.2, 0 + 0.2], [perf_rightctl, perf_right], color='blue', alpha=0.3)
    plt.plot([0 - 0.2, 0 + 0.2], [perf_leftctl, perf_left], color='red', alpha=0.3)
    # plt.scatter(0 - 0.2, perf_rightctl, c='b', marker='o')
    # plt.scatter(0 - 0.2, perf_leftctl, c='r', marker='o')
    # plt.scatter(0 - 0.2, perf_all, facecolors='white', edgecolors='black')
    # plt.scatter(0 + 0.2, perf_right, c='b', marker='o')
    # plt.scatter(0 + 0.2, perf_left, c='r', marker='o')
    
    perf_right, perf_left, perf_all = s1.performance_in_trials(right_stim_trials)
    performance_opto_right += [perf_all]
    stim_right_performance_opto_left += [perf_left]
    stim_right_performance_opto_right += [perf_right]
    
    plt.plot([1 - 0.2, 1 + 0.2], [perf_rightctl, perf_right], color='blue', alpha=0.3)
    plt.plot([1 - 0.2, 1 + 0.2], [perf_leftctl, perf_left], color='red', alpha=0.3)
    # plt.scatter(1 - 0.2, perf_rightctl, c='b', marker='o')
    # plt.scatter(1 - 0.2, perf_leftctl, c='r', marker='o')
    # plt.scatter(0 - 0.2, perf_all, facecolors='white', edgecolors='black')
    # plt.scatter(1 + 0.2, perf_right, c='b', marker='o')
    # plt.scatter(1 + 0.2, perf_left, c='r', marker='o')



plt.errorbar([-0.2, 0.2], [np.mean(performance_ctl_right), np.mean(stim_left_performance_opto_right)], 
             yerr = [np.std(performance_ctl_right), np.std(stim_left_performance_opto_right)], marker = 'o', color='blue')
plt.errorbar([-0.2, 0.2], [np.mean(performance_ctl_left), np.mean(stim_left_performance_opto_left)], 
             yerr = [np.std(performance_ctl_left), np.std(stim_left_performance_opto_left)], marker = 'o', color='red')

plt.errorbar([1-0.2, 1+0.2], [np.mean(performance_ctl_right), np.mean(stim_right_performance_opto_right)], 
             yerr = [np.std(performance_ctl_right), np.std(stim_right_performance_opto_right)], marker = 'o', color='blue')
plt.errorbar([1-0.2, 1+0.2], [np.mean(performance_ctl_left), np.mean(stim_right_performance_opto_left)],
             yerr = [np.std(performance_ctl_left), np.std(stim_right_performance_opto_left)], marker = 'o', color='red')

plt.xticks([0,1],['Left stim', 'Right stim'])
plt.ylabel('Performance')
plt.show()

#%% No left right info
fig = plt.figure()

plt.scatter(np.ones(len(performance_ctl)) * 0.2, performance_opto_left, c='b', marker='x', label="Perturbation trials")
plt.scatter(np.ones(len(performance_ctl)) * -0.2, performance_ctl, c='b', marker='o', label="Control trials")


plt.bar([0.2], np.mean(performance_opto_left), 0.4, fill=False)
plt.bar([-0.2], np.mean(performance_ctl), 0.4, fill=False)

plt.scatter(np.ones(len(performance_ctl)) * 1.2, performance_opto_right, c='b', marker='x', label="Perturbation trials")
plt.scatter(np.ones(len(performance_ctl)) * 0.8, performance_ctl, c='b', marker='o', label="Control trials")

plt.bar([1.2], np.mean(performance_opto_right), 0.4, fill=False)
plt.bar([0.8], np.mean(performance_ctl), 0.4, fill=False)

for i in range(len(performance_ctl)):
    
    plt.plot([-0.2, 0.2], [performance_ctl[i], performance_opto_left[i]], color='grey')
    plt.plot([0.8, 1.2], [performance_ctl[i], performance_opto_right[i]], color='grey')


plt.xticks([0,1],['Left stim', 'Right stim'])

plt.show()

#%% Learning speed comparison to normal learning

# Number of trials to reach 70% at <1.3s

delay_length = 1.3
performance = 0.75
window = 25

imaging_paths = [
    
    r'F:\data\Behavior data\BAYLORCW032\python_behavior',
    r'F:\data\Behavior data\BAYLORCW034\python_behavior',
    r'F:\data\Behavior data\BAYLORCW035\python_behavior',
    r'F:\data\Behavior data\BAYLORCW036\python_behavior',
    r'H:\data\Behavior data\BAYLORCW044\python_behavior',
    r'H:\data\Behavior data\BAYLORCW046\python_behavior',
    
    ]
ephys_paths = [
    
    r'J:\ephys_data\Behavior data\CW49\python_behavior',
    r'J:\ephys_data\Behavior data\CW53\python_behavior',
    r'J:\ephys_data\Behavior data\CW54\python_behavior',

    
    ]

imaging_trials = []
for path in imaging_paths:
    b = behavior.Behavior(path, behavior_only=True)
    imaging_trials += [b.time_to_reach_perf(performance, delay_length, window=window)]


ephys_trials = []
for path in ephys_paths:
    b = behavior.Behavior(path, behavior_only=True)
    ephys_trials += [b.time_to_reach_perf(performance, delay_length, window=window)]


f=plt.figure()

plt.bar([0,1],[np.mean(imaging_trials), np.mean(ephys_trials)])
plt.scatter(np.zeros(len(imaging_trials)), imaging_trials)
plt.scatter(np.ones(len(ephys_trials)), ephys_trials)
plt.xticks([0,1],['Regular learning', 'Corr. learning'])
plt.ylabel('Number of trials')
plt.title('Time to reach {}% performance at <{}s delay'.format(performance*100, delay_length))












