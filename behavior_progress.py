# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:15:56 2024

Analyze ephys behavior


@author: catherinewang
"""
import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

import behavior
import sys
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 

from ephysSession import Session
#%% Paths

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


#%% Plot learning progression

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW48\python_behavior', behavior_only=True)
# b.learning_progression(window = 50)

b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW49\python_behavior', behavior_only=True)
b.learning_progression(window = 50,  color_background=range(15)) # All but the last 6 days

b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW53\python_behavior', behavior_only=True)
b.learning_progression(window = 75, color_background=range(36))
# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW49t\python_behavior', behavior_only=True)
# b.learning_progression(window = 75, early_lick_ylim=False)

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW53t\python_behavior', behavior_only=True)
# b.learning_progression(window = 75, early_lick_ylim=False)

b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW59\python_behavior', behavior_only=True)
b.learning_progression(window = 75, color_background=range(22))

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW59t\python_behavior', behavior_only=True)
# b.learning_progression(window = 75, early_lick_ylim=False)#, color_background=range(22))

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW61\python_behavior', behavior_only=True)
# b.learning_progression(window = 75, color_background=range(36))

b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW57\python_behavior', behavior_only=True)
b.learning_progression(window = 75, color_background=range(33))

b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW60\python_behavior', behavior_only=True)
b.learning_progression(window = 75, color_background=range(48))

b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW62\python_behavior', behavior_only=True)
b.learning_progression(window = 75, color_background=range(59))

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW63\python_behavior', behavior_only=True)
# b.learning_progression(window = 75, color_background=range(17))

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW54t\python_behavior', behavior_only=True)
# b.learning_progression(window = 75)#, color_background=range(31))

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW54\python_behavior', behavior_only=True)
# b.learning_progression(window = 50, color_background=range(25))

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW62\python_behavior', behavior_only=True)
# b.learning_progression(window = 50, color_background=range(4))

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW60\python_behavior', behavior_only=True)
# b.learning_progression(window = 75)#, color_background=range(3))

#%% Plot learning on one graph to compare learning speeds
mice = ['CW49', 'CW53', 'CW59', 'CW57', 'CW60', 'CW62']
cutoffs = [15,36,22,33,48,59]

f=plt.figure(figsize=(15,4))
for i in range(len(mice)):
    b = behavior.Behavior(r'J:\ephys_data\Behavior data\{}\python_behavior'.format(mice[i]), behavior_only=True)
    _, perf, sess_trials = b.get_acc_EL(150)

    plt.plot(perf[:sess_trials[cutoffs[i]]], color='red', alpha=0.5)
    plt.scatter(len(perf[:sess_trials[cutoffs[i]]]), perf[:sess_trials[cutoffs[i]]][-1], s=150, label=mice[i])
    
plt.legend()
plt.xlabel('# trials')
plt.ylabel('Performance (prop. correct)')    
plt.title('Opto corruption learning trajectories')

#%% Look at early lick behavior in corruption days
b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW53\python_behavior', behavior_only=True)
binsize = 100 # ms
buckets = np.arange(0, s1.response+binsize/1000, binsize/1000)
f = plt.figure(figsize=(8,6))

for t in b.early_lick_time.values():
    all_times = cat(t)

    counts, bin_edges = np.histogram(all_times, bins=buckets)
    plt.plot(bin_edges[:-1], counts, alpha=0.5)
plt.axvline(s1.sample, ls = '--', color = 'black')
plt.axvline(s1.delay, ls = '--', color = 'black')
plt.axvline(s1.response, ls = '--', color = 'black')
plt.hlines(y=305, xmin=s1.sample, xmax=s1.sample+2.3 , linewidth=10, color='red')
plt.axvspan(s1.sample, s1.delay+1, color='red', alpha=0.1)
plt.ylabel('Early lick count')
plt.xlabel('Time (s)')
plt.legend()

# look at the side of early lick
f = plt.figure(figsize=(8,6))

for t in b.early_lick_time.keys():
    all_times = cat(b.early_lick_time[t])
    all_sides = cat(b.early_lick_side[t])

    counts, bin_edges = np.histogram(all_times[np.where(all_sides =='r')[0]], bins=buckets)
    plt.plot(bin_edges[:-1], counts, alpha=0.5, color='blue')
    counts, bin_edges = np.histogram(all_times[np.where(all_sides =='l')[0]], bins=buckets)
    plt.plot(bin_edges[:-1], counts, alpha=0.5, color='red')

plt.axvline(s1.sample, ls = '--', color = 'black')
plt.axvline(s1.delay, ls = '--', color = 'black')
plt.axvline(s1.response, ls = '--', color = 'black')
plt.ylabel('Early lick count')
plt.xlabel('Time (s)')
plt.hlines(y=95, xmin=s1.sample, xmax=s1.sample+2.3 , linewidth=10, color='red')
plt.axvspan(s1.sample, s1.delay+1, color='red', alpha=0.1)
plt.legend()


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

for path in cat(all_learning_paths):
    s1 = Session(path, passive=False, filter_low_perf=True)
    # all_stim_trials = np.where(s1.stim_ON)[0]
    # left_stim_trials = [i for i in np.where(s1.stim_side == 'L')[0] if i in all_stim_trials]
    # right_stim_trials = [i for i in np.where(s1.stim_side == 'R')[0] if i in all_stim_trials]
    # control_trials = np.where(~s1.stim_ON)[0]
    
    left_stim_trials = s1.i_good_L_stim_trials
    right_stim_trials = s1.i_good_R_stim_trials
    control_trials = s1.i_good_non_stim_trials
    
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

#%% Behavioral effect of stim per mouse

paths = [r'G:\ephys_data\CW59\python\2025_02_22',
 r'G:\ephys_data\CW59\python\2025_02_24',
 r'G:\ephys_data\CW59\python\2025_02_25',
 r'G:\ephys_data\CW59\python\2025_02_26',
 r'G:\ephys_data\CW59\python\2025_02_28',
 ]
paths = [
                        r'J:\ephys_data\CW49\python\2024_12_11',
                        r'J:\ephys_data\CW49\python\2024_12_12',
                        r'J:\ephys_data\CW49\python\2024_12_13',
                        r'J:\ephys_data\CW49\python\2024_12_14',
                        r'J:\ephys_data\CW49\python\2024_12_15',
                        r'J:\ephys_data\CW49\python\2024_12_16',
                
                          ]
paths = [r'J:\ephys_data\CW54\python\2025_02_01',
 r'J:\ephys_data\CW54\python\2025_02_03']

paths = [r'G:\ephys_data\CW61\python\2025_03_08',
 # r'G:\ephys_data\CW61\python\2025_03_09', 
 r'G:\ephys_data\CW61\python\2025_03_10', 
 r'G:\ephys_data\CW61\python\2025_03_11', 
 # r'G:\ephys_data\CW61\python\2025_03_12', 
 # r'G:\ephys_data\CW61\python\2025_03_14', 
 r'G:\ephys_data\CW61\python\2025_03_17', 
 # r'G:\ephys_data\CW61\python\2025_03_18', 
 ]
# paths = [
#     r'J:\ephys_data\CW53\python\2025_01_27',
#     r'J:\ephys_data\CW53\python\2025_01_28',
#     r'J:\ephys_data\CW53\python\2025_01_29',
#     # r'J:\ephys_data\CW53\python\2025_01_30',
#     r'J:\ephys_data\CW53\python\2025_02_01',
#     r'J:\ephys_data\CW53\python\2025_02_02',
#       ]

performance_opto_left, performance_opto_right = [], []
performance_ctl = []

stim_left_performance_opto_left, stim_left_performance_opto_right = [], []
stim_right_performance_opto_left, stim_right_performance_opto_right = [], []
performance_ctl_right, performance_ctl_left = [], []

fig = plt.figure()

for path in paths:
    s1 = Session(path, passive=False, filter_low_perf=True)
    # all_stim_trials = np.where(s1.stim_ON)[0]
    # left_stim_trials = [i for i in np.where(s1.stim_side == 'L')[0] if i in all_stim_trials]
    # right_stim_trials = [i for i in np.where(s1.stim_side == 'R')[0] if i in all_stim_trials]
    # control_trials = np.where(~s1.stim_ON)[0]
    
    left_stim_trials = s1.i_good_L_stim_trials
    right_stim_trials = s1.i_good_R_stim_trials
    control_trials = s1.i_good_non_stim_trials
    
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
    r'J:\ephys_data\Behavior data\CW59\python_behavior',
    r'J:\ephys_data\Behavior data\CW61\python_behavior',
    r'J:\ephys_data\Behavior data\CW63\python_behavior',
    r'J:\ephys_data\Behavior data\CW62\python_behavior',
    r'J:\ephys_data\Behavior data\CW60\python_behavior',

    
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



#%% Plot performance in recording sessions only


expert_paths = [
    
    r'J:\ephys_data\Behavior data\CW49\python_behavior',
    r'J:\ephys_data\Behavior data\CW53\python_behavior',
    r'J:\ephys_data\Behavior data\CW59\python_behavior',

    ]
expert_sess = [(15,21), (36,42), (22,29)]

learning_paths = [
    r'J:\ephys_data\Behavior data\CW54\python_behavior',
    r'J:\ephys_data\Behavior data\CW61\python_behavior',
    r'J:\ephys_data\Behavior data\CW63\python_behavior',
    ]
learning_sess = [(25,28), (36,42), (17,24)]

window=75

learning_accs = []
for idx in range(len(learning_paths)):
    b = behavior.Behavior(learning_paths[idx], behavior_only=True)
    _, acc, _ = b.get_acc_EL(window = window, sessions=learning_sess[idx])
    learning_accs += [acc]

expert_accs = []
for idx in range(len(expert_paths)):
    b = behavior.Behavior(expert_paths[idx], behavior_only=True)
    _, acc, _ = b.get_acc_EL(window = window, sessions=expert_sess[idx])
    expert_accs += [acc]


f, ax = plt.subplots(1,2,figsize=(15,5), sharey='row')

for i in range(3):
    ax[0].plot(range(len(learning_accs[i])), learning_accs[i], color='blue')
    ax[1].plot(range(len(expert_accs[i])), expert_accs[i], color='green')
for i in range(2):
    ax[i].axhline(0.5, ls = '--', color='grey')
    ax[i].axhline(0.7, ls = '--', color='grey')
ax[0].set_title('Learning')
ax[1].set_title('Expert')
ax[0].set_ylabel('Behavior performance')
ax[0].set_xlabel('Trials')

f = plt.figure()
for i in range(3):
    plt.plot(range(len(learning_accs[i])), learning_accs[i], color='blue', alpha=0.75)
    plt.plot(range(len(expert_accs[i])), expert_accs[i], color='green', alpha=0.75)
plt.axhline(0.5, ls = '--', color='grey')
plt.axhline(0.7, ls = '--', color='grey')
plt.ylabel('Behavior performance')
plt.xlabel('Trials')

#%% Filter out bad behavior using sliding window threshold? visualize the results


def find_all_consecutive_segments(arr, threshold, count=10):
    is_below = arr < threshold

    start_indices = []
    all_indices = []
    ranges = []

    start = None
    for i, val in enumerate(is_below):
        if val:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= count:
                start_indices.append(start)
                all_indices.extend(range(start, i))
                ranges.append((start, i - 1))
            start = None

    # Handle trailing run
    if start is not None and (len(arr) - start) >= count:
        start_indices.append(start)
        all_indices.extend(range(start, len(arr)))
        ranges.append((start, len(arr) - 1))

    return np.array(start_indices), np.array(all_indices), ranges


window=20
path = r'J:\ephys_data\Behavior data\CW54\python_behavior'
b = behavior.Behavior(path, behavior_only=True)
learning_accs = []
for i in range(25,28):
    _, acc, _ = b.get_acc_EL(window = window, sessions=(i,i+1))
    learning_accs += [acc]

start_idxs, all_idxs, span_ranges = find_all_consecutive_segments(cat(learning_accs), 0.55, 15)

plt.plot(range(len(cat(learning_accs))), cat(learning_accs), color='green')
for start,end in span_ranges:
    plt.plot(np.arange(start,end), cat(learning_accs)[np.arange(start,end)], color='red')

# draw session boundaries
counter = 0
for i in range(len(learning_accs)):
    counter += len(learning_accs[i])
    plt.axvline(counter, ls='--', color='grey')
    
    




