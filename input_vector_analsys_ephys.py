# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:54:31 2025

Do input vector analysis on ephys data to compare with imaging results

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
from numpy.linalg import norm



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
    return np.dot(a, b)/(norm(a)*norm(b))

#%% Paths
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


#%% Alignment of input vector with choice CD

path = r'J:\ephys_data\CW53\python\2025_01_29'
CD_angle = []
for path in expert_path:
    
    s1 = Mode(path, side='R', stimside='L')
    
    # input_vec = s1.input_vector(by_trialtype=True, plot=True)
    
    input_vec = s1.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=True)
    # input_vec = s1.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=False)
    
        
    cd_choice, _ = s1.plot_CD(mode_input='choice', plot=False)
    
    CD_angle += [cos_sim(input_vec, cd_choice)]
    
# Plot

f=plt.figure()
plt.scatter(np.ones(len(CD_angle)), CD_angle)
plt.bar([1], [np.mean(CD_angle)])
plt.ylabel('Angle')


#%% Alignment of excited vs inhibited cells
p_s = 0.01

CD_angle_filtered, CD_angle = [], []

for path in expert_path:
    
    s1 = Mode(path, side='R', stimside='L')
    
    # input_vec = s1.input_vector(by_trialtype=True, plot=True)
    stim_period = (s1.delay+0, s1.delay+1)
    exp_susc, tstat = s1.susceptibility(stimside= 'L', period = stim_period, p=p_s, return_n=True)
    exc_n = np.array(exp_susc)[np.where(np.array(tstat) > 0)[0]]
    exc_n_idx = [np.where(s1.good_neurons == n)[0][0] for n in exc_n]
    
    input_vec = s1.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=True)    
        
    cd_choice, _ = s1.plot_CD(mode_input='choice', plot=False)
    
    CD_angle += [cos_sim(input_vec, cd_choice)]
    CD_angle_filtered += [(cos_sim(input_vec[exc_n_idx], cd_choice[exc_n_idx]))]
    
    
    
# Plot angle between choice CD and input vector

plt.bar([0],np.nanmean(CD_angle_filtered))
plt.scatter(np.zeros(len(CD_angle_filtered)), np.array(CD_angle_filtered))
# plt.scatter(np.ones(len(CD_angle_filtered)), np.array(CD_angle_filtered)[:, 1])
# for i in range(len(CD_angle_filtered)):
#     plt.plot([0,1],[CD_angle_filtered[i,0], CD_angle_filtered[i,1]], color='grey')
# plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Dot product')
plt.title('Input vector alignment to choice CD (inh)')
plt.show()


    
for path in expert_path:
    
    s1 = Mode(path, side='R', stimside='L')
    
    # input_vec = s1.input_vector(by_trialtype=True, plot=True)
    stim_period = (s1.delay+0, s1.delay+1)
    exp_susc, tstat = s1.susceptibility(stimside= 'L', period = stim_period, p=p_s, return_n=True)
    exc_n = np.array(exp_susc)[np.where(np.array(tstat) < 0)[0]]
    exc_n_idx = [np.where(s1.good_neurons == n)[0][0] for n in exc_n]
    
    input_vec = s1.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=True)    
        
    cd_choice, _ = s1.plot_CD(mode_input='choice', plot=False)
    
    CD_angle += [cos_sim(input_vec, cd_choice)]
    CD_angle_filtered += [(cos_sim(input_vec[exc_n_idx], cd_choice[exc_n_idx]))]
    
    
    
# Plot angle between choice CD and input vector


plt.bar([0],np.nanmean(CD_angle_filtered[11:]))
plt.scatter(np.zeros(len(CD_angle_filtered[11:])), np.array(CD_angle_filtered[11:]))
# plt.scatter(np.ones(len(CD_angle_filtered)), np.array(CD_angle_filtered)[:, 1])
# for i in range(len(CD_angle_filtered)):
#     plt.plot([0,1],[CD_angle_filtered[i,0], CD_angle_filtered[i,1]], color='grey')
# plt.xticks([0,1],['Learning','Expert'])
plt.ylabel('Dot product')
plt.title('Input vector alignment to choice CD (exc)')
plt.show()


#%% Correlate alignment with behavior
deltas, perf = [],[]
for path in expert_path:
    l1 = Session(path)
    
     ## BEHAVIOR PERFORMANCE 
    stim_trials = np.where(l1.stim_ON)[0]
    control_trials = np.where(~l1.stim_ON)[0]
    stim_trials = [c for c in stim_trials if c in l1.i_good_trials]
    stim_trials = [c for c in stim_trials if l1.stim_side[c] == 'L']
    stim_trials = [c for c in stim_trials if ~l1.early_lick[c]]
    control_trials = [c for c in control_trials if c in l1.i_good_trials]
    control_trials = [c for c in control_trials if ~l1.early_lick[c]]
    
    _, _, perf_all = l1.performance_in_trials(stim_trials)
    _, _, perf_all_c = l1.performance_in_trials(control_trials)
 
    if perf_all_c < 0.5: #or perf_all / perf_all_c > 1: #Skip low performance sessions
        print(l1.path)
        continue

    deltas += [perf_all_c - perf_all]
    perf += [perf_all_c]

f=plt.figure()
plt.scatter(CD_angle_filtered[11:], deltas)
# plt.scatter(CD_angle_filtered[:11], deltas)
plt.xlabel('Angle (exc input vector and CD)')
plt.ylabel('Delta in behavior')
r_value, p_value = pearsonr(CD_angle_filtered[11:], deltas)
print(r_value, p_value)


f=plt.figure()
plt.scatter(CD_angle_filtered[11:], perf)
plt.xlabel('Angle (exc input vector and CD)')
plt.ylabel('Behavior performance')
        
f=plt.figure()
plt.scatter(CD_angle_filtered[:11], deltas)
plt.xlabel('Angle (inh input vector and CD)')
plt.ylabel('Delta in behavior')
r_value, p_value = pearsonr(CD_angle_filtered[:11], deltas)
print(r_value, p_value)

    
f=plt.figure()
plt.scatter(CD_angle_filtered[:11], perf)
plt.xlabel('Angle (inh input vector and CD)')
plt.ylabel('Behavior performance')
             