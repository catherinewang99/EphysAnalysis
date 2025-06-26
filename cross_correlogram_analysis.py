# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:46:55 2025

@author: catherinewang

modeled after fig s4 in chen et al 2021
"""

import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from ephysSession import Session
import behavior
from activitymode import Mode

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


#%% Function to calculate cross-correlation


def cross_correlation(x, y, mode='full', unbiased=False):
    """
    Compute the cross-correlation between two 1D vectors.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector. Must be the same length as x.
    mode : {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
        - 'full': returns the full discrete cross-correlation (default)
        - 'same': returns output of length max(M, N)
        - 'valid': returns output of length max(M, N) - min(M, N) + 1
    unbiased : bool, optional
        If True, scale the correlation at each lag by 1/(N - |lag|) instead of 1/N.

    Returns
    -------
    corr : ndarray
        Cross-correlation vector. Length depends on the chosen mode.

    Raises
    ------
    ValueError
        If input vectors have different lengths.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D vectors.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Vectors must have the same length.")

    N = x.shape[0]
    corr = np.correlate(x, y, mode=mode)

    if unbiased:
        if mode == 'full':
            lags = np.arange(-N + 1, N)
            denom = N - np.abs(lags)
            corr = corr / denom
        else:
            corr = corr / N

    return corr

def pearson_correlation(x, y):
    """
    Compute the Pearson correlation coefficient between two 1D vectors.

    Returns a single scalar in [-1,1] representing the zero-lag correlation.

    Parameters
    ----------
    x, y : array_like
        Input vectors of equal length.

    Returns
    -------
    r : float
        Pearson correlation coefficient.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Vectors must have the same shape.")

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    numerator = np.dot(x_centered, y_centered)
    denominator = np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    if denominator == 0:
        raise ValueError("Standard deviation of input is zero.")
    return numerator / denominator

def average_centered(arrays):
    """
    Center each 1D array at its middle and compute a point-wise average and standard error.
    Edges have fewer values, reflected in larger standard errors.

    Parameters
    ----------
    arrays : sequence of array_like
        List of 1D arrays of varying lengths.

    Returns
    -------
    avg : ndarray
        Averaged array, length = max pre- + post-center spans +1.
    std_err : ndarray
        Standard error of the mean at each position.
    """
    seq = [np.asarray(a) for a in arrays]
    if any(a.ndim != 1 for a in seq):
        raise ValueError("All inputs must be 1D arrays.")

    spans = [(len(a) - 1) // 2 for a in seq]
    post_spans = [(len(a) - 1) - pre for a, pre in zip(seq, spans)]
    max_pre = max(spans)
    max_post = max(post_spans)

    total_len = max_pre + 1 + max_post
    sums = np.zeros(total_len, dtype=float)
    sum_sqs = np.zeros(total_len, dtype=float)
    counts = np.zeros(total_len, dtype=int)

    for a, pre in zip(seq, spans):
        L = len(a)
        offset = max_pre - pre
        region = slice(offset, offset + L)
        sums[region] += a
        sum_sqs[region] += a**2
        counts[region] += 1

    # Compute mean
    avg = sums / counts
    # Compute sample variance
    var = np.empty_like(avg)
    mask = counts > 1
    var[mask] = (sum_sqs[mask] - sums[mask]**2 / counts[mask]) / (counts[mask] - 1)
    var[~mask] = np.nan
    # Standard error = sqrt(var) / sqrt(n)
    std_err = np.sqrt(var) / np.sqrt(counts)

    return avg, std_err



#%% Get CD projections and correlations for control trials
from scipy import signal
path = r'G:\ephys_data\CW59\python\2025_02_22'
path = r'J:\ephys_data\CW53\python\2025_01_27'
s1 = Mode(path)
train_test_trials = ([s1.r_train_idx, s1.l_train_idx, s1.r_test_idx, s1.l_test_idx],
                     [s1.r_train_err_idx, s1.l_train_err_idx, s1.r_test_err_idx, s1.l_test_err_idx])


s1 = Mode(path, side='R', train_test_trials=train_test_trials)

proj_allDimR, proj_allDimL = s1.plot_CD(mode_input='choice', auto_corr_return=True, single_trial=True)
delay_idx = np.where((s1.t > s1.delay) & (s1.t < s1.response))[0]
r_projR = [t[delay_idx] for t in proj_allDimR] 
r_projL = [t[delay_idx] for t in proj_allDimL]



s1 = Mode(path, side='L', train_test_trials=train_test_trials)
left_proj_allDimR, left_proj_allDimL = s1.plot_CD(mode_input='choice', auto_corr_return=True, single_trial=True)
l_projR = [t[delay_idx] for t in left_proj_allDimR] 
l_projL = [t[delay_idx] for t in left_proj_allDimL]


all_corrs = []
for i in range(len(r_projR)):
    # all_corrs += [cross_correlation(r_projR[i][:,0], l_projR[i][:,0], unbiased=True)]
    all_corrs += [signal.correlate(r_projR[i][:,0], l_projR[i][:,0], 'full') / len(r_projR[0])*2]
for i in range(len(r_projL)):
    # all_corrs += [cross_correlation(r_projL[i][:,0], l_projL[i][:,0], unbiased=True)]
    all_corrs += [signal.correlate(r_projL[i][:,0], l_projL[i][:,0], 'full') / len(r_projL[0])*2]
    
all_corrs /= np.max(all_corrs)

f=plt.figure()
plt.plot(np.mean(all_corrs, axis=0))
plt.axhline(0, ls='--', color='grey')
plt.axvline(len(all_corrs[0])/2, ls='--', color='grey')
# np.argmax()

# plt.plot(r_projR[0])
# plt.plot(l_projR[0])


# f=plt.figure()
# plt.axhline(0, ls = '--', color='black')
# plt.axvline(0, ls = '--', color='black')
# plt.scatter(l_projL, r_projL, color='red')
# plt.scatter(l_projR, r_projR, color='blue')
# plt.xlabel('Left ALM')
# plt.ylabel('Right ALM')
# plt.title('CD projections (a.u.)')
#%% Get cross correlations for error trials / early lick trials
path =  r'G:\ephys_data\CW59\python\2025_02_22'
path = r'J:\ephys_data\CW53\python\2025_01_27'

# use all to calculate CD choice for early lick
r1 = Mode(path, side='R', proportion_train=1, binsize=400, timestep=50)
cd_choice_right, _ = r1.plot_CD(mode_input='choice')
s1 = Mode(path, side='L', proportion_train=1, binsize=400, timestep=50)
cd_choice_left, _ = s1.plot_CD(mode_input='choice')

# early_lick_left, early_lick_right = [],[] #collect idx of el sides
# for i in range(len(s1.early_lick_side)):
#     if s1.early_lick_side[i][0] == 'l':
#         early_lick_left += [i]
#     else:
#         early_lick_right += [i]
# early_lick_left = [i for i in early_lick_left if i in s1.i_good_trials]
# early_lick_right = [i for i in early_lick_right if i in s1.i_good_trials]
early_lick_trials = [i for i in np.where(s1.early_lick)[0] if i in s1.i_good_trials and i in r1.i_good_trials]

time = s1.t

all_corrs = []

for i,t in enumerate(early_lick_trials):
    time_adj = time[np.where(time<s1.early_lick_time[i][0][0])]
    if len(time_adj) == 0:
        continue
    activity, _, _ = s1.get_PSTH_multiple(s1.good_neurons, [t])#, binsize=self.binsize, timestep = self.timestep)
    proj_allDim_L = np.dot(activity.T, cd_choice_left)[:len(time_adj)]
    
    activity_r, _, _ = r1.get_PSTH_multiple(r1.good_neurons, [t])#, binsize=self.binsize, timestep = self.timestep)
    proj_allDim_R = np.dot(activity_r.T, cd_choice_right)[:len(time_adj)]
    
    c=cross_correlation(proj_allDim_L[:,0], proj_allDim_R[:,0], unbiased=True)
    # c= signal.correlate(proj_allDim_L[:,0], proj_allDim_R[:,0], 'full')
    all_corrs += [c ]
    

    
# all_corrs /= np.max(all_corrs)
avg, err = average_centered(all_corrs)
f=plt.figure()
plt.plot(avg/np.max(abs(avg)))
plt.axhline(0, ls='--', color='grey')
plt.axvline(len(avg)/2, ls='--', color='grey')


# Error trials






