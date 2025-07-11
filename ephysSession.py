# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:34:50 2024

Main object to store and analyze neuropixel data

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
from numpy import concatenate as cat
import matplotlib.pyplot as plt
from scipy import stats
import scipy
import copy
import scipy.io as scio
from sklearn.preprocessing import normalize
from matplotlib.pyplot import figure
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import normalize
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import mstats
from scipy.ndimage import median_filter
# from .LinRegpval import LinearRegression
plt.rcParams['pdf.fonttype'] = 42 
import random
from itertools import groupby
from operator import itemgetter
from scipy.signal import convolve
from scipy.signal import fftconvolve
import time as time_

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class Session:
    """
    A class used to store and process electrophysiology data alongside
    behavior data
 
    ...
 
    Attributes
    ----------

 
    Methods
    -------

    """
    def __init__(self, path, side = 'all', stimside = 'all',
                 passive=False, laser='blue', only_ppyr=True,
                 filter_low_perf=True, filter_by_stim=True,
                 anterior_shank=False,
                 include_early_lick=False):
        

        """
        Parameters
        ----------
        path : str
            Path to data
        side : str
            Default 'all', can optionally grab only 'L' or 'R' alm hemi
        passive : bool
            If true, then we are looking at a passive session. Otherwise, beh
        laser : str
            Default blue laser, only matters for passive recordings

        """       
        
        self.path = path
        self.laser = laser
        self.side = side
        self.passive = passive
        
        name = laser + 'passive_units.mat' if passive else 'units.mat'
        units_tmp = scio.loadmat(os.path.join(path, name))
        units = copy.deepcopy(units_tmp)
            
        imec = units['imec']
        self.unit_side = np.where(imec == 0, 'L', 'R')[0] # Shape: (units)
        self.L_alm_idx = np.where(imec == 0)[1]
        self.R_alm_idx = np.where(imec == 1)[1]
        
        self.celltype = units['celltype'][0] # Shape: (units,)
        self.fs_idx = np.where(self.celltype == 1)[0]
        self.pyr_idx = np.where(self.celltype == 3)[0]
        
        if 'shank' in units.keys():
            self.shank = units['shank'][0]
        
        self.num_neurons = units['units'].shape[1]
        self.good_neurons = np.arange(self.num_neurons)
        
        if side == 'L':
            self.num_neurons = len(self.L_alm_idx)
            self.good_neurons = self.L_alm_idx
        elif side == 'R':
            self.num_neurons = len(self.R_alm_idx)
            self.good_neurons = self.R_alm_idx

        self.side = side
        
        self.num_trials = units['units'][0,0].shape[1]
        
        self.spks = units['units'][0] # Shape: (units,)(1, trials)
        
        for trial in range(units['stabletrials'][0].shape[0]):
            units['stabletrials'][0,trial] = units['stabletrials'][0,trial][0] - 1 # Convert to 0 index
        self.stable_trials = units['stabletrials'][0] # Shape: (units,)(stable_trials,)
        
        self.waveform = units['mean_waveform'][0]
        
        self.data_type = 'ephys'
        self.sample = .570 # 570 ms offset for sample start
        self.delay = 0.57 + 1.4
        self.response = 0.57 + 1.4 + 3
        self.time_cutoff = self.response+3
        if passive:
            self.sample = .570 # 570 ms offset for sample start
            self.delay = 0.57 + 1.4
            self.response = 0.57 + 1.4 + 1
            self.time_cutoff = self.response+1
            
        self.sampling_freq = 30000 # 30k for npxl 2.0
        
        # Behavior
        beh_name = laser + 'passive_behavior.mat' if passive else 'behavior.mat'
        behavior = scio.loadmat(os.path.join(path, beh_name))
        self.L_correct = cat(behavior['L_hit_tmp'])
        self.R_correct = cat(behavior['R_hit_tmp'])
        
        self.early_lick = cat(behavior['LickEarly_tmp'])

        # Early lick timing info (if applicable)
        if 'earlyLick_time' in behavior.keys():
            self.early_lick_time = cat(behavior['earlyLick_time'])
            self.early_lick_side = cat(behavior['earlyLick_side'])
            
        self.L_wrong = cat(behavior['L_miss_tmp'])
        self.R_wrong = cat(behavior['R_miss_tmp'])
        
        self.L_ignore = cat(behavior['L_ignore_tmp'])
        self.R_ignore = cat(behavior['R_ignore_tmp'])
        
                        
        self.lick_L_trials = np.sort(cat((np.where(self.L_correct)[0], 
                                     np.where(self.R_wrong)[0])))
        
        self.lick_R_trials = np.sort(cat((np.where(self.R_correct)[0], 
                                     np.where(self.L_wrong)[0]))) 
        
        self.L_trials = np.sort(cat((np.where(self.L_correct)[0], 
                                     np.where(self.L_wrong)[0],
                                     np.where(self.L_ignore)[0])))
        
        self.R_trials = np.sort(cat((np.where(self.R_correct)[0], 
                                     np.where(self.R_wrong)[0],
                                     np.where(self.R_ignore)[0]))) 
        
        self.stim_ON = cat(behavior['StimDur_tmp']) > 0
        
        self.i_good_trials = cat(behavior['i_good_trials']) - 1 # zero indexing in python
        if filter_low_perf:
            self.filter_low_performance()        

        
        # Take out non stable trials from igoodtrials
        stable_trials_tmp = self.stable_trials[self.good_neurons] # Only filter based on the relevant neurons
        common_values = stable_trials_tmp[0]
        for arr in stable_trials_tmp[1:]:
            common_values = np.intersect1d(common_values, arr)
        print('{} trials removed for stability reasons'.format(len(self.i_good_trials) - len(np.intersect1d(common_values, self.i_good_trials))))
        
        self.i_good_trials = np.intersect1d(common_values, self.i_good_trials)
        self.i_good_non_stim_trials = [t for t in self.i_good_trials if not self.stim_ON[t] and not self.early_lick[t]]
        self.i_good_stim_trials = [t for t in self.i_good_trials if self.stim_ON[t] and not self.early_lick[t]]
        
        
        if 'StimLevel' in behavior.keys():
            self.stim_level = cat(behavior['StimLevel'])
            self.all_stim_levels = sorted(list(set(self.stim_level)))
            if not passive and laser == 'blue':
                x_galvo = cat(behavior['xGalvo'])
                self.stim_side = np.where(x_galvo < 0, 'L', 'R')
            elif laser == 'red':
                self.stim_side = np.full(self.num_trials, 'L')
                
        self.i_good_R_stim_trials = [t for t in self.i_good_trials if self.stim_side[t] == 'R' and self.stim_ON[t] and not self.early_lick[t]]
        self.i_good_L_stim_trials = [t for t in self.i_good_trials if self.stim_side[t] == 'L' and self.stim_ON[t] and not self.early_lick[t]]

        
        # Re-adjust with i good trials
        self.stim_trials = np.where(self.stim_ON)[0]
        
        if only_ppyr:
            oldn = len(self.good_neurons)
            self.good_neurons = [n for n in self.good_neurons if n in np.where(self.celltype == 3)[0]]
            if filter_by_stim:
                self.filter_by_stim_effect()
            print('only ppyr neurons: {} --> {}'.format(oldn, len(self.good_neurons)))
            self.L_alm_idx = [n for n in self.L_alm_idx if n in self.good_neurons]
            self.R_alm_idx = [n for n in self.R_alm_idx if n in self.good_neurons]
        
        if anterior_shank:
            
            new_good_neurons = []
            for n in self.good_neurons:
                if n in self.L_alm_idx:
                    if self.shank[n] == 3 or self.shank[n] == 4:
                        new_good_neurons += [n]
                elif n in self.R_alm_idx:
                    if self.shank[n] == 1 or self.shank[n] == 2:
                        new_good_neurons += [n]
            
            print("new anterior neuron count: {} --> {}".format(len(self.good_neurons), len(new_good_neurons)))
            self.good_neurons = new_good_neurons

        
    ### BEHAVIOR ONLY FUNCTIONS ###
    
    def get_acc_conv(self, window = 50):
        """Returns agg accuracy over all sessions
        
        Parameters
        -------
        window : int
        imaging : bool
        sessions : tuple, optional
            if provided, use these session for the return array
    
        Returns
        -------
        int
            int corresponding to the length of shortest trial in whole session
        """
        # Concatenate all sessions
        correctarr = np.array([])
        earlylicksarr = np.array([])
        num_trials = []
        window = int(window/2)

        
        correct = self.L_correct + self.R_correct
        correct = np.convolve(correct, np.ones(window*2)/(window*2), mode = 'same')
        correctarr = np.append(correctarr, correct[window:-window])
        
        earlylicks = np.convolve(self.early_lick, np.ones(window*2)/(window*2), mode = 'same')
        earlylicksarr = np.append(earlylicksarr, earlylicks[window:-window])
          
        return earlylicksarr, correctarr, cat(([0], num_trials))
    
    
    def find_all_consecutive_segments(self, arr, threshold, count=10):
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
    
    def filter_low_performance(self, threshold=0.5, consec_len=15):
        """
        Filter out the low performing trials from i_good_trials

        Parameters
        ----------
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        _, correctarr, _ = self.get_acc_conv(window=20)
        start_idxs, all_idxs, span_ranges = self.find_all_consecutive_segments(correctarr, threshold, consec_len)
        trials = self.i_good_trials[~np.isin(self.i_good_trials, all_idxs)]
        print("Remove {} trials from session for bad behavior".format(len(all_idxs)))
        self.i_good_trials = trials
    
    def performance_in_trials(self, trials):
        """
        Get the performance as a percentage correct for the given trials numbers

        Parameters
        ----------
        trials : list
            List of trials to calculate correctness.

        Returns
        -------
        A single number corresponding to proportion correct in left and right trials.

        """
        
        proportion_correct_left = np.sum(self.L_correct[trials]) / np.sum(self.L_correct[trials] + self.L_wrong[trials] + self.L_ignore[trials])
        proportion_correct_right = np.sum(self.R_correct[trials]) /  np.sum(self.R_correct[trials] + self.R_wrong[trials] + self.R_ignore[trials])
        proportion_correct = np.sum(self.L_correct[trials] + self.R_correct[trials]) / np.sum(self.L_correct[trials] + 
                                                                                              self.L_wrong[trials] + 
                                                                                              self.L_ignore[trials] +
                                                                                              self.R_correct[trials] + 
                                                                                              self.R_wrong[trials] + 
                                                                                              self.R_ignore[trials])
                                                                                          
    
        return proportion_correct_right, proportion_correct_left, proportion_correct
    
    def lick_correct_direction(self, direction):
        """Finds trial numbers corresponding to correct lick in specified direction

        Parameters
        ----------
        direction : str
            'r' or 'l' indicating desired lick direction
        
        Returns
        -------
        idx : array
            list of correct, no early lick, i_good trials licking in specified
            direction
        """
        
        if direction == 'l':
            idx = np.where(self.L_correct == 1)[0]
        elif direction == 'r':
            idx = np.where(self.R_correct == 1)[0]
        else:
            raise Exception("Sorry, only 'r' or 'l' input accepted!")
            
        early_idx = np.where(self.early_lick == 1)[0]
        
        idx = [i for i in idx if i not in early_idx]
        
        idx = [i for i in idx if i in self.i_good_trials]
        
        return idx
    
    def lick_incorrect_direction(self, direction):
        """Finds trial numbers corresponding to incorrect lick in specified direction

        Parameters
        ----------
        direction : str
            'r' or 'l' indicating desired lick direction
        
        Returns
        -------
        idx : array
            list of incorrect, no early lick, i_good trials licking in specified
            direction
        """
        
        if direction == 'l':
            idx = np.where(self.L_wrong == 1)[0]
        elif direction == 'r':
            idx = np.where(self.R_wrong == 1)[0]
        else:
            raise Exception("Sorry, only 'r' or 'l' input accepted!")
            
        early_idx = np.where(self.early_lick == 1)[0]
        
        idx = [i for i in idx if i not in early_idx]
        
        idx = [i for i in idx if i in self.i_good_trials]
        
        return idx
    
    def lick_actual_direction(self, direction):
        """Finds trial numbers corresponding to an actual lick direction
        
        Filters out early lick and non i_good trials but includes correct and 
        error trials

        Parameters
        ----------
        direction : str
            'r' or 'l' indicating desired lick direction
        
        Returns
        -------
        idx : array
            list of trials corresponding to specified lick direction
        """
        
        ## Returns list of indices of actual lick left/right trials
        
        if direction == 'l':
            idx = np.where((self.L_correct + self.R_wrong) == 1)[0]
        elif direction == 'r':
            idx = np.where((self.R_correct + self.L_wrong) == 1)[0]
        else:
            raise Exception("Sorry, only 'r' or 'l' input accepted!")
            
        early_idx = np.where(self.early_lick == 1)[0]
        
        idx = [i for i in idx if i not in early_idx]
        
        idx = [i for i in idx if i in self.i_good_trials]
        
        return idx
    
    def trial_type_direction(self, direction):
        """Finds trial numbers corresponding to trial type direction
        
        Filters out early lick and non i_good trials but includes correct and 
        error trials

        Parameters
        ----------
        direction : str
            'r' or 'l' indicating desired trial type
        
        Returns
        -------
        idx : array
            list of trials corresponding to specified lick direction
        """
        
        ## Returns list of indices of actual lick left/right trials
        
        if direction == 'l':
            idx = np.where((self.L_correct + self.L_wrong) == 1)[0]
        elif direction == 'r':
            idx = np.where((self.R_correct + self.R_wrong) == 1)[0]
        else:
            raise Exception("Sorry, only 'r' or 'l' input accepted!")
            
        early_idx = np.where(self.early_lick == 1)[0]
        
        idx = [i for i in idx if i not in early_idx]
        
        idx = [i for i in idx if i in self.i_good_trials]
        
        return idx
    
    def instructed_direction(self):
        """
        Returns the instructed direction for every trial 
        """
        all_trials_direction = np.ones(self.num_trials)
        
        idx = np.where((self.L_correct + self.L_wrong) == 1)[0]
        all_trials_direction = ['l' if i in idx else 'r' for i in range(self.num_trials)]
        # idx = np.where((self.R_correct + self.R_wrong) == 1)[0]
        # all_trials_direction[idx] = 'r'
        
        return all_trials_direction
        
    ### QUALITY FUNCTIONS ###

    def filter_by_stim_effect(self):
        """
        Filter out all the ppy neurons that are excited by stim (sst or other)

        Returns
        -------
        None.

        """
        
        keep_n = []
        p_sst_n = []
        window = (self.delay, self.delay+1)
        for n in self.good_neurons:
                
            stim_trials = self.i_good_L_stim_trials if n in self.L_alm_idx else self.i_good_R_stim_trials
            ctl_rate = self.get_spike_rate(n, window, self.i_good_non_stim_trials)
            stim_rate = self.get_spike_rate(n, window, stim_trials)
            
            if stim_rate < ctl_rate:
                
                keep_n += [n]
            
            else:
                
                p_sst_n += [n]
                
        self.good_neurons = keep_n

        self.sst_neurons = p_sst_n
    
    ### NEURAL FUNCTIONS ###
    
    def get_waveform_width(self, neuron):
        
        wf = self.get_single_waveform(neuron)
        minidx = np.argmin(wf)
        maxidx = np.argmax(wf[minidx:])
        
        # return np.abs((maxidx - minidx) / self.sampling_freq * 1000) # by ms
        return (maxidx / self.sampling_freq * 1000) # by ms
    
    def get_single_waveform(self, neuron):
        # Peak channel is 165-246
        wf = self.waveform[neuron][0, 164:245] / np.linalg.norm(self.waveform[neuron][0, 164:245])
        return wf 
    
    def plot_mean_waveform(self, n):
        
        plt.plot(np.arange(len(self.get_single_waveform(n))),
                 self.get_single_waveform(n))
        
    
    def plot_mean_waveform_by_celltype(self, both = True):
        """
        Plot waveforms by cell type

        Parameters
        ----------
        both : bool, optional
            Both sides L/R alm included. The default is True.

        Returns
        -------
        Plots a 1x3 average waveform plot. 2x3 if both is true

        """
        
        num_cell_types = len(set(self.celltype))
        
        if both:
            f, ax = plt.subplots(2, num_cell_types, figsize=(15,8))
        else:
            f, ax = plt.subplots(1, num_cell_types)
        
        counts = []
        
        for l_unit in self.L_alm_idx: # go thru all the l units first
            av_array = np.zeros(410)
            counter = 0
            for i in list(set(self.celltype)): # go thru all the cell types
                if self.celltype[l_unit] == i:
                    av_array = np.vstack((av_array, self.waveform[l_unit][0]))
                    ax[0, i-1].plot(self.waveform[l_unit][0])
                    counter += 1
                counts += [counter]
                
                
        for r_unit in self.R_alm_idx: # go thru all the r units 
            av_array = np.zeros(410)
            counter = 0
            for i in list(set(self.celltype)): # go thru all the cell types
                if self.celltype[r_unit] == i:
                    av_array = np.vstack((av_array, self.waveform[r_unit][0]))
                    ax[1, i-1].plot(self.waveform[r_unit][0])
                    counter += 1
                counts += [counter]

    

        ax[0,0].set_ylabel('L ALM waveforms')
        ax[1,0].set_ylabel('R ALM waveforms')
        ax[0,0].set_title('Cell type: FS')
        ax[0,1].set_title('Cell type: intermediate')
        ax[0,2].set_title('Cell type: ppyr')
        ax[0,3].set_title('Cell type: pDS')
        plt.show()
        
        
        # return counts
    
    def stim_effect_per_neuron(self, n, stimtrials, 
                               p=0.01, passive=True, window=()):
        """
        Give the fraction change per neuron in spk rt and -1 if supressed 
        significantly and 1 if excited significantly (0 otherwise)
        
        Return as a single neuron measure

        Parameters
        ----------
        p : Int, optional
            Significantly modulated neuron threshold. The default is 0.01.
        
        period : array, optional
            Time frame to calculate effect of stim

        Returns
        -------
        frac : int
        neuron_sig : array of length corresponding to number of neurons, {0, 1, -1}

        """

        if len(window) == 0:
            window = (0.57, 0.57+1.3) if passive else (0.57+1.3, 0.57+2.3)
            
        control_trials = np.where(~self.stim_ON)[0]
        control_trials = [c for c in control_trials if c in self.stable_trials[n]]

        ctl_counts = self.get_spike_count(n, window, control_trials)
        stim_counts = self.get_spike_count(n, window, stimtrials)

        tstat, p_val = stats.ttest_ind(stim_counts, ctl_counts)
        
        if p_val < p:
            neuron_sig = 1 if tstat > 0 else -1
        else:
            neuron_sig = 0
            
        return neuron_sig


        
    def count_spikes(self, arr, window):
        """
        Return the number of spikes in given arr within time window

        Parameters
        ----------
        arr : array 
            List of spike times. Shape: (timesteps, 1)
        window : tuple, 
            start and end of spike calculation.

        Returns
        -------
        number denoting number of spikes.

        """
        if arr.size == 0: # if no spikes
            return 0
        start, stop = window
        arr = arr[:,0]
        count = np.sum((arr >= start) & (arr <= stop))
        return count
    
    def get_spike_count(self, neuron, window, trials):
        """
        Vectorized: Get spike counts per trial for a specific neuron and time window.
        """
        start, stop = window
    
        # Extract all spike times for the neuron and selected trials
        trial_spks = self.spks[neuron][0, trials]
    
        # Use list comprehension + NumPy for speed
        counts = [np.sum((spk >= start) & (spk < stop)) for spk in trial_spks]
    
        return counts
    
    def get_spike_count_old(self, neuron, window, trials):
        """
        Get spike counts per trial for specific window for specific neuron
        
        """
        all_spk_rate = []

        start, stop = window
        
        trial_spks = self.spks[neuron][0, trials]
        for arr in trial_spks:
            count = self.count_spikes(arr, window)
            all_spk_rate += [count]
            
        return all_spk_rate
        
        
    
    def get_spike_rate(self, neuron, window, trials, mean=True):
        """
        Get average spike rate of neuron in a window, over specified trials

        Parameters
        ----------
        neuron : int
            index of neuron.
        window : tuple
            start and end time over which to grab the spikes
        trials : array
            which trials to calculate over

        Returns
        -------
        One number denoting avg spks/sec over trials.

        """
        all_spk_rate = []

        start, stop = window
        window_len = stop - start 
        
        trial_spks = self.spks[neuron][0, trials]
        for arr in trial_spks:
            count = self.count_spikes(arr, window)
            all_spk_rate += [count / window_len]
            
        if mean:
            return np.mean(all_spk_rate)
        else:
            return all_spk_rate
    
        
    def get_PSTHfaster(self, neuron, trials, binsize=50, timestep=1, period=()):
        """
        Return Peristimulus Time Histogram (PSTH) of a given neuron over specific trials.
        
        Optimized for speed.
        """
        start = time_.time()
        # Find valid trials (intersection for stable neurons)
        trials = np.intersect1d(trials, self.stable_trials[neuron])
    
        # Return early if no trials are found
        if len(trials) == 0:
            return [], [], []
    
        # Convert timestep to seconds
        timestep = timestep / 1000
    
        # Define time window
        start, stop = (-0.2, self.time_cutoff)
        time = np.arange(start, stop, 0.001)  # High resolution binning
    
        # Compute histogram in a vectorized manner
        bins = np.arange(start, stop + 0.001, 0.001)  # Histogram bin edges
        total_counts = np.zeros((len(trials), len(time)))
    
        spike_data = np.array([self.spks[neuron][0, i_rep] for i_rep in trials])  # Gather all spike data
    
        hist_counts = np.array([np.histogram(spikes, bins=bins)[0] for spikes in spike_data])
        total_counts[:] = hist_counts / len(trials)  # Normalize by number of trials
    
        # Compute standard error (vectorized)
        stderr = np.std(total_counts, axis=0) / np.sqrt(len(trials))
    
        # Define smoothing window for convolution
        window = np.ones(binsize) / (binsize / 1000)
    
        # Compute PSTH using fast FFT-based convolution
        total_counts = np.sum(total_counts, axis=0)
        PSTH = fftconvolve(total_counts, window, mode='same')
    
        # Trim edge effects from convolution
        trim_indices = slice(binsize, -binsize)
        time, PSTH, stderr = time[trim_indices], PSTH[trim_indices], stderr[trim_indices]
    
    
        print(start - time_.time())
        # Handle optional period slicing
        if len(period) != 0:
            start, stop = period
            period_idx = (time > start) & (time < stop)
            return PSTH[period_idx], time[period_idx], stderr[period_idx]
    
        return PSTH, time, stderr

    
    def get_PSTH(self, neuron, trials, binsize=50, timestep=1, period=()):
        """
        
        Return peristimulus time histogram of a given neuron over specific trials

        Parameters
        ----------
        neuron : TYPE
            DESCRIPTION.
        trials : TYPE
            DESCRIPTION.
        binsize : TYPE, optional
            DESCRIPTION. The default is 50.
        timestep : TYPE, optional
            DESCRIPTION. The default is 1.
        window : TYPE, optional
            DESCRIPTION. The default is ().

        Returns:
        - PSTH: array, convolved histogram of spike times
        - time: array, time values corresponding to PSTH
        """
        
        start = time_.time()

        trials = np.intersect1d(trials, self.stable_trials[neuron])
        
        if len(trials) == 0:
            return [], [], []
        
        timestep = timestep / 1000
        # if self.passive:
        #     start, stop = window if len(window) != 0 else (-0.2, self.time_cutoff)
        # else:
        #     start, stop = window if len(window) != 0 else (-0.2, self.time_cutoff)
        start, stop = (-0.2, self.time_cutoff)
        time = np.arange(start, stop, 0.001)
        
        n_rep = len(trials)
        total_counts = np.zeros((n_rep, len(time)))

        for idx, i_rep in enumerate(trials):
            counts, _ = np.histogram(self.spks[neuron][0, i_rep], 
                                     bins=np.arange(start, stop+0.001, 0.001))
            total_counts[idx] = counts / n_rep  # Fill the preallocated array
        # total_counts = np.zeros_like(time)
        
        # # Loop over each repetition
        # for i_rep in trials:
        #     # Calculate histogram for each spike train
        #     counts, _ = np.histogram(self.spks[neuron][0, i_rep], 
        #                              bins=np.arange(start, stop + 0.001, 0.001))
        #     total_counts = np.vstack((total_counts, counts / n_rep))

        # stderr = np.std(total_counts[1:], axis=0) / np.sqrt(total_counts.shape[0])
        stderr = np.std(total_counts, axis=0) / np.sqrt(total_counts.shape[0])

        # Define window for convolution (smoothing)
        window = np.ones(binsize) / (binsize / 1000)
        
        # Convolve histogram with smoothing window
        # total_counts = np.sum(total_counts[1:], axis=0)
        # PSTH = convolve(total_counts, window, mode='same')
        total_counts = total_counts.sum(axis=0)
        PSTH = fftconvolve(total_counts, window, mode='same')

        # Adjust time and PSTH to remove edge effects from convolution
        trim_indices = slice(binsize, -binsize)
        time = time[trim_indices]
        PSTH = PSTH[trim_indices]
        stderr = stderr[trim_indices]
        
        # print(start - time_.time())
        self.time=time
        
        if len(period) != 0:
            start,stop = period
            period_idx = np.where((time > start) & (time < stop))
            return PSTH[period_idx], time[period_idx], stderr[period_idx]
        
        return PSTH, time, stderr
    
    def get_PSTH_multiple(self, neurons, trials, binsize=50, timestep=1, window=()):
        
        """
        returns the concatenated PSTH of multiple of neurons over some trials
        """
        all_PSTH = []
        all_std_err = []
        drop_n = [] # record which neurons are dropped
        for neuron in neurons:
            PSTH, time, stderr = self.get_PSTH(neuron = neuron, trials=trials, 
                                               binsize=binsize, timestep=timestep,
                                               period=window)
            # if len(PSTH) == 0:
                # drop_n += [neuron]
                
                # continue
            
            all_PSTH += [PSTH]
            all_std_err += [stderr]

        # return 
        if len(drop_n) == 0:
            return np.array(all_PSTH), time, np.array(all_std_err)       
        else:
            return np.array(all_PSTH), time, np.array(all_std_err), drop_n      

            
    
    ### SIMPLE PLOTTING FUNCTIONS ###
        
    
    def plot_raster(self, neuron, window=(), trials=[], fig=None):
        """
        

        Parameters
        ----------
        neuron : TYPE
            DESCRIPTION.
        trials : TYPE, optional
            DESCRIPTION. The default is [].

        Returns
        -------
        None.

        """
        if fig is None:
            fig = plt.figure()
            ax = fig.gca()
        else:
            ax = fig.gca()
        if len(trials) == 0:
            trials = self.stable_trials[neuron]
            
        for i in range(len(trials)):
            
            if len(window) == 0:
                ax.scatter(self.spks[neuron][0, trials[i]], 
                            np.ones(len(self.spks[neuron][0, trials[i]])) * i, 
                            color='black', s=5)
            else:
                
                start,stop = window
                
                spks = self.spks[neuron][0, trials[i]]
                spks = [s for s in spks if s > start and s < stop]
                
                ax.scatter(spks,
                            np.ones(len(spks)) * i,
                            color='black', s=5)
                
        return fig
        
        
    def plot_raster_and_PSTH(self, neuron, window=(), opto=False,
                             binsize=200, timestep=10, stimside = 'both',
                             save=[]):
        """
        
        Plot raster and PSTH (top, bottom) for given neuron

        Parameters
        ----------
        neuron : TYPE
            DESCRIPTION.
        window : TYPE, optional
            DESCRIPTION. The default is ().
        opto : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        # stable_trials = self.stable_trials[neuron]
        L_correct_trials = self.lick_correct_direction('l')
        L_correct_trials = [i for i in L_correct_trials if not self.stim_ON[i]]# and i in stable_trials]
        R_correct_trials = self.lick_correct_direction('r')
        R_correct_trials = [i for i in R_correct_trials if not self.stim_ON[i]]# and i in stable_trials]
        
        if not opto:

            title = "Neuron {}: Raster and PSTH".format(neuron)
            
            f, axarr = plt.subplots(2, sharex=True, figsize=(10,10))
    
            #RASTER:
            counter = 0
            for i in R_correct_trials:
                axarr[0].scatter(self.spks[neuron][0, i], 
                            np.ones(len(self.spks[neuron][0, i])) * counter, 
                            color='blue', s=5)
                counter += 1
            for i in L_correct_trials:
                axarr[0].scatter(self.spks[neuron][0, i], 
                            np.ones(len(self.spks[neuron][0, i])) * counter, 
                            color='red', s=5)
                counter += 1

                        
            axarr[0].axis('off')

            #TRACES:
                
            L_av, time, left_err = self.get_PSTH(neuron, L_correct_trials, 
                                                 binsize=binsize, timestep=timestep)
            R_av, _, right_err = self.get_PSTH(neuron, R_correct_trials,
                                                  binsize=binsize, timestep=timestep)
                
            axarr[1].plot(time, L_av, 'r-')
            axarr[1].plot(time, R_av, 'b-')
            
            
            axarr[1].fill_between(time, L_av - left_err, 
                     L_av + left_err,
                     color=['#ffaeb1'])
            axarr[1].fill_between(time, R_av - right_err, 
                     R_av + right_err,
                     color=['#b4b2dc'])
            
            axarr[0].axvline(self.sample, linestyle = '--', color='white')
            axarr[0].axvline(self.delay, linestyle = '--', color='white')
            axarr[0].axvline(self.response, linestyle = '--', color='white')
            
        
            axarr[1].axvline(self.sample, linestyle = '--', color='lightgrey')
            axarr[1].axvline(self.delay, linestyle = '--', color='lightgrey')
            axarr[1].axvline(self.response, linestyle = '--', color='lightgrey')
            
            axarr[0].set_title(title)
            
            axarr[1].set_ylabel('Spike rate (Hz)')
            axarr[1].set_xlabel('Time (s)')
            
                        
            if len(save) != 0:
                plt.savefig(save)
                
            plt.show()
            
        elif opto and stimside != "both":
        
            title = "Neuron {}: Control".format(neuron)
            
            f, axarr = plt.subplots(2,2, sharex='col', figsize=(10,6))
            
            #RASTER:
            counter = 0
            for i in R_correct_trials:
                axarr[0, 0].scatter(self.spks[neuron][0, i], 
                            np.ones(len(self.spks[neuron][0, i])) * counter, 
                            color='blue', s=5)
                counter += 1
            for i in L_correct_trials:
                axarr[0, 0].scatter(self.spks[neuron][0, i], 
                            np.ones(len(self.spks[neuron][0, i])) * counter, 
                            color='red', s=5)
                counter += 1


                        
            axarr[0, 0].axis('off')

            #TRACES:
                
            L_av, time, left_err = self.get_PSTH(neuron, L_correct_trials, 
                                                 binsize=binsize, timestep=timestep)
            R_av, _, right_err = self.get_PSTH(neuron, R_correct_trials,
                                                  binsize=binsize, timestep=timestep)

            vmax = max(cat([R_av, L_av]))

            axarr[1,0].plot(time, L_av, 'r-')
            axarr[1,0].plot(time, R_av, 'b-')
            
            
            axarr[1,0].fill_between(time, L_av - left_err, 
                     L_av + left_err,
                     color=['#ffaeb1'])
            axarr[1,0].fill_between(time, R_av - right_err, 
                     R_av + right_err,
                     color=['#b4b2dc'])
            
            axarr[0,0].axvline(self.sample, linestyle = '--', color='white')
            axarr[0,0].axvline(self.delay, linestyle = '--', color='white')
            axarr[0,0].axvline(self.response, linestyle = '--', color='white')
            
        
            axarr[1,0].axvline(self.sample, linestyle = '--', color='lightgrey')
            axarr[1,0].axvline(self.delay, linestyle = '--', color='lightgrey')
            axarr[1,0].axvline(self.response, linestyle = '--', color='lightgrey')
            
            
            axarr[1,0].set_ylabel('Spike rate (Hz)')
            axarr[1,0].set_xlabel('Time (s)')
            
            axarr[0,0].set_title(title)
            
            #OPTO:
            title = "Neuron {}: Opto".format(neuron)

    
            L_opto_trials = self.trial_type_direction('l')
            L_opto_trials = [i for i in L_opto_trials if self.stim_ON[i]]# and i in stable_trials]
            R_opto_trials = self.trial_type_direction('r')
            R_opto_trials = [i for i in R_opto_trials if self.stim_ON[i]]# and i in stable_trials]
                        
            L_opto_trials = [i for i in L_opto_trials if self.stim_side[i] == stimside]
            R_opto_trials = [i for i in R_opto_trials if self.stim_side[i] == stimside]
            
            counter = 0
            for i in R_opto_trials:
                axarr[0, 1].scatter(self.spks[neuron][0, i], 
                            np.ones(len(self.spks[neuron][0, i])) * counter, 
                            color='blue', s=5)
                counter += 1
            for i in L_opto_trials:
                axarr[0, 1].scatter(self.spks[neuron][0, i], 
                            np.ones(len(self.spks[neuron][0, i])) * counter, 
                            color='red', s=5)
                counter += 1
                        
            axarr[0, 1].axis('off')
            
            L_av, time, left_err = self.get_PSTH(neuron, L_opto_trials, 
                                                 binsize=binsize, timestep=timestep)
            R_av, _, right_err = self.get_PSTH(neuron, R_opto_trials,
                                                  binsize=binsize, timestep=timestep)

            vmax_opto = max(cat([R_av, L_av]))

            axarr[1,1].plot(time, L_av, 'r-')
            axarr[1,1].plot(time, R_av, 'b-')
            
            
            axarr[1,1].fill_between(time, L_av - left_err, 
                     L_av + left_err,
                     color=['#ffaeb1'])
            axarr[1,1].fill_between(time, R_av - right_err, 
                     R_av + right_err,
                     color=['#b4b2dc'])
            
            axarr[0,1].axvline(self.sample, linestyle = '--', color='white')
            axarr[0,1].axvline(self.delay, linestyle = '--', color='white')
            axarr[0,1].axvline(self.response, linestyle = '--', color='white')
            
        
            axarr[1,1].axvline(self.sample, linestyle = '--', color='lightgrey')
            axarr[1,1].axvline(self.delay, linestyle = '--', color='lightgrey')
            axarr[1,1].axvline(self.response, linestyle = '--', color='lightgrey')
            
            if vmax > vmax_opto:
                axarr[1, 1].sharey(axarr[1, 0])  # Make Bottom-Right share y-axis with Bottom-Left
                if self.laser == 'blue':
                    axarr[1, 1].hlines(y=vmax, xmin=self.delay, xmax=self.delay + 1, linewidth=10, color='lightblue')
                elif self.laser == 'red':
                    axarr[1, 1].hlines(y=vmax, xmin=self.sample, xmax=self.delay + 1, linewidth=10, color='red')

            else:
                axarr[1, 0].sharey(axarr[1, 1])  # Make Bottom-Right share y-axis with Bottom-Left
                if self.laser == 'blue':
                    axarr[1, 1].hlines(y=vmax_opto, xmin=self.delay, xmax=self.delay + 1, linewidth=10, color='lightblue')
                elif self.laser == 'red':
                    axarr[1, 1].hlines(y=vmax_opto, xmin=self.sample, xmax=self.delay + 1, linewidth=10, color='red')

            axarr[0,1].set_title(title)

            plt.tight_layout()
            
            if len(save) != 0:
                plt.savefig(save)
            plt.show()
        else:
            
            print("Both side opto plot not implemented yet!")
            
    ## ANALYSIS FUNCTIONS 
    
    def get_epoch_selective(self, epoch, p = 0.0001,
                            lickdir=False, trialtype=False,
                            return_stat=False):
        """Identifies neurons that are selective in a given epoch
        
        Saves neuron list in self.selective_neurons as well.
        
        Parameters
        ----------
        epoch : tuple
            start, stop timepoints to evaluate selectivity
        p : int, optional
            P-value cutoff to be deemed selectivity (default 0.0001)
        bias : bool, optional
            If true, only use the bias trials to evaluate (default False)
        rtrials, ltrials: list, optional
            If provided, use these trials to evaluate selectivty
        return_stat : bool, optional
            If true, returns the t-statistic to use for ranking
        lickdir : bool, optional
            If True, use the lick direction instaed of correct only
            
        Returns
        -------
        list
            List of neurons that are selective
        list, optional
            T-statistic associated with neuron, positive if left selective, 
            negative if right selective
        """
        
        selective_neurons = []
        all_tstat = []
        
        rtrials=self.lick_correct_direction('r') if not lickdir else self.lick_actual_direction('r')
        ltrials=self.lick_correct_direction('l') if not lickdir else self.lick_actual_direction('l')
        
        if trialtype:
            rtrials=self.trial_type_direction('r') 
            ltrials=self.trial_type_direction('l') 

        for neuron in self.good_neurons: # Only look at non-noise neurons
            
            # stable_trials = self.stable_trials[neuron]

            rtrials = [i for i in rtrials if not self.stim_ON[i] ]#and i in stable_trials]
            ltrials = [i for i in ltrials if not self.stim_ON[i] ]#and i in stable_trials]

            right = self.get_spike_count(neuron, epoch, rtrials)
            left = self.get_spike_count(neuron, epoch, ltrials)

            tstat, p_val = stats.ttest_ind(left, right)
            # p = 0.01/self.num_neurons
            # p = 0.01
            # p = 0.0001
            if p_val < p:
                selective_neurons += [neuron]
                all_tstat += [tstat] # Positive if L selective, negative if R selective

        self.selective_neurons = selective_neurons
        
        if return_stat:
            return selective_neurons, all_tstat
        
        return selective_neurons
    
    def get_number_selective(self, epoch, mode, p=0.05):
        """
        Get the number of selective neurons in a given epoch for a give mode
        using a simple t-test

        Parameters
        ----------
        epoch : TYPE
            DESCRIPTION.
        mode : string
            'Stimulus' or 'Choice' will use diff trial type groupings.
        p : TYPE, optional
            DESCRIPTION. The default is 0.05.

        Returns
        -------
        Number of selective neurons per epoch.

        """
        if mode=='stimulus':
            n = self.get_epoch_selective(epoch, p=p, trialtype=True)
            
        elif mode=='choice':
            n = self.get_epoch_selective(epoch, p=p, lickdir=True)

        else:
            print('invalid mode {}'.format(mode))
            return
        
        return len(n)
    
    def shuffle_test_fast(self, spikes_A, spikes_B, n_shuffles=1000):
        """Fast shuffle test using precomputed spike arrays (1D arrays)
            Spikes_A is always the preferred direction
        
        """
        # all_spikes = np.concatenate([spikes_A, spikes_B])
        # n_A = len(spikes_A)
        # n_B = len(spikes_B)
        # total = n_A + n_B
        prob_groupA, total_groupA = self.compute_preference_probability(spikes_A[0], spikes_B[0])
        prob_groupB, total_groupB = self.compute_preference_probability(spikes_A[1], spikes_B[1])
    
        actual_prob = (prob_groupA + prob_groupB) / (total_groupA + total_groupB)
        # if n_A == 0 or n_B == 0:
        #     return np.nan, np.nan
    
        # Use numpy for fast shuffling
        shuffle_probs = np.empty(n_shuffles)
        for i in range(n_shuffles): # 1000 shuffles
            # Get a value for each comparison group
            numerator, denominator = 0,0
            for j in range(2):
                idx = np.random.permutation(len(spikes_A[j]) + len(spikes_B[j]))
                all_spikes = np.concatenate([spikes_A[j], spikes_B[j]])
                shuffled_A = all_spikes[idx[:len(spikes_A[j])]]
                shuffled_B = all_spikes[idx[len(spikes_A[j]):]]
                probgroup, totalgroup = self.compute_preference_probability(shuffled_A, shuffled_B)
                numerator += probgroup
                denominator += totalgroup
                
            shuffle_probs[i] = numerator / denominator
            
        p_val = np.mean(shuffle_probs >= actual_prob)
        return actual_prob, p_val
    
                
    def compute_preference_probability(self, group_A, group_B):
        A = np.array(group_A)
        B = np.array(group_B)
        total = len(A) * len(B)
        if total == 0:
            return np.nan, None
    
        comp_A = np.sum(A[:, None] > B[None, :]) 
    
        return comp_A, total
        # comp_B = np.sum(B[:, None] > A[None, :]) / total
    
        # if comp_A >= comp_B:
        #     return comp_A, 'A'
        # else:
        #     return comp_B, 'B'

        
    def shuffle_test(self, neuron, trials_A, trials_B, time_bin, n_shuffles=1000):
        all_trials = np.array(trials_A + trials_B)
        labels = np.array([1] * len(trials_A) + [0] * len(trials_B))
    
        spikes_A = self.get_spike_count(neuron, time_bin, trials_A)
        spikes_B = self.get_spike_count(neuron, time_bin, trials_B)
    
        actual_prob, _ = self.compute_preference_probability(spikes_A, spikes_B)
    
        shuffle_probs = []
        for _ in range(n_shuffles):
            np.random.shuffle(labels)
            shuffled_A = all_trials[labels == 1]
            shuffled_B = all_trials[labels == 0]
            spikes_A_shuf = self.get_spike_count(neuron, time_bin, shuffled_A)
            spikes_B_shuf = self.get_spike_count(neuron, time_bin, shuffled_B)
            prob_shuf, _ = self.compute_preference_probability(spikes_A_shuf, spikes_B_shuf)
            shuffle_probs.append(prob_shuf)
    
        p_val = np.mean(np.array(shuffle_probs) >= actual_prob)
        return actual_prob, p_val

        
    def count_significant_neurons_by_time(self, neurons, mode='stimulus', alpha=0.05, n_shuffles=1000):
        """Return array of significant neuron counts per time bin        
            Get the number of selective neurons in a given epoch for a give mode
            using a shuffle test (Chen et al 2021)
    
            Parameters
            ----------
            epoch : TYPE
                DESCRIPTION.
            mode : string
                'Stimulus' or 'Choice' will use diff trial type groupings.
            p : TYPE, optional
                DESCRIPTION. The default is 0.05.
    
            Returns
            -------
            Number of selective neurons per epoch.
            a given mode."""
        
        neurons = np.array(neurons)
        self.CR_trials = np.array([i for i in np.where(self.R_correct)[0] if i in self.i_good_non_stim_trials])
        self.EL_trials = np.array([i for i in np.where(self.L_wrong)[0] if i in self.i_good_non_stim_trials])
        self.ER_trials = np.array([i for i in np.where(self.R_wrong)[0] if i in self.i_good_non_stim_trials])
        self.CL_trials = np.array([i for i in np.where(self.L_correct)[0] if i in self.i_good_non_stim_trials])
    
        if len(self.CR_trials) < 6 or len(self.EL_trials) < 6 or len(self.ER_trials) < 6 or len(self.CL_trials) < 6:
            print('Fewer than 5 trials in group for {}'.format(self.path))
            return []
        
        hold_out_trial_num = int(min([len(self.CR_trials), len(self.EL_trials), len(self.ER_trials), len(self.CL_trials)]) / 2)
        CR_indices = np.random.choice(range(len(self.CR_trials)), hold_out_trial_num, replace=False)
        EL_indices = np.random.choice(range(len(self.EL_trials)), hold_out_trial_num, replace=False)
        ER_indices = np.random.choice(range(len(self.ER_trials)), hold_out_trial_num, replace=False)
        CL_indices = np.random.choice(range(len(self.CL_trials)), hold_out_trial_num, replace=False)
        
        self.CR_out_trials = self.CR_trials[CR_indices]
        self.EL_out_trials = self.EL_trials[EL_indices]
        self.ER_out_trials = self.ER_trials[ER_indices]
        self.CL_out_trials = self.CL_trials[CL_indices]

        self.CR_in_trials = np.setdiff1d(self.CR_trials, self.CR_out_trials, assume_unique=True)
        self.EL_in_trials = np.setdiff1d(self.EL_trials, self.EL_out_trials, assume_unique=True)
        self.ER_in_trials = np.setdiff1d(self.ER_trials, self.ER_out_trials, assume_unique=True)
        self.CL_in_trials = np.setdiff1d(self.CL_trials, self.CL_out_trials, assume_unique=True)

        mode_pairs = {
               'stimulus': [('CR', 'EL'), ('ER', 'CL')],
               'choice': [('CR', 'ER'), ('EL', 'CL')],
               'outcome': [('CR', 'ER'), ('CL', 'EL')],
           }
            
        if mode not in mode_pairs:
            raise ValueError(f"Unsupported mode: {mode}")
    
        comparisons = mode_pairs[mode]
        self.time_bins =  [(i, i+0.2) for i in np.arange(-0.2, self.time_cutoff, 0.2)]
        sig_counts = np.zeros(len(self.time_bins), dtype=int)
    
        for time_idx, time_window in enumerate(self.time_bins):
            sig_neurons = 0
    
            # Cache spike counts per neuron per trial group for this time window
            spike_cache = {
                neuron: {
                    group_name: np.array(self.get_spike_count(neuron, time_window, getattr(self, f"{group_name}_in_trials")))
                    for group_name in ['CR', 'EL', 'ER', 'CL']
                } for neuron in neurons
            }
            
            spike_cache_out = {
                neuron: {
                    group_name: np.array(self.get_spike_count(neuron, time_window, getattr(self, f"{group_name}_out_trials")))
                    for group_name in ['CR', 'EL', 'ER', 'CL']
                } for neuron in neurons
            }
            
            for neuron in neurons:
                                
                spikes_A_out = cat([spike_cache_out[neuron][group_A_name] for group_A_name, _ in comparisons])
                spikes_B_out = cat([spike_cache_out[neuron][group_B_name] for _, group_B_name in comparisons]) 
                
                spikes_A = [spike_cache[neuron][group_A_name] for group_A_name, _ in comparisons]
                spikes_B = [spike_cache[neuron][group_B_name] for _, group_B_name in comparisons]
                
                if sum(spikes_A_out) < sum(spikes_B_out): # Switch pref
                    spikes_A, spikes_B = spikes_B, spikes_A    
                
                _, p_val = self.shuffle_test_fast(spikes_A, spikes_B, n_shuffles=n_shuffles)

                if not np.isnan(p_val) and p_val < alpha:
                    sig_neurons += 1
                    # break  # move on to next neuron once one comparison is significant
    
            sig_counts[time_idx] = sig_neurons
    
        return sig_counts


            
    
    def screen_preference(self, neuron_num, epoch, bootstrap=False, samplesize = 20, 
                          lickdir=False, return_remove=False):
        """Determine if a neuron is left or right preferring
                
        Iterate 30 times over different test batches to get a high confidence
        estimation of neuron preference.
        
        Parameters
        ----------
        neuron_num : int
            Neuron to screen in function
        epoch : list
            Timesteps over which to evaluate the preference
        samplesize : int, optional
            Number of trials to use in the test batch (default 10)
        return_remove : bool, optional
            Return trials to mreove (train trials)
            
        Returns
        -------
        choice : bool
            True if left preferring, False if right preferring
        l_trials, r_trials : list
            All left and right trials        
        """
        # Input: neuron of interest
        # Output: (+) if left pref, (-) if right pref, then indices of trials to plot
        
        # All trials where the mouse licked left or right AND non stim
      
        if lickdir:
            r_trials = self.lick_actual_direction('r')
            l_trials = self.lick_actual_direction('l')
        else:
            r_trials = self.lick_correct_direction('r')
            l_trials = self.lick_correct_direction('l')
            
        r_trials = [i for i in r_trials if not self.stim_ON[i] and i in self.stable_trials[neuron_num]]
        l_trials = [i for i in l_trials if not self.stim_ON[i] and i in self.stable_trials[neuron_num]]
        
        # Skip neuron if less than 15
        if len(l_trials) < samplesize or len(r_trials) < samplesize:
            print("There are fewer than {} trials R/L: {} R trials and {} L trials".format(samplesize, len(l_trials), len(r_trials)))
            samplesize = min(len(l_trials), len(r_trials))
        
        if bootstrap:
            pref = 0
            for _ in range(30): # Perform 30 times
                # Pick 20 random trials as screen for left and right
                screen_l = np.random.choice(l_trials, size = samplesize, replace = False)
                screen_r = np.random.choice(r_trials, size = samplesize, replace = False)
                
                # Compare late delay epoch for preference
                avg_r = np.mean(self.get_spike_count(neuron_num, epoch, screen_r))
                avg_l = np.mean(self.get_spike_count(neuron_num, epoch, screen_l))
              
                pref += avg_l > avg_r
                
            choice = True if pref/30 > 0.5 else False

            return choice, l_trials, r_trials
            
            
        # Pick 20 random trials as screen for left and right
        screen_l = np.random.choice(l_trials, size = samplesize, replace = False)
        screen_r = np.random.choice(r_trials, size = samplesize, replace = False)
    
        # Remainder of trials are left for plotting in left and right separately
        test_l = [t for t in l_trials if t not in screen_l]
        test_r = [t for t in r_trials if t not in screen_r]
        
        # Compare late delay epoch for preference
        avg_r = np.mean(self.get_spike_count(neuron_num, epoch, screen_r))
        avg_l = np.mean(self.get_spike_count(neuron_num, epoch, screen_l))

        if return_remove:

            return avg_l > avg_r, screen_l, screen_r

        return avg_l > avg_r, test_l, test_r

        

    
    ### SELECTIVITY PLOTS NO OPTO ###
    

    def plot_selectivity(self, neurons, plot=True, 
                         epoch=None, opto=False,
                         binsize=50, timestep=1,
                         downsample=False, bootstrap = False, 
                         correct_only=True,
                         lickdir=False, return_pref_np = True):
    
        """
        Plots or returns a single line representing selectivity of given 
        neuron over all trials
        
        Evaluates the selectivity using preference in delay epoch
        
        Parameters
        ----------
        neurons : int
            Neurons to plot
        plot : bool, optional
            Whether to plot or not (default True)
        epoch : list, optional
            Start and stop times for epoch calculations
            
        Returns
        -------
        list
            Selectivity calculated and plotted
        """
        
        # x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]
        if epoch is None:
            epoch = (self.response-1.5, self.response) # Late delay
            
        
        def process_neurons(n, e):
            n_pref, n_nonpref = [],[]
            sel = []
            # start = time_.time()

            for _ in range(30):
    
                L_pref, screenl, screenr = self.screen_preference(n, e, bootstrap=bootstrap)
    
                if correct_only:
                    
                    l_control_trials = self.lick_correct_direction('l') 
                    r_control_trials = self.lick_correct_direction('r')
                    
                else:
                    
                    l_control_trials = self.trial_type_direction('l') if not lickdir else self.lick_actual_direction('l')            
                    r_control_trials = self.trial_type_direction('r') if not lickdir else self.lick_actual_direction('r')
            
            
                all_exclude_trials = cat((screenl, screenr))
                l_control_trials = [i for i in l_control_trials if i not in all_exclude_trials]
                r_control_trials = [i for i in r_control_trials if i not in all_exclude_trials]
    
                if opto:
                    l_control_trials = [i for i in l_control_trials if i in self.stable_trials[n] and self.stim_ON[i]]
                    r_control_trials = [i for i in r_control_trials if i in self.stable_trials[n] and self.stim_ON[i]]    
                else:
                    l_control_trials = [i for i in l_control_trials if i in self.stable_trials[n] and not self.stim_ON[i]]
                    r_control_trials = [i for i in r_control_trials if i in self.stable_trials[n] and not self.stim_ON[i]] 
                    
                if L_pref:
                    pref, time,_ = self.get_PSTH(n, l_control_trials, binsize=binsize, timestep=timestep)
                    nonpref,_,_ = self.get_PSTH(n, r_control_trials, binsize=binsize, timestep=timestep)
    
                else:
                    pref, time,_ = self.get_PSTH(n, r_control_trials, binsize=binsize, timestep=timestep)
                    nonpref,_,_ = self.get_PSTH(n, l_control_trials, binsize=binsize, timestep=timestep)
    
                if return_pref_np:
                    n_pref += [pref]
                    n_nonpref += [nonpref]
                else:
                    sel += [pref-nonpref]
  
            if return_pref_np:

                return np.mean(n_pref, axis=0), np.mean(n_nonpref, axis=0)
            
            else:
                
                return np.mean(sel, axis=0)
        
        # Late delay selective neurons

        if bootstrap:
            allpref, allnonpref = [], []

            for n in neurons:
                L_pref, screenl, screenr = self.screen_preference(n, epoch, bootstrap=bootstrap)
                
                l_control_trials = self.trial_type_direction('l') if not lickdir else self.lick_actual_direction('l')            
                r_control_trials = self.trial_type_direction('r') if not lickdir else self.lick_actual_direction('r')
      
                l_control_trials = [i for i in l_control_trials if i in self.stable_trials[n] and not self.stim_ON[i]]
                r_control_trials = [i for i in r_control_trials if i in self.stable_trials[n] and not self.stim_ON[i]]    
                
                if L_pref:
                    pref, time,_ = self.get_PSTH(n, l_control_trials, binsize=binsize, timestep=timestep)
                    nonpref,_,_ = self.get_PSTH(n, r_control_trials, binsize=binsize, timestep=timestep)
    
                else:
                    pref, time,_ = self.get_PSTH(n, r_control_trials, binsize=binsize, timestep=timestep)
                    nonpref,_,_ = self.get_PSTH(n, l_control_trials, binsize=binsize, timestep=timestep)
    
    
                allpref += [pref]
                allnonpref += [nonpref]
        else:
            results = []
            executor = ThreadPoolExecutor(max_workers=2)
            iterator = executor.map(process_neurons, neurons, epoch)
            
            iterator = tqdm(iterator, total=len(neurons), desc="Computing selectivity")
        
            if return_pref_np:
                for pref, nonpref in iterator:
                    if pref is not None and nonpref is not None:
                        results.append((pref, nonpref))
            
                allpref = [r[0] for r in results]
                allnonpref = [r[1] for r in results]
            else:
                for sel in iterator:
                    if sel is not None:
                        results.append((sel))
                            

        if return_pref_np:
            return allpref, allnonpref, self.time
        else:
            return results, self.time


    
    ### OPTO SELECTIVITY PLOTS ### 
    
    def selectivity_optogenetics(self, save=False, p = 0.0001, lickdir = False, 
                                 return_traces = False, exclude_unselective=False,
                                 binsize=50, timestep=1, epoch = None,
                                 fix_axis = [], selective_neurons = [], downsample=False,
                                 bootstrap=False, return_modularity=False):
        
        """Returns overall selectivity trace across opto vs control trials and L / R hemi
        
        Uses late delay epoch to calculate selectivity
                                
        Parameters
        ----------
        save : bool, optional
            Whether to save fig to file (default False)
        p : int, optional
            P-value to use in the selectivity calculations
        fix_axis : tuple, optional
            Provide top and bottom limits for yaxis

        selective_neurons : list, optional        
            List of selective neurons to plot from
        """


        # x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]
        if epoch is None:
            epoch = (self.response-1.5, self.response) # Late delay
        
        # Late delay selective neurons
        delay_neurons = self.get_epoch_selective(epoch, p=p)
        if len(delay_neurons) == 0: # No selective neurons
            print('No selective neurons')
            return None, None
        
        left_delay_neurons = [n for n in delay_neurons if n in self.L_alm_idx]
        right_delay_neurons = [n for n in delay_neurons if n in self.R_alm_idx]

        control_sel = []
        opto_sel_stim_left = []
        opto_sel_stim_right = []

        for n in left_delay_neurons:
                
            L_pref, screenl, screenr = self.screen_preference(n, epoch, bootstrap=bootstrap)
            
            l_control_trials = self.trial_type_direction('l') if not lickdir else self.lick_actual_direction('l')            
            r_control_trials = self.trial_type_direction('r') if not lickdir else self.lick_actual_direction('r')
            
            if not bootstrap:
                all_exclude_trials = cat((screenl, screenr))
                l_control_trials = [i for i in l_control_trials if i not in all_exclude_trials]
                r_control_trials = [i for i in r_control_trials if i not in all_exclude_trials]
                
            l_opto_trials = [i for i in l_control_trials if i in self.i_good_stim_trials]
            r_opto_trials = [i for i in r_control_trials if i in self.i_good_stim_trials]
            
            l_opto_trials_stim_left = [i for i in l_opto_trials if self.stim_side[i] == 'L']
            r_opto_trials_stim_left = [i for i in r_opto_trials if self.stim_side[i] == 'L']
            
            l_opto_trials_stim_right = [i for i in l_opto_trials if self.stim_side[i] == 'R'] 
            r_opto_trials_stim_right = [i for i in r_opto_trials if self.stim_side[i] == 'R'] 
            
            l_control_trials = [i for i in l_control_trials if i in self.stable_trials[n] and not self.stim_ON[i]]
            r_control_trials = [i for i in r_control_trials if i in self.stable_trials[n] and not self.stim_ON[i]]    
        
            if L_pref:
                pref, time,_ = self.get_PSTH(n, l_control_trials, binsize=binsize, timestep=timestep)
                nonpref,_,_ = self.get_PSTH(n, r_control_trials, binsize=binsize, timestep=timestep)
                optop_stim_left,_,_ = self.get_PSTH(n, l_opto_trials_stim_left, binsize=binsize, timestep=timestep)
                optonp_stim_left,_,_ = self.get_PSTH(n, r_opto_trials_stim_left, binsize=binsize, timestep=timestep)
                optop_stim_right,_,_ = self.get_PSTH(n, l_opto_trials_stim_right, binsize=binsize, timestep=timestep)
                optonp_stim_right,_,_ = self.get_PSTH(n, r_opto_trials_stim_right, binsize=binsize, timestep=timestep)
                
            else:
                pref, time,_ = self.get_PSTH(n, r_control_trials, binsize=binsize, timestep=timestep)
                nonpref,_,_ = self.get_PSTH(n, l_control_trials, binsize=binsize, timestep=timestep)
                optop_stim_left,_,_ = self.get_PSTH(n, r_opto_trials_stim_left, binsize=binsize, timestep=timestep)
                optonp_stim_left,_,_ = self.get_PSTH(n, l_opto_trials_stim_left, binsize=binsize, timestep=timestep)
                optop_stim_right,_,_ = self.get_PSTH(n, r_opto_trials_stim_right, binsize=binsize, timestep=timestep)
                optonp_stim_right,_,_ = self.get_PSTH(n, l_opto_trials_stim_right, binsize=binsize, timestep=timestep)
                
            control_sel += [pref-nonpref]
            opto_sel_stim_left += [optop_stim_left-optonp_stim_left]
            if self.laser == 'blue':
                opto_sel_stim_right += [optop_stim_right-optonp_stim_right]
            
        control_sel_R = []
        opto_sel_stim_left_R = []
        opto_sel_stim_right_R = []

        for n in right_delay_neurons:
                
            L_pref, screenl, screenr = self.screen_preference(n, epoch, bootstrap=bootstrap)
            
            l_control_trials = self.trial_type_direction('l') if not lickdir else self.lick_actual_direction('l')            
            r_control_trials = self.trial_type_direction('r') if not lickdir else self.lick_actual_direction('r')
            
            if not bootstrap:
                all_exclude_trials = cat((screenl, screenr))
                l_control_trials = [i for i in l_control_trials if i not in all_exclude_trials]
                r_control_trials = [i for i in r_control_trials if i not in all_exclude_trials]
                
            l_opto_trials = [i for i in l_control_trials if i in self.i_good_stim_trials]
            r_opto_trials = [i for i in r_control_trials if i in self.i_good_stim_trials]
            
            l_opto_trials_stim_left = [i for i in l_opto_trials if self.stim_side[i] == 'L']
            r_opto_trials_stim_left = [i for i in r_opto_trials if self.stim_side[i] == 'L']
            
            l_opto_trials_stim_right = [i for i in l_opto_trials if self.stim_side[i] == 'R'] 
            r_opto_trials_stim_right = [i for i in r_opto_trials if self.stim_side[i] == 'R'] 
            
            l_control_trials = [i for i in l_control_trials if i in self.stable_trials[n] and not self.stim_ON[i]]
            r_control_trials = [i for i in r_control_trials if i in self.stable_trials[n] and not self.stim_ON[i]]    
        
            if L_pref:
                pref, time,_ = self.get_PSTH(n, l_control_trials, binsize=binsize, timestep=timestep)
                nonpref,_,_ = self.get_PSTH(n, r_control_trials, binsize=binsize, timestep=timestep)
                optop_stim_left,_,_ = self.get_PSTH(n, l_opto_trials_stim_left, binsize=binsize, timestep=timestep)
                optonp_stim_left,_,_ = self.get_PSTH(n, r_opto_trials_stim_left, binsize=binsize, timestep=timestep)
                optop_stim_right,_,_ = self.get_PSTH(n, l_opto_trials_stim_right, binsize=binsize, timestep=timestep)
                optonp_stim_right,_,_ = self.get_PSTH(n, r_opto_trials_stim_right, binsize=binsize, timestep=timestep)
                
            else:
                pref, time,_ = self.get_PSTH(n, r_control_trials, binsize=binsize, timestep=timestep)
                nonpref,_,_ = self.get_PSTH(n, l_control_trials, binsize=binsize, timestep=timestep)
                optop_stim_left,_,_ = self.get_PSTH(n, r_opto_trials_stim_left, binsize=binsize, timestep=timestep)
                optonp_stim_left,_,_ = self.get_PSTH(n, l_opto_trials_stim_left, binsize=binsize, timestep=timestep)
                optop_stim_right,_,_ = self.get_PSTH(n, r_opto_trials_stim_right, binsize=binsize, timestep=timestep)
                optonp_stim_right,_,_ = self.get_PSTH(n, l_opto_trials_stim_right, binsize=binsize, timestep=timestep)
                
            control_sel_R += [pref-nonpref]
            opto_sel_stim_left_R += [optop_stim_left-optonp_stim_left]
            if self.laser == 'blue':
                opto_sel_stim_right_R += [optop_stim_right-optonp_stim_right]
            
        if exclude_unselective:
            time_steps = np.where((time >= self.delay+1.5) & (time <= self.response))
            keep_n = [c for c in range(len(control_sel)) if np.mean(np.array(control_sel[c])[time_steps]) > 3] # Spike rate diff FIXME
            control_sel = np.array(control_sel)[keep_n]
            opto_sel_stim_left = np.array(opto_sel_stim_left)[keep_n]
            opto_sel_stim_right = np.array(opto_sel_stim_right)[keep_n]
            
        sel = np.mean(control_sel, axis=0)
        selo_stimleft = np.mean(opto_sel_stim_left, axis=0)
        selo_stimright = np.mean(opto_sel_stim_right, axis=0)

        err = np.std(control_sel, axis=0) / np.sqrt(len(left_delay_neurons))
        erro_stimleft = np.std(opto_sel_stim_left, axis=0) / np.sqrt(len(left_delay_neurons))
        erro_stimright = np.std(opto_sel_stim_right, axis=0) / np.sqrt(len(left_delay_neurons))
        
        
        
        sel_R = np.mean(control_sel_R, axis=0)
        selo_stimleft_R = np.mean(opto_sel_stim_left_R, axis=0)
        selo_stimright_R = np.mean(opto_sel_stim_right_R, axis=0)

        err_R = np.std(control_sel_R, axis=0) / np.sqrt(len(right_delay_neurons))
        erro_stimleft_R = np.std(opto_sel_stim_left_R, axis=0) / np.sqrt(len(right_delay_neurons))
        erro_stimright_R = np.std(opto_sel_stim_right_R, axis=0) / np.sqrt(len(right_delay_neurons))

        
        
        if return_traces: # return granular version for aggregating across FOVs
            # Left alm, right alm
            return ([control_sel, opto_sel_stim_left, opto_sel_stim_right, time],
                    [control_sel_R, opto_sel_stim_left_R, opto_sel_stim_right_R, time]) 
        
        else: # Single FOV view
            # Left alm, right alm
            return ([sel, selo_stimleft, selo_stimright, err, erro_stimleft, erro_stimright, time],
                    [sel_R, selo_stimleft_R, selo_stimright_R, err_R, erro_stimleft_R, erro_stimright_R, time])
    
    def selectivity_optogenetics_by_hemi(self, save=False, p = 0.0001, lickdir = False, 
                                 return_traces = False, exclude_unselective=False,
                                 binsize=50, timestep=1, epoch = None,
                                 fix_axis = [], selective_neurons = [], downsample=False,
                                 bootstrap=False):
        
        """Returns overall selectivity trace across opto vs control trials
        
        Uses late delay epoch to calculate selectivity
                                
        Parameters
        ----------
        save : bool, optional
            Whether to save fig to file (default False)
        p : int, optional
            P-value to use in the selectivity calculations
        fix_axis : tuple, optional
            Provide top and bottom limits for yaxis

        selective_neurons : list, optional        
            List of selective neurons to plot from
        """
        


        # x = np.arange(-6.97,4,self.fs)[:self.time_cutoff]
        if epoch is None:
            epoch = (self.response-1.5, self.response) # Late delay
        
        # Late delay selective neurons
        delay_neurons = self.get_epoch_selective(epoch, p=p)

        control_sel = []
        opto_sel_stim_left = []
        opto_sel_stim_right = []
                      
        if len(delay_neurons) == 0: # No selective neurons
            print('No selective neurons')
            return None, None
        
        for n in delay_neurons:
                
            L_pref, screenl, screenr = self.screen_preference(n, epoch, bootstrap=bootstrap)
            
            l_control_trials = self.trial_type_direction('l') if not lickdir else self.lick_actual_direction('l')            
            r_control_trials = self.trial_type_direction('r') if not lickdir else self.lick_actual_direction('r')
            
            if not bootstrap:
                all_exclude_trials = cat((screenl, screenr))
                l_control_trials = [i for i in l_control_trials if i not in all_exclude_trials]
                r_control_trials = [i for i in r_control_trials if i not in all_exclude_trials]
                
            l_opto_trials = [i for i in l_control_trials if i in self.i_good_stim_trials]
            r_opto_trials = [i for i in r_control_trials if i in self.i_good_stim_trials]
            
            l_opto_trials_stim_left = [i for i in l_opto_trials if self.stim_side[i] == 'L']
            r_opto_trials_stim_left = [i for i in r_opto_trials if self.stim_side[i] == 'L']
            
            l_opto_trials_stim_right = [i for i in l_opto_trials if self.stim_side[i] == 'R'] 
            r_opto_trials_stim_right = [i for i in r_opto_trials if self.stim_side[i] == 'R'] 
            
            l_control_trials = [i for i in l_control_trials if i in self.stable_trials[n] and not self.stim_ON[i]]
            r_control_trials = [i for i in r_control_trials if i in self.stable_trials[n] and not self.stim_ON[i]]    
        
            if L_pref:
                pref, time,_ = self.get_PSTH(n, l_control_trials, binsize=binsize, timestep=timestep)
                nonpref,_,_ = self.get_PSTH(n, r_control_trials, binsize=binsize, timestep=timestep)
                optop_stim_left,_,_ = self.get_PSTH(n, l_opto_trials_stim_left, binsize=binsize, timestep=timestep)
                optonp_stim_left,_,_ = self.get_PSTH(n, r_opto_trials_stim_left, binsize=binsize, timestep=timestep)
                optop_stim_right,_,_ = self.get_PSTH(n, l_opto_trials_stim_right, binsize=binsize, timestep=timestep)
                optonp_stim_right,_,_ = self.get_PSTH(n, r_opto_trials_stim_right, binsize=binsize, timestep=timestep)
                
            else:
                pref, time,_ = self.get_PSTH(n, r_control_trials, binsize=binsize, timestep=timestep)
                nonpref,_,_ = self.get_PSTH(n, l_control_trials, binsize=binsize, timestep=timestep)
                optop_stim_left,_,_ = self.get_PSTH(n, r_opto_trials_stim_left, binsize=binsize, timestep=timestep)
                optonp_stim_left,_,_ = self.get_PSTH(n, l_opto_trials_stim_left, binsize=binsize, timestep=timestep)
                optop_stim_right,_,_ = self.get_PSTH(n, r_opto_trials_stim_right, binsize=binsize, timestep=timestep)
                optonp_stim_right,_,_ = self.get_PSTH(n, l_opto_trials_stim_right, binsize=binsize, timestep=timestep)
                
            control_sel += [pref-nonpref]
            opto_sel_stim_left += [optop_stim_left-optonp_stim_left]
            opto_sel_stim_right += [optop_stim_right-optonp_stim_right]
            
         
            
        if exclude_unselective:
            time_steps = np.where((time >= self.delay+1.5) & (time <= self.response))
            keep_n = [c for c in range(len(control_sel)) if np.mean(np.array(control_sel[c])[time_steps]) > 3] # Spike rate diff FIXME
            control_sel = np.array(control_sel)[keep_n]
            opto_sel_stim_left = np.array(opto_sel_stim_left)[keep_n]
            opto_sel_stim_right = np.array(opto_sel_stim_right)[keep_n]
            
        sel = np.mean(control_sel, axis=0)
        selo_stimleft = np.mean(opto_sel_stim_left, axis=0)
        selo_stimright = np.mean(opto_sel_stim_right, axis=0)

        err = np.std(control_sel, axis=0) / np.sqrt(len(delay_neurons))
        erro_stimleft = np.std(opto_sel_stim_left, axis=0) / np.sqrt(len(delay_neurons))
        erro_stimright = np.std(opto_sel_stim_right, axis=0) / np.sqrt(len(delay_neurons))
        
        
        if return_traces: # return granular version for aggregating across FOVs

            return control_sel, opto_sel_stim_left, opto_sel_stim_right, time #, err, erro_stimleft, erro_stimright, time
    
        
        else: # Single FOV view
        
            return sel, selo_stimleft, selo_stimright, err, erro_stimleft, erro_stimright, time

    def susceptibility(self, stimside, p=0.01,
                       period=None, return_n=False,
                       by_trial_type = False,
                       provide_trials = [],
                       binsize=400, timestep=10):
        """
        Calculates the per neuron susceptibility to perturbation, measured as a
        simple difference between control/opto trials during the specified period

        Returns
        -------
        all_sus : one positive value for every good neuron
        p_value : provide a significance measure

        """
        if period is None:
            period = (self.delay, self.response)
            
        all_sus = []
        sig_p = [] 
        sig_n = []
        all_tstat = []
        
        for n in self.good_neurons:
            if by_trial_type: # FIXME: should be trial-wise
                control_trials = [t for t in self.L_trials if t in self.i_good_non_stim_trials]
                pert_trials = [t for t in self.L_trials if t in self.i_good_stim_trials]
                
                control_left, time, _ = self.get_PSTH(n, control_trials, period=period, binsize=binsize, timestep=timestep)
                pert_left, _, _ = self.get_PSTH(n, pert_trials, period=period, binsize=binsize, timestep=timestep)
                diff = np.abs(control_left - pert_left)
                
                control_trials = [t for t in self.R_trials if t in self.i_good_non_stim_trials]
                pert_trials = [t for t in self.R_trials if t in self.i_good_stim_trials]
    
                control, _, _ = self.get_PSTH(n, control_trials, period=period, binsize=binsize, timestep=timestep)
                pert, _, _ = self.get_PSTH(n, pert_trials, period=period, binsize=binsize, timestep=timestep)
                diff += np.abs(control - pert)
                
                all_sus += [np.sum(diff)]
    
                tstat_left, p_val_left = stats.ttest_ind(control_left, pert_left)
                tstat_right, p_val_right = stats.ttest_ind(control, pert)
                
                if p_val_left < p or p_val_right < p:
                    sig_p += [1]
                    sig_n += [n]
                else:
                    sig_p += [0]
            else:
                if len(provide_trials) == 0:
                    control_trials = [t for t in self.i_good_non_stim_trials]
                    pert_trials = [t for t in self.i_good_stim_trials]
                else:
                    
                    control_trials, pert_trials = provide_trials
                    
                control_left = self.get_spike_rate(n, period, control_trials, mean=False)
                pert_left = self.get_spike_rate(n, period, pert_trials, mean=False)
                # diff = np.abs(control_left - pert_left)
                
                # all_sus += [np.sum(diff)]
    
                tstat_left, p_val_left = stats.ttest_ind(control_left, pert_left)
                
                if p_val_left < p:
                    sig_p += [1]
                    sig_n += [n]
                    
                    all_tstat += [tstat_left]
                    
                else:
                    sig_p += [0]
                
        if return_n:
            return sig_n, all_tstat
        return all_sus, sig_p
            
            
  
    ## OLD EPHYS PLOTS UNEDITED
    
    def plot_number_of_sig_neurons(self, window = 200, p=0.01, return_nums=False, save=False, y_axis = []):
        
        """Plots number of contra / ipsi neurons over course of trial
                                
        Parameters
        ----------
        return_nums : bool, optional
            return number of contra ispi neurons to do an aggregate plot
        
        save : bool, optional
            Whether to save fig to file (default False)
            
        y_axis : list, optional
            set top and bottom ylim
        """
        p_value = p

        # x = np.arange(-6.97,6,self.fs)[:self.time_cutoff]
        x = np.arange(0, 8.5, window/1000)
        steps = np.arange(len(x))
        contra = np.zeros(len(steps))
        ipsi = np.zeros(len(steps))

        for n in self.good_neurons:
            
            stable_trials = self.stable_trials[n]
            
            L_correct_trials = self.lick_correct_direction('l')
            L_correct_trials = [i for i in L_correct_trials if not self.stim_ON[i] and i in stable_trials]
            R_correct_trials = self.lick_correct_direction('r')
            R_correct_trials = [i for i in R_correct_trials if not self.stim_ON[i] and i in stable_trials]
            
            for t in steps:
                
                r = self.get_spike_count(n, (x[t], x[t]+window/1000), R_correct_trials)
                l = self.get_spike_count(n, (x[t], x[t]+window/1000), L_correct_trials)
                
                t_val, p = stats.ttest_ind(r, l)
                p = p < p_value
                
                if self.unit_side[n] == 'L':
                    if t_val > 0: # R > L
                        contra[t] += p
                    else:
                        ipsi[t] += p
                else:
                    if t_val > 0: # R > L
                        ipsi[t] += p
                    else:
                        contra[t] += p

        
        if return_nums:
            return contra, ipsi

        plt.bar(x, contra, color = 'b', edgecolor = 'white', width = window/1000, label = 'contra')
        plt.bar(x, -ipsi, color = 'r',edgecolor = 'white', width = window/1000, label = 'ipsi')
        plt.axvline(self.sample, ls = '--', color='grey')
        plt.axvline(self.delay, ls = '--', color='grey')
        plt.axvline(self.response, ls = '--', color='grey')

        if len(y_axis) != 0:
            plt.ylim(bottom = y_axis[0])
            plt.ylim(top = y_axis[1])
        plt.ylabel('Number of sig sel neurons')
        plt.xlabel('Time from Go cue (s)')
        plt.legend()
        plt.title('{} ALM neurons'.format(self.side))
        
        if save:
            plt.savefig(self.path + r'number_sig_neurons.pdf')
        
        plt.show()
        
    def selectivity_table_by_epoch(self, save=False):
        """Plots table of L/R traces of selective neurons over three epochs and contra/ipsi population proportions
                                
        Parameters
        ----------
        save : bool, optional
            Whether to save fig to file (default False)
        """

        f, axarr = plt.subplots(4,3, sharex='col', figsize=(14, 12))
        epochs = [range(self.time_cutoff), range(self.sample, self.delay), range(self.delay, self.response), range(self.response, self.time_cutoff)]

        x = np.arange(-5.97,6,self.fs)[:self.time_cutoff]
        if 'CW03' in self.path:
            x = np.arange(-6.97,6,self.fs)[:self.time_cutoff]

        titles = ['Whole-trial', 'Sample', 'Delay', 'Response']
        
        for i in range(4):
            
            contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epochs[i])

            # Bar plot
            contraratio = len(contra_neurons)/len(self.selective_neurons) if len(self.selective_neurons) > 0 else 0
            ipsiratio = len(ipsi_neurons)/len(self.selective_neurons) if len(self.selective_neurons) > 0 else 0
            
            axarr[i, 0].bar(['Contra', 'Ipsi'], [contraratio, ipsiratio], 
                            color = ['b', 'r'])
            
            axarr[i, 0].set_ylim(0,1)
            axarr[i, 0].set_title(titles[i])
            
            if len(ipsi_neurons) != 0:
            
                overall_R, overall_L = ipsi_trace['r'], ipsi_trace['l']
                overall_R = [np.mean(overall_R[r], axis=0) for r in range(len(overall_R))]
                overall_L = [np.mean(overall_L[l], axis=0) for l in range(len(overall_L))]
                
                R_av = np.mean(overall_R, axis = 0) 
                L_av = np.mean(overall_L, axis = 0)
                
                left_err = np.std(overall_L, axis=0) / np.sqrt(len(overall_L)) 
                right_err = np.std(overall_R, axis=0) / np.sqrt(len(overall_R))
                            
                # if 'CW03' in self.path:
                #     L_av = L_av[5:]
                #     R_av = R_av[5:]
                #     left_err = left_err[5:]
                #     right_err = right_err[5:]
                    
                    
                axarr[i, 2].plot(x, L_av, 'r-')
                axarr[i, 2].plot(x, R_av, 'b-')
                        
                axarr[i, 2].fill_between(x, L_av - left_err, 
                         L_av + left_err,
                         color=['#ffaeb1'])
                axarr[i, 2].fill_between(x, R_av - right_err, 
                         R_av + right_err,
                         color=['#b4b2dc'])
                axarr[i, 2].set_title("Ipsi-preferring neurons")
            
            else:
                print('No ipsi selective neurons')
        
            if len(contra_neurons) != 0:
    
                overall_R, overall_L = contra_trace['r'], contra_trace['l']
                overall_R = [np.mean(overall_R[r], axis=0) for r in range(len(overall_R))]
                overall_L = [np.mean(overall_L[l], axis=0) for l in range(len(overall_L))]
                
                R_av = np.mean(overall_R, axis = 0) 
                L_av = np.mean(overall_L, axis = 0)
                
                left_err = np.std(overall_L, axis=0) / np.sqrt(len(overall_L)) 
                right_err = np.std(overall_R, axis=0) / np.sqrt(len(overall_R))
                            
                # if 'CW03' in self.path:
                #     L_av = L_av[5:]
                #     R_av = R_av[5:]
                #     left_err = left_err[5:]
                #     right_err = right_err[5:]
                    
                axarr[i, 1].plot(x, L_av, 'r-')
                axarr[i, 1].plot(x, R_av, 'b-')
                        
                axarr[i, 1].fill_between(x, L_av - left_err, 
                          L_av + left_err,
                          color=['#ffaeb1'])
                axarr[i, 1].fill_between(x, R_av - right_err, 
                          R_av + right_err,
                          color=['#b4b2dc'])
                axarr[i, 1].set_title("Contra-preferring neurons")

            else:
                print('No contra selective neurons')
                
        axarr[0,0].set_ylabel('Proportion of neurons')
        axarr[0,1].set_ylabel('dF/F0')
        axarr[3,1].set_xlabel('Time from Go cue (s)')
        axarr[3,2].set_xlabel('Time from Go cue (s)')
        
        if save:
            plt.savefig(self.path + r'contra_ipsi_SDR_table.png')
        
        plt.show()

    def plot_three_selectivity(self,save=False):
        """Plots selectivity traces over three epochs and number of neurons in each population
                                
        Parameters
        ----------
        save : bool, optional
            Whether to save fig to file (default False)
        """
        
        f, axarr = plt.subplots(1,5, sharex='col', figsize=(21,5))
        
        epochs = [range(self.time_cutoff), range(8,14), range(19,28), range(29,self.time_cutoff)]
        x = np.arange(-5.97,4,self.fs)[:self.time_cutoff]
        titles = ['Whole-trial', 'Sample', 'Delay', 'Response']
        
        num_epochs = []
        
        for i in range(4):
            
            contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epochs[i])
            
            if len(contra_neurons) == 0:
                
                nonpref, pref = ipsi_trace['r'], ipsi_trace['l']
                
            elif len(ipsi_neurons) == 0:
                nonpref, pref = contra_trace['l'], contra_trace['r']

            else:
                nonpref, pref = cat((ipsi_trace['r'], contra_trace['l'])), cat((ipsi_trace['l'], contra_trace['r']))
                
                
            sel = np.mean(pref, axis = 0) - np.mean(nonpref, axis = 0)
            
            err = np.std(pref, axis=0) / np.sqrt(len(pref)) 
            err += np.std(nonpref, axis=0) / np.sqrt(len(nonpref))
                        
            axarr[i + 1].plot(x, sel, 'b-')
                    
            axarr[i + 1].fill_between(x, sel - err, 
                      sel + err,
                      color=['#b4b2dc'])

            axarr[i + 1].set_title(titles[i])
            
            num_epochs += [len(contra_neurons) + len(ipsi_neurons)]

        # Bar plot
        axarr[0].bar(['S', 'D', 'R'], np.array(num_epochs[1:]) / sum(num_epochs[1:]), color = ['dimgray', 'darkgray', 'gainsboro'])
        
        axarr[0].set_ylim(0,1)
        axarr[0].set_title('Among all ALM neurons')
        
        axarr[0].set_ylabel('Proportion of neurons')
        axarr[1].set_ylabel('Selectivity')
        axarr[2].set_xlabel('Time from Go cue (s)')
        
        
        plt.show()
        
    def population_sel_timecourse(self, save=False):
        """Plots selectivity traces over three periods and number of neurons in each population
                                
        Parameters
        ----------
        save : bool, optional
            Whether to save fig to file (default False)
        """
        
        f, axarr = plt.subplots(2, 1, sharex='col', figsize=(20,15))
        epochs = [range(14,28), range(21,self.time_cutoff), range(29,self.time_cutoff)]
        x = np.arange(-5.97,4,self.fs)[:self.time_cutoff]
        titles = ['Preparatory', 'Prep + response', 'Response']
        
        sig_n = dict()
        sig_n['c'] = []
        sig_n['i'] = []
        contra_mat = np.zeros(self.time_cutoff)
        ipsi_mat = np.zeros(self.time_cutoff)
        
        for i in range(3):
            
            # contra_neurons, ipsi_neurons, contra_trace, ipsi_trace = self.contra_ipsi_pop(epochs[i])
            
            for n in self.get_epoch_selective(epochs[i]):
                                
                r, l = self.get_trace_matrix(n)
                r, l = np.array(r), np.array(l)
                side = 'c' if np.mean(r[:, epochs[i]]) > np.mean(l[:,epochs[i]]) else 'i'
                
                r, l = np.mean(r,axis=0), np.mean(l,axis=0)
                
                if side == 'c' and n not in sig_n['c']:
                    
                    sig_n['c'] += [n]
    
                    contra_mat = np.vstack((contra_mat, r - l))

                if side == 'i' and n not in sig_n['i']:
                    
                    sig_n['i'] += [n]
    
                    ipsi_mat = np.vstack((ipsi_mat, l - r))

        axarr[0].matshow((ipsi_mat[1:]), aspect="auto", cmap='jet')
        axarr[0].set_title('Ipsi-preferring neurons')
        
        axarr[1].matshow(-(contra_mat[1:]), aspect="auto", cmap='jet')
        axarr[1].set_title('Contra-preferring neurons')
        
        if save:
            plt.savefig(self.path + r'population_selectivity_overtime.jpg')
        
        plt.show()