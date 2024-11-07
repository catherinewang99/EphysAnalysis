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
from scipy.stats import mannwhitneyu
from scipy.stats import mstats
from scipy.ndimage import median_filter
# from .LinRegpval import LinearRegression
plt.rcParams['pdf.fonttype'] = 42 
import time 
import random
from itertools import groupby
from operator import itemgetter

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
    def __init__(self, path, side = 'all', passive=False, laser='blue'):
        

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
        
        self.num_neurons = units['units'].shape[1]
        self.num_trials = units['units'][0,0].shape[1]
        
        self.spks = units['units'][0] # Shape: (units,)(1, trials)
        
        for trial in range(units['stabletrials'][0].shape[0]):
            units['stabletrials'][0,trial] = units['stabletrials'][0,trial][0] - 1 # Convert to 0 index
        self.stable_trials = units['stabletrials'][0] # Shape: (units,)(stable_trials,)
        
        self.waveform = units['mean_waveform'][0]
        
        self.data_type = 'ephys'
        self.sample = .570 # 570 ms offset for sample start
        self.sampling_freq = 30000 # 30k for npxl 2.0
        
        # Behavior
        beh_name = laser + 'passive_behavior.mat' if passive else 'behavior.mat'
        behavior = scio.loadmat(os.path.join(path, beh_name))
        self.L_correct = cat(behavior['L_hit_tmp'])
        self.R_correct = cat(behavior['R_hit_tmp'])
        
        self.early_lick = cat(behavior['LickEarly_tmp'])
        
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

        if 'StimLevel' in behavior.keys():
            self.stim_level = cat(behavior['StimLevel'])
            self.all_stim_levels = sorted(list(set(self.stim_level)))
        
        # Re-adjust with i good trials
        self.stim_trials = np.where(self.stim_ON)[0]
        
    ### BEHAVIOR ONLY FUNCTIONS ###
    
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
        start, stop = window
        arr = arr[:,0]
        count = np.sum((arr >= start) & (arr <= stop))
        return count
    
    
    def get_spike_rate(self, neuron, window, trials):
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
            
        return np.mean(all_spk_rate)
    
    def get_PSTH(self ):
        return