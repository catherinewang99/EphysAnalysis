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
        self.delay = 0.57 + 1.3
        self.response = 0.57 + 1.3 + 3
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
        start, stop = window
        arr = arr[:,0]
        count = np.sum((arr >= start) & (arr <= stop))
        return count
    
    def get_spike_count(self, neuron, window, trials):
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
    
    
    
    def get_PSTH(self, neuron, trials, binsize=50, timestep=1, window=()):
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
        timestep = timestep / 1000
        if self.passive:
            start, stop = window if len(window) != 0 else (-0.8, 0.57+1.3+1)
        else:
            start, stop = window if len(window) != 0 else (-0.8, 0.57+1.3+3+3)

        time = np.arange(start, stop, timestep)
        
        n_rep = len(trials)
        total_counts = np.zeros_like(time)
        
        # Loop over each repetition
        for i_rep in trials:
            # Calculate histogram for each spike train
            counts, _ = np.histogram(self.spks[neuron][0, i_rep], 
                                     bins=np.arange(start, stop + timestep, timestep))
            total_counts = np.vstack((total_counts, counts / n_rep))

        stderr = np.std(total_counts[1:], axis=0) / np.sqrt(total_counts.shape[0])
        
        # Define window for convolution (smoothing)
        window = np.ones(binsize) / (binsize / 1000)
        
        # Convolve histogram with smoothing window
        total_counts = np.sum(total_counts[1:], axis=0)
        PSTH = convolve(total_counts, window, mode='same')
    
        # Adjust time and PSTH to remove edge effects from convolution
        time = time[binsize:-binsize]
        PSTH = PSTH[binsize:-binsize]
        stderr = stderr[binsize:-binsize]
        
        return PSTH, time, stderr
        
    
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
                             binsize=50, timestep=1, save=[]):
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
        stable_trials = self.stable_trials[neuron]
        L_correct_trials = np.where(self.L_correct)[0]
        L_correct_trials = [i for i in L_correct_trials if not self.early_lick[i] and i in stable_trials]
        R_correct_trials = np.where(self.R_correct)[0]
        R_correct_trials = [i for i in R_correct_trials if not self.early_lick[i] and i in stable_trials]
        
        if not opto:

            title = "Neuron {}: Raster and PSTH".format(neuron)
            
            f, axarr = plt.subplots(2, sharex=True, figsize=(10,10))
    
            #RASTER:
            for i in range(len(stable_trials)):
                axarr[0].scatter(self.spks[neuron][0, stable_trials[i]], 
                            np.ones(len(self.spks[neuron][0, stable_trials[i]])) * i, 
                            color='black', s=5)
                        
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
            
        else:
        
            title = "Neuron {}: Control".format(neuron)
            
    
            # f, axarr = plt.subplots(2,2, sharex='col', sharey = 'row')
            f, axarr = plt.subplots(2,2, sharex='col', figsize=(10,6))
            
            #RASTER:
            for i in range(len(stable_trials)):
                axarr[0, 0].scatter(self.spks[neuron][0, stable_trials[i]], 
                            np.ones(len(self.spks[neuron][0, stable_trials[i]])) * i, 
                            color='black', s=1)
                        
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
            L_opto_trials = [i for i in L_correct_trials if self.stim_ON[i] and i in stable_trials]
            R_opto_trials = self.trial_type_direction('r')
            R_opto_trials = [i for i in R_correct_trials if not self.stim_ON[i] and i in stable_trials]
            
            stable_opto_trials = [s for s in stable_trials if self.stim_ON[s]]
            
            for i in range(len(stable_opto_trials)):
                axarr[0, 1].scatter(self.spks[neuron][0, stable_opto_trials[i]], 
                            np.ones(len(self.spks[neuron][0, stable_opto_trials[i]])) * i, 
                            color='black', s=1)
                        
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
                axarr[1, 1].hlines(y=vmax, xmin=self.delay, xmax=self.delay + 1, linewidth=10, color='lightblue')
            else:
                axarr[1, 0].sharey(axarr[1, 1])  # Make Bottom-Right share y-axis with Bottom-Left
                axarr[1, 1].hlines(y=vmax_opto, xmin=self.delay, xmax=self.delay + 1, linewidth=10, color='lightblue')

            axarr[0,1].set_title(title)

            plt.tight_layout()
            
            if len(save) != 0:
                plt.savefig(save)
            plt.show()
            
        