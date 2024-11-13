# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:20:22 2024

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
from session import Session


class multiSession(Session):
    
    """
    A class used to store and process multiple days of 
    electrophysiology data alongside behavior data, 
    mostly used for passive sessions
 
    ...
 
    Attributes
    ----------

 
    Methods
    -------

    """
    def __init__(self, paths, side = 'all', passive=False, laser='blue'):
        

        """
        Parameters
        ----------
        paths : str
            all paths to data
        side : str
            Default 'all', can optionally grab only 'L' or 'R' alm hemi
        passive : bool
            If true, then we are looking at a passive session. Otherwise, beh
        laser : str
            Default blue laser, only matters for passive recordings

        """       
        
        self.num_sessions = len(paths)
        self.laser = laser
        self.side = side
        self.passive = passive
                    
        self.data_type = 'ephys'
        self.sample = .570 # 570 ms offset for sample start
        self.sampling_freq = 30000 # 30k for npxl 2.0
        
        # Aggregate data:
            
        self.unit_side = {}
        self.L_alm_idx = dict()
        self.R_alm_idx = dict()
        
        self.celltype = dict() # Shape: (units,)
        
        self.num_neurons = dict()
        self.num_trials = dict()
        
        self.spks = dict() # Shape: (units,)(1, trials)
        self.stable_trials = dict() # Shape: (units,)(stable_trials,)
        
        self.waveform = dict()
        
        self.stim_ON = dict()
        

        self.stim_level = dict()
        self.all_stim_levels = dict()
        
        self.stim_trials = dict()
        
        
        num_session = 0
        for path in paths:
            super().__init__(path, side = side, passive=passive, laser=laser)
            
            
            name = laser + 'passive_units.mat' if passive else 'units.mat'
            units_tmp = scio.loadmat(os.path.join(path, name))
            units = copy.deepcopy(units_tmp)
                
            imec = units['imec']
            self.unit_side[num_session] = np.where(imec == 0, 'L', 'R')[0] # Shape: (units)
            self.L_alm_idx[num_session] = np.where(imec == 0)[1]
            self.R_alm_idx[num_session] = np.where(imec == 1)[1]
            
            self.celltype[num_session] = units['celltype'][0] # Shape: (units,)
            
            self.num_neurons[num_session] = units['units'].shape[1]
            self.num_trials[num_session] = units['units'][0,0].shape[1]
            
            self.spks[num_session] = units['units'][0] # Shape: (units,)(1, trials)
            
            for trial in range(units['stabletrials'][0].shape[0]):
                units['stabletrials'][0,trial] = units['stabletrials'][0,trial][0] - 1 # Convert to 0 index
            self.stable_trials[num_session] = units['stabletrials'][0] # Shape: (units,)(stable_trials,)
            
            self.waveform[num_session] = units['mean_waveform'][0]

            
            # Behavior
            beh_name = laser + 'passive_behavior.mat' if passive else 'behavior.mat'
            behavior = scio.loadmat(os.path.join(path, beh_name))
            # self.L_correct = cat(behavior['L_hit_tmp'])
            # self.R_correct = cat(behavior['R_hit_tmp'])
            
            # self.early_lick = cat(behavior['LickEarly_tmp'])
            
            # self.L_wrong = cat(behavior['L_miss_tmp'])
            # self.R_wrong = cat(behavior['R_miss_tmp'])
            
            # self.L_ignore = cat(behavior['L_ignore_tmp'])
            # self.R_ignore = cat(behavior['R_ignore_tmp'])
            
                            
            # self.lick_L_trials = np.sort(cat((np.where(self.L_correct)[0], 
            #                              np.where(self.R_wrong)[0])))
            
            # self.lick_R_trials = np.sort(cat((np.where(self.R_correct)[0], 
            #                              np.where(self.L_wrong)[0]))) 
            
            # self.L_trials = np.sort(cat((np.where(self.L_correct)[0], 
            #                              np.where(self.L_wrong)[0],
            #                              np.where(self.L_ignore)[0])))
            
            # self.R_trials = np.sort(cat((np.where(self.R_correct)[0], 
            #                              np.where(self.R_wrong)[0],
            #                              np.where(self.R_ignore)[0]))) 
            
            self.stim_ON[num_session] = cat(behavior['StimDur_tmp']) > 0
            
            # self.i_good_trials = cat(behavior['i_good_trials']) - 1 # zero indexing in python
    
            if 'StimLevel' in behavior.keys():
                self.stim_level[num_session] = cat(behavior['StimLevel'])
                self.all_stim_levels[num_session] = sorted(list(set(self.stim_level)))
            
            # Re-adjust with i good trials
            self.stim_trials[num_session] = np.where(self.stim_ON)[0]
            
            num_session += 1
            
            
            
            
            
            
            