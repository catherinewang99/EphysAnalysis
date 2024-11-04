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
        
            
        name = laser + 'passive_units.mat' if passive else 'units.mat'
        units_tmp = scio.loadmat(os.path.join(path, name))
        units = copy.deepcopy(units_tmp)
            
        imec = units['imec']
        self.unit_side = np.where(imec == 0, 'L', 'R')
        
        self.num_neurons = units['units'].shape[1]
        self.num_trials = units['units'][0,0].shape[1]
        
        
        # Behavior
        behavior = scio.loadmat(os.path.join(path,"behavior.mat"))
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
        
        
        if 'StimLevel' in behavior.keys():
            self.stim_level = cat(behavior['StimLevel'])
            
        if self.i_good_trials[-1] > self.num_trials:
            
            print('More Bpod trials than 2 photon trials')
            self.i_good_trials = [i for i in self.i_good_trials if i < self.num_trials]
            self.stim_ON = self.stim_ON[:self.num_trials]
        
        # Re-adjust with i good trials
        self.stim_trials = np.where(self.stim_ON)[0]

        