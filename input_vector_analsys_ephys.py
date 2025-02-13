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

#%% Run analysis

path = r'J:\ephys_data\CW53\python\2025_01_29'
s1 = Mode(path, side='R', stimside='L')

input_vec = s1.input_vector(by_trialtype=True, plot=True)

input_vec = s1.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=True)
input_vec = s1.input_vector(by_trialtype=False, plot=True, plot_ctl_opto=False)


cd_choice, _ = s1.plot_CD(mode_input='choice', plot=False)