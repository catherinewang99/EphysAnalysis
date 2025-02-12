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

#%% Paths

path = r'J:\ephys_data\CW53\python\2025_02_02'
l1 = Mode(path, passive=False, side='R', stimside='L')

input_vec = l1.input_vector(by_trialtype=True, plot=True)
cd_choice, _ = l1.plot_CD(mode_input='choice', plot=False)