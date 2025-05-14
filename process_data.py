# -*- coding: utf-8 -*-
"""
Created on Wed May  7 10:27:13 2025

Written to process data to correctly relabel SST neurons

@author: catherinewang
"""

import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from ephysSession import Session
from matplotlib.pyplot import figure
# import decon
from scipy.stats import chisquare
import pandas as pd
plt.rcParams['pdf.fonttype'] = '42' 
import random
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
cat = np.concatenate
# from multisession import multiSession
import behavior


#%% Paths
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
#%% Relabel SST neurons and save file

np.load()
