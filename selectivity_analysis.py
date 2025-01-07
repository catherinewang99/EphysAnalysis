# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 09:38:49 2025

Recreate elements from Chen et al 2021 Fig 1

@author: catherinewang

"""


import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from session import Session
import behavior
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 

#%% Aggregate over all FOVs for this analysis

paths = [
            r'J:\ephys_data\CW49\python\2024_12_11',
            r'J:\ephys_data\CW49\python\2024_12_12',
            r'J:\ephys_data\CW49\python\2024_12_13',
            r'J:\ephys_data\CW49\python\2024_12_14',
            r'J:\ephys_data\CW49\python\2024_12_15',
            r'J:\ephys_data\CW49\python\2024_12_16',
        
        ]


# Get selectivity (spikes/s) Chen et al 2021 Fig 1C



# Proportion selective: 200 ms time bins, p < 0.05