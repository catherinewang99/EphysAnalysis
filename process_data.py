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

#%% Relabel SST neurons


