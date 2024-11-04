# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:27:04 2024

@author: catherinewang
"""



import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from session import Session
from matplotlib.pyplot import figure
# import decon
from scipy.stats import chisquare
import pandas as pd
plt.rcParams['pdf.fonttype'] = '42' 
import random
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

## Paths

path = r'H:\ephys_data\CW47\python\2024_10_17'

s1 = Session(path, passive=True)