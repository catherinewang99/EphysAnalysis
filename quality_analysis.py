# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:48:12 2024

Analyze the quality of collected data

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

path = r'H:\ephys_data\CW47\python\2024_10_17'

paths = [r'H:\ephys_data\CW47\python\2024_10_17',
         r'H:\ephys_data\CW47\python\2024_10_17',
         r'H:\ephys_data\CW47\python\2024_10_17',
         r'H:\ephys_data\CW47\python\2024_10_17']

s1 = Session(path, passive=False)#, side='R')


