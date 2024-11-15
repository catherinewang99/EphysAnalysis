# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:41:04 2024

@author: catherinewang
"""


import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from session import Session
import behavior
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 

path = r'H:\ephys_data\CW47\python\2024_10_17'

s1 = Session(path, passive=False)