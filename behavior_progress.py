# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:15:56 2024

Analyze ephys behavior


@author: catherinewang
"""

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import session
import behavior
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 

#%% Plot learning progression
# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW48\python_behavior', behavior_only=True)
# b.learning_progression(window = 50)

b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW49\python_behavior', behavior_only=True)
b.learning_progression(window = 50)

# b = behavior.Behavior(r'J:\ephys_data\Behavior data\CW52\python_behavior', behavior_only=True)
# b.learning_progression(window = 50)