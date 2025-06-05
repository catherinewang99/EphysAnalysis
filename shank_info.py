# -*- coding: utf-8 -*-
"""
Created on Wed May  7 10:33:21 2025

@author: catherinewang
"""

import probeinterface as pif
from probeinterface.plotting import plot_probe, plot_probe_group
import numpy as np

path = r'G:\data_tmp\CW59\20250222\catgt_CW59_20250222_g0\CW59_20250222_g0_imec0\CW59_20250222_g0_tcat.imec0.ap.meta'


probe = pif.read_spikeglx(path)

# pif.plotting.plot_probe(probe)

shanks = probe.shank_ids
path = r'G:\data_tmp\CW59\20250222\catgt_CW59_20250222_g0\CW59_20250222_g0_imec0\imec0_ks2_trimmed\\'


channel_map = np.load(path + r'channel_map.npy')
channel_positions = np.load(path + r'channel_positions.npy')

#%% Analysis across probes



import sys
sys.path.append("C:\scripts\Ephys analysis\ephys_pipeline")
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from ephysSession import Session
import behavior
cat = np.concatenate
plt.rcParams['pdf.fonttype'] = '42' 

# paths:
    
    
all_expert_paths = [[
                        # r'J:\ephys_data\CW49\python\2024_12_11',
                        # r'J:\ephys_data\CW49\python\2024_12_12',
                        r'J:\ephys_data\CW49\python\2024_12_13',
                        r'J:\ephys_data\CW49\python\2024_12_14',
                        r'J:\ephys_data\CW49\python\2024_12_15',
                        r'J:\ephys_data\CW49\python\2024_12_16',
                
                          ],
                    [
                        # r'J:\ephys_data\CW53\python\2025_01_27',
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
all_learning_paths_stimcorrected = [[r'G:\ephys_data\CW63\python\2025_03_19',
                          r'G:\ephys_data\CW63\python\2025_03_20',         
                          r'G:\ephys_data\CW63\python\2025_03_22',         
                           # r'G:\ephys_data\CW63\python\2025_03_23',         
                          r'G:\ephys_data\CW63\python\2025_03_25',],
                          
                          [r'G:\ephys_data\CW61\python\2025_03_08',
                           # r'G:\ephys_data\CW61\python\2025_03_09', 
                           r'G:\ephys_data\CW61\python\2025_03_10', 
                           r'G:\ephys_data\CW61\python\2025_03_11', 
                           # r'G:\ephys_data\CW61\python\2025_03_12', 
                           # r'G:\ephys_data\CW61\python\2025_03_14', 
                           r'G:\ephys_data\CW61\python\2025_03_17', 
                           # r'G:\ephys_data\CW61\python\2025_03_18', 
                           ],
                          [r'J:\ephys_data\CW54\python\2025_02_01',
                           r'J:\ephys_data\CW54\python\2025_02_03']
                          ]


#%% look at probe info, figure out which is anterior vs posterior


path = r'G:\ephys_data\CW59\python\2025_02_22'
s1 = Session(path, passive=False, filter_low_perf=True, filter_by_stim=True) # blue laser session


L_side_shank = s1.shank[s1.L_alm_idx]
R_side_shank = s1.shank[s1.R_alm_idx]

# Number of units per shank

for i in range(1,5):
    plt.bar([i], [sum(R_side_shank == i)])

