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