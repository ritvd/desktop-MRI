import numpy as np
import os

os.chdir('/home/rithika/Desktop/desktop MRI/dMRI/data_5') 

brain_v1 = np.load('brain_v1.npy')
brain_v2 = np.load('brain_v2.npy')
brain_v3 = np.load('brain_v3.npy')
brain_v4 = np.load('brain_v4.npy')
brain_v5 = np.load('brain_v5.npy')
brain_v6 = np.load('brain_v6.npy')
brain_v7 = np.load('brain_v7.npy')
brain_v8 = np.load('brain_v8.npy')

mod_brain_vol = np.concatenate((brain_v1, brain_v2, brain_v3, brain_v4, brain_v5,brain_v6,brain_v7,brain_v8), axis=0)
np.save('mod_brain_vol', mod_brain_vol)