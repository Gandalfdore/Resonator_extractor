# -*- coding: utf-8 -*-
"""

@author: Boris Nedyalkov
"""
import numpy as np
import matplotlib.pyplot as plt
import Qf_extractor_functions as qf

data_set = np.loadtxt("localsweep_20221201_134354.sweep",dtype=None)

Master_array = qf.data_massager(data_set)  # this function takes the raw I and Q data and gives amliptude and phase

m = 19   # the name (number) of the resonator you want ##in this sample there are only 60 resonators, the rest is blank
freq =  Master_array [m,0,:]
amp =  Master_array [m,1,:] 
 
Q_i, Qe, Q_tot, f_0, df, phi, e = qf.analysis_function (freq, amp) ## this function gives you the parameters of the resonator chosen

###########################################################

Re, Im = qf.Re_and_Im(freq, f_0, Q_tot, Qe, phi, e)  ## get the Real and Imaginary parts of the S21 data

############################################################

fig1, (ax1) = plt.subplots(1, 1, figsize = (5, 5), dpi = 100)
ax1.plot(Re, Im, '-', color='#d63c49',linewidth=1.5)

plt.show()

###########################################################
