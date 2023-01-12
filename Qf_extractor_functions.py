# -*- coding: utf-8 -*-
"""
Code for fitting Loerentzian function, implemented according to doi:10.1063/1.3692073  

author: Boris Nedyalkov
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


####################################################################################

def I_Q_to_Ampl_and_Phase (I_array,Q_array):
    
    """ AUXILLARY FUNCTION (it is used not as a final function, but insted as a part of on of the functions below)
    
    This function takes I and Q data arrays and otputs arrays of the amplitude and the phase"""
    
    Ampl = np.sqrt(I_array**2 + Q_array**2)
    Phase =np.arctan(I_array/Q_array)
    
    return Ampl, Phase



def data_massager (data_set):
    
    """MAIN FUNCTION
    This functon takes the raw I and Q data and transforms it into Amplitude and Phase in a neat 3D NumPy array.
    
    It accepts the standard sweep arrays of the Delft THz group's resonators and outputs the 3D array.
    
    The output 3D array looks like this:
    
    OUTPUT_ARRAY [# OF THE RESONATOR, X, DATA_for_X], WHERE:
                                                   X==0 is the frequency
                                                   X==1 is the Amplitude
                                                   X==2 is thee Phase
                                                   
    ,for example OUTPUT_ARRAY [24,0,:] are the frequency values that resonator #24 was sweeped for 
    and OUTPUT_ARRAY [24,1,:] would be the corresponding amplitude values for this frequency sweep for resonator #24
    and OUTPUT_ARRAY [24,2,:] would be the corresponding phase values for this frequency sweep for resonator #24"""
    
    
    num_of_res = int((data_set.shape[1]-1)/3)
    num_of_datapoints = data_set.shape[0]
    
    print("---Converting I and Q data to Amplitude and Phase--- \n",
          "Number of resonators:", num_of_res)
    
    Master_array = np.empty ((num_of_res,3,len(data_set)))
    
    for i in range(num_of_res):
        # print (i)
        sweep_freq = data_set[:,0] # x10 to make the values in MHz
        fixed_freq = data_set[:,1 + 3*i]/10 # values in MHz
        I = data_set[:,2 + 3*i]
        Q = data_set[:,3 + 3*i]
        
        Amplitude, Phase = I_Q_to_Ampl_and_Phase (I, Q)
        
        Master_array [i,0,:] = sweep_freq+fixed_freq
        Master_array [i,1,:] = Amplitude
        Master_array [i,2,:] = Phase
    
    
    return Master_array


#################################################################

def func_for_fitting_modified_lorentzian (f, f0, Q, Qe, phi, vertical_shift, e):
    
    """AUXILLARY FUNCTION
    This function produces a assymetric lorentzian curve
    f - the sweep parameter
    f0 - the median
    Q - the Q factor
    Qe - dissipation Q factor
    phi - the phase shift
    vertical_shift - just a shift along the vertical axis
    """
    
    eps_f = (f - f0)/f
    
    part0 = np.cos(phi) + 2*Q*eps_f*np.sin(phi)
    part1 = 1 + 4*(Q**2)*(eps_f**2)
    part2 = np.sin(phi) - 2*Q*eps_f*np.cos(phi)
    
    real_part = 1 - (Q/Qe)*(part0/part1)
    
    imaginary_part = (Q/Qe)*(part2/part1)
    
    formula = vertical_shift + np.sqrt((1+e)*(real_part**2 + imaginary_part**2))
    
    return formula

#################################################################

def normalize(array):
    """AUXILLARY FUNCTION
    This function normalizes a given NumPy array. The normalization range is between 0 and 1"""
    max_value = np.amax(array)
    min_value = np.amin(array)
    diff_array = max_value-min_value
    
    norm_arr = (array - min_value)/diff_array
    
    return norm_arr

#################################################################

def analysis_function (freq_array, amp_array):
    """MAIN FUNCTION
    This function takes the amplitude of the signal, analyses it and outputs useful parameters
    INPUT PARAMETERS: freq_array - the frequncy sweep datapoints
                      amp_array - the amplitude of the signal's datapoints
    OUTPUT PARAMETERS: Q_i - internal Q factor's magnitude
                        Qe - enternal Q factor's magnitude
                        Q_tot - total Q factors's magnitude (loaded)
                        f_0 - resonator's resonance frequency
                        df - the offshift of the frequency due to phase mismatch
                        phi - the phase mismatch across the resonance curve
                        e - amplitude shift due to loss
    It also plots the fit
    """
    
    min_val = np.amin(amp_array)
    index_of_the_min_number= np.where(amp_array == min_val)
    f0_suggestion = freq_array[index_of_the_min_number [0][0]]

    norm_amp_array = normalize(amp_array)
    
    # plt.plot(freq_array,norm_amp_array)
    
    f0 = f0_suggestion # initial guess values for the curve_fit function, if the fits don't work, try to adjust the parameters below
    Q = 25000
    Qe = 25000
    phi = 0
    vertical_shift = 0
    e = 0
    
    
    """Here is the part which fits with a modified lorentzian curve"""
    
    popt_lor, pcov_lor = curve_fit(func_for_fitting_modified_lorentzian, freq_array, norm_amp_array, p0 = [f0, Q, Qe, phi, vertical_shift, e])

    # plt.plot(freq_array,func_for_fitting_modified_lorentzian(freq_array, *popt_lor), '--')
    
    fig1, (ax1) = plt.subplots(1, 1, figsize = (7, 5), dpi = 100)
    ax1.plot(freq_array,norm_amp_array, '-', color='blue',linewidth=1.5, label = "data")
    ax1.plot(freq_array,func_for_fitting_modified_lorentzian(freq_array, *popt_lor), '--', color='yellow',linewidth=1.5, label = "fit")
    ax1.set_xlabel('$Frequency$ (MHz)', fontsize=12)
    ax1.set_ylabel('Normalized power', fontsize=12)
    plt.legend()
    plt.show()

    standard_devaition_errors = np.sqrt(np.diag(pcov_lor))

    print("Variances of the fit:",standard_devaition_errors)
    print("==================================")
    print("Parameters of the fit:",popt_lor)
    print("==================================")

    Q_tot = abs(popt_lor[1])
    Qe = abs(popt_lor[2])
    phi = abs(popt_lor[3]) 
    e = abs(popt_lor[5])

    Q_i = 1/((1/Q) - (np.cos(phi)/(Qe))) 
    
    f_0 = popt_lor[0]
    df = abs(f_0-f0_suggestion)
    
    print("Q internal is:", abs(Q_i))
    print("Q external is:", abs(Qe))
    print("Q loaded is:", abs(Q_tot))
    print("Resonance frequency (in MHz) is:", f_0)
    print("frequency off-shift (in kHz) is:", df*1000)
    print("Phase shift is:", phi)
    print("Loss amplitude:", e)
    
    return Q_i,Qe,Q_tot,f_0,df,phi,e


def Re_and_Im (f_array, f0, Q, Qe, phi, e):
    
    """  This function takes in the parameters from the S21 and outputs the Re{S21} and Im{S21} """
    
    Re = np.zeros (len(f_array))
    Im = np.zeros (len(f_array))
    
    for i in range(len(f_array)):
        
        f=f_array[i]
        eps_f = (f - f0)/f
        # print (i,"===",f)
        
        part0 = np.cos(phi) + 2*Q*eps_f*np.sin(phi)
        part1 = 1 + 4*(Q**2)*(eps_f**2)
        part2 = np.sin(phi) - 2*Q*eps_f*np.cos(phi)
        
        Re[i] = 1 - (Q/Qe)*(part0/part1)
        Im[i] = (Q/Qe)*(part2/part1)
               
    return Re, Im

def I_Q_for_Circle (Real_array, Im_array):
    
    """ Implemented with Eq. 3.35 and 3.36 from the thesis of Pieter de Visser
    This function takes the real and imaginary compenents of the S21 data and produces 
    the Amplitude and Phase of the resonant circle, centered on the x_c point """
    
    amp = np.sqrt(Im_array**2 + Real_array**2)
    Smin = np.amin(amp)
    x_c = (1 + Smin)/2
    print(x_c)
    
    Ampl_modified = (np.sqrt((Real_array - x_c)**2 + Im_array**2))/(1-x_c)
    tan_theta = Im_array/(x_c - Real_array)
    Phase_angle_modified = np.arctan(tan_theta)
    
    return Ampl_modified, Phase_angle_modified