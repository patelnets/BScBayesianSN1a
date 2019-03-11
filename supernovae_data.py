#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 10:44:22 2018

@author: vatsal
"""
import numpy as np
#import pystan
#import corner as corner



def load_supernovae_data(): 
    '''Returns full data list and another list which does not 
    have the name for each one  '''
    
    filename = "jla_lcparams.txt"
    file = open(filename, "r")
    supernovae_data_array = []
    supernovae_data_floats = []
    counter = 0 
    for line in file:
    
        line_split = line.split() #split each line into x and y values
    
        supernovae_data_array.append(line_split)
        #print(supernovae_data_array)
        if counter != 0:
            supernovae_data_floats.append(line_split[1:] )
            
        counter += 1
    
    for i in range(len(supernovae_data_floats)):
        supernovae_data_floats[i] = [ float(j) for j in supernovae_data_floats[i] ]
    
    return(supernovae_data_array, supernovae_data_floats)
    
def individual_arrays(supernovae_data_array, supernovae_data_floats): 
    """
    ,----
    | name: name of the SN
    | zcmb: CMB frame redshift (including peculiar velocity corrections for
    | nearby supernova based on the models of M.J. Hudson)
    | zhel: Heliocentric redshift (note both zcmb and zhel are needed
    |       to compute the luminosity distance)
    | dz: redshift error (no longer used by the plugin)
    | mb: B band peak magnitude
    | dmb: Error in mb (includes contributions from intrinsic dispersion, 
    |      lensing, and redshift uncertainty)
    | x1: SALT2 shape parameter
    | dx1: Error in shape parameter
    | colour: Colour parameter
    | dcolour: Error in colour
    | 3rdvar: In these files, the log_10 host stellar mass
    | d3rdvar: Error in 3rdvar
    | tmax: Date of peak brightness (mjd)
    | dtmax: Error in tmax
    | cov_m_c: The covariance between mb and colour
    | cov_s_c: The covariance between x1 and colour
    | set: A number indicating which sample this SN belongs to, with
    |    1 - SNLS, 2 - SDSS, 3 - low-z, 4 - Riess HST
    | ra: Right Ascension in degree (J2000)
    | dec: Declination in degree (J2000)
    | biascor: The correction for analysis bias applied to measured magnitudes
    | 	 (this correction is already applied to mb, original measurements
    | 	  can be obtained by subtracting this term to mb)
    
    """
    
    zcmb = []
    zhel = []
    dz = []
    mb = []
    dmb= [] 
    x1 = []
    dx1= [] 
    color = [] 
    dcolor = [] 
    three_rdvar = [] 
    d_three_rdvar = []
    tmax = []
    dtmax = []
    cov_m_s = []
    cov_m_c = []
    cov_s_c = []
    set_ = []
    ra = []
    dec = []
    biascor = []
    name = []
    for i in range(len(supernovae_data_floats)):
        name.append(supernovae_data_array[i][0])
        zcmb.append(supernovae_data_floats[i][0])
        zhel.append(supernovae_data_floats[i][1])
        dz.append(supernovae_data_floats[i][2])
        mb.append(supernovae_data_floats[i][3])
        dmb.append(supernovae_data_floats[i][4])
        x1.append(supernovae_data_floats[i][5])
        dx1.append(supernovae_data_floats[i][6])
        color.append(supernovae_data_floats[i][7])
        dcolor.append(supernovae_data_floats[i][8])
        three_rdvar.append(supernovae_data_floats[i][9]) 
        d_three_rdvar.append(supernovae_data_floats[i][10])
        tmax.append(supernovae_data_floats[i][11])
        dtmax.append(supernovae_data_floats[i][12])
        cov_m_s.append(supernovae_data_floats[i][13])
        cov_m_c.append(supernovae_data_floats[i][14])
        cov_s_c.append(supernovae_data_floats[i][15])
        set_.append(supernovae_data_floats[i][16])
        ra.append(supernovae_data_floats[i][17])
        dec.append(supernovae_data_floats[i][18])
        biascor.append(supernovae_data_floats[i][19]) 
    return(name, zcmb,zhel ,dz ,mb ,dmb ,x1 ,dx1 ,color ,dcolor ,three_rdvar ,d_three_rdvar ,tmax ,dtmax ,cov_m_s ,cov_m_c ,cov_s_c ,set_ ,ra ,dec ,biascor)
    
    
    
    
    
#supernovae_data_array, supernovae_data_floats = load_supernovae_data()
#name, zcmb,zhel ,dz ,mb ,dmb ,x1 ,dx1 ,color ,dcolor ,three_rdvar ,d_three_rdvar ,tmax ,dtmax ,cov_m_s ,cov_m_c ,cov_s_c ,set_ ,ra ,dec ,biascor = individual_arrays(supernovae_data_array, supernovae_data_floats)

##covariance matrix already saves as .npy file so only need to load 

#cov_matrix = np.load('Covariance matrix.npy')


