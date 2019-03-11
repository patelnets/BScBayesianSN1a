"""
Created on Fri Jan 26 12:34:58 2018
@author: vatsal
"""
import numpy as np
import pystan
import corner as corner
import glob
from astropy.io import fits
import supernovae_data as sup_data
import pickle
import time as time 
import matplotlib as mpl
import matplotlib.pyplot as plt
import math as math
import seaborn as sns
import pandas as pd  
#
#"""FUNCTIONS USED"""
#N = 740
#chains = 4
#iterations = 2000
#
#def order_for_pick(zhel_list, step):
#    #gives list which Maps sorted to original order 
#    # ie If i knew A from zhel_sorted and wanted to find it in zhel then i do zhel[original_order[index(A)]]
#    if step != 1:
#        sorted_list = set(np.sort(zhel_list))
#        sorted_list = np.sort(np.array(list(sorted_list)))
#        original_order = []
#        for j in range(len(sorted_list)):
#            for i in range(len(zhel_list)):
#                if zhel[:N][i] == sorted_list[j]:
#                    original_order.append(i)
#                    break
#        original_order = np.array(original_order)
#        
#        original_order_pick = []
#        zhel_pick = []
#        for i in range(0, len(sorted_list), step):
#            original_order_pick.append(original_order[i])
#            zhel_pick.append(zhel_list[original_order[i]])
#            
#        if original_order_pick[-1] != 239:
#            original_order_pick.append(239)
#            zhel_pick.append(1.3)
#            
#        return(original_order_pick, zhel_pick)
#    else :
#        full_original_order = np.arange(0,len(zhel_list), 1)
#        return(full_original_order, zhel_list)        
#
#def covariance_matrix_pick(full_cov, index_list):
#    
#    index_3 = []
#    for i in index_list:
#        index_3.append(i*3)
#        index_3.append(i*3 +1)
#        index_3.append(i*3 +2)
#        
#    b_list = []
#    for i in index_3: 
#        temp_list = []
#        for j in index_3:
#            temp_list.append(full_cov[i,j])
#        b_list.append(temp_list)
#        
#    b = np.array(b_list)
#    return(b)
#
#def sort_ODE (z_list):
#    #sorting the data for ODE
#    sorted_list = set(np.sort(z_list))
#    sorted_list = np.sort(np.array(list(sorted_list)))
#    original_order = []
#    for i in range(len(z_list)):
#        for j in range(len(sorted_list)):
#            if zhel_pick[i] == sorted_list[j]:
#                original_order.append(j)
#                break
#    original_order = np.array(original_order) + 1
#    
#    return(original_order, sorted_list)
#
#
#""" Gather the data of 740 supernovae into appropriate named arrays.
#    KEY
#
#    ,----
#    | name: name of the SN
#    | zcmb: CMB frame redshift (including peculiar velocity corrections for
#    | nearby supernova based on the models of M.J. Hudson)
#    | zhel: Heliocentric redshift (note both zcmb and zhel are needed
#    |       to compute the luminosity distance)
#    | dz: redshift error (no longer used by the plugin)
#    | mb: B band peak magnitude
#    | dmb: Error in mb (includes contributions from intrinsic dispersion,
#    |      lensing, and redshift uncertainty)
#    | x1: SALT2 shape parameter
#    | dx1: Error in shape parameter
#    | colour: Colour parameter
#    | dcolour: Error in colour
#    | 3rdvar: In these files, the log_10 host stellar mass
#    | d3rdvar: Error in 3rdvar
#    | tmax: Date of peak brightness (mjd)
#    | dtmax: Error in tmax
#    | cov_m_c: The covariance between mb and colour
#    | cov_s_c: The covariance between x1 and colour
#    | set: A number indicating which sample this SN belongs to, with
#    |    1 - SNLS, 2 - SDSS, 3 - low-z, 4 - Riess HST
#    | ra: Right Ascension in degree (J2000)
#    | dec: Declination in degree (J2000)
#    | biascor: The correction for analysis bias applied to measured magnitudes
#    | 	 (this correction is already applied to mb, original measurements
#    | 	  can be obtained by subtracting this term to mb)
#
#    cov_matrix:  covariance variance matrix.
#
#"""
##LOAD THE OBSERVED DATA
#supernovae_data_array, supernovae_data_floats = sup_data.load_supernovae_data()
#name, zcmb,zhel ,dz ,mb ,dmb ,x1 ,dx1 ,color ,dcolor ,three_rdvar ,d_three_rdvar ,tmax ,dtmax ,cov_m_s ,cov_m_c ,cov_s_c ,set_ ,ra ,dec ,biascor = sup_data.individual_arrays(supernovae_data_array, supernovae_data_floats)
#
#cov_matrix = sum([fits.getdata(mat) for mat in glob.glob('C*.fits')])
#cov_matrix_stat = fits.getdata('C_stat.fits')
#
#data_meas = np.zeros([3*N])
#for i in range(N):
#    data_meas[3*i] = mb[i]
#    data_meas[3*i+1] = x1[i]
#    data_meas[3*i+2] = color[i]
#
#
##PICKING THE DATA 
#N_pick = 740
#pick_step = math.ceil(740/N_pick)
#original_order_pick, zhel_pick = order_for_pick(zhel[:740], pick_step)
#cov_pick = covariance_matrix_pick(cov_matrix, original_order_pick)
#N_pick = len(zhel_pick)
#
#data_meas_pick = []
#for i in original_order_pick:
#    data_meas_pick.append(data_meas[3*i])
#    data_meas_pick.append(data_meas[3*i+1])
#    data_meas_pick.append(data_meas[3*i+2])
#data_meas_pick = np.array(data_meas_pick)
#
#
#original_order_ODE, sorted_list_ODE = sort_ODE (zhel_pick)
#
##initial_par = {"Omega_m": 0.31, "Omega_kappa": -0.019, "alpha": 0.14, "beta": 3.1, "M_0_e": -19.1, "sigma_res": 0.1, } #, init=[initial_par]*chains
#
#data_stan = {'N': len(zhel_pick),
#             'N_real': len(sorted_list_ODE),
#              'z': zhel_pick,
#              'z_real': sorted_list_ODE, 
#              'z_order' : original_order_ODE,
#              'data_meas': data_meas_pick,
#              'cov': cov_pick,
#              'y0': [0],
#              'z0': 0.}
#
##RUN CODE 
#parameters = ['Omega_m','Omega_kappa', 'Omega_lambda' ,'alpha','beta','M_0_e','sigma_res', 'x_star_1' ,'c_star', 'R_x_1', 'R_c']
#time1 = time.time()
#sm = pystan.StanModel(file='supernovae_curved_w-1.stan', verbose=False)
#
##control = dict(stepsize=0.01, adapt_delta=0.99),
#fit = sm.sampling( warmup= 500,  data=data_stan, iter=iterations, chains=chains, verbose=True, n_jobs = -1, refresh = 10, pars = parameters)
#time2 = time.time()
#print('time python = ', time2- time1)
#
#
##Save RESULT 
#stan_result = {'model': sm, 'fit': fit}
#f = open('N_'+str(N_pick)+'Chains_'+str(chains)+'iter_'+str(iterations)+'_fitdata.pkl', 'wb')
#pickle.dump(stan_result, f)
#f.close()


load_result = True

if load_result:
    f = open('N_311Chains_4iter_2000_fitdata.pkl', 'rb')
    result = pickle.load(f)
    f.close()

    chains = result['fit'].extract()
    #,levels=(1-np.exp(-0.5),)
    data =[]
    ones = [1]*6000
    Omega_lambda = ones - chains['Omega_m'] - chains['Omega_kappa']    
    for i in range(len(chains['Omega_m'])):
        data.append([chains['Omega_m'][i], Omega_lambda[i], chains['alpha'][i], chains['beta'][i]])

    figure = corner.corner(data,labels=[r"$\Omega_m $", r"$\Omega_{\lambda}$", r"$\alpha$", r"$\beta$"],show_titles=True, title_kwargs={"fontsize": 12}, levels=[0.86, 0.39])

    dataframe=pd.DataFrame({'Omega_m': chains['Omega_m'],'Omega_kappa': chains['Omega_kappa'], 'alpha': chains['alpha'], 'beta':chains['beta']})
    sns.set(style="white")
    
    g = sns.PairGrid(dataframe, diag_sharey=False)
    g.map_lower(sns.kdeplot, cmap="Blues_d", n_levels=2)
    g.map_diag(sns.kdeplot, lw=3)
    
#    names = ['alpha', 'beta', 'Omega_m','Omega_kappa']
#    latex = [r'$\alpha$',r'$\beta$',r'$\Omega_m$',r'$\Omega_{\kappa}$']
#    colours = ['blue','orange','green','red']
#    for j in range(len(names)):
#        plt.figure(names[j])
#        for i in range(4):
#            plt.plot(np.arange(1,1501,1), chains[names[j]][(i*1500):((i+1)*1500)], color = colours[i], linewidth = 0.4, label = 'Chain ' + str(i+1))
#        plt.xlabel('Iteration', size=13)
#        plt.ylabel(latex[j], size=13)
#        plt.legend(loc = 'best', frameon = True) #, prop = { 'size': 11}
#    plt.show()
#        
        