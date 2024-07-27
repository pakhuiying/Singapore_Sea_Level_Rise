"""
test run of bathtub.py with test np arrays
"""

from bathtub import *
import numpy as np
import sys
import time
# from scipy import signal
import scipy.signal as signal
import matplotlib.pyplot as plt

#Set global attenuation factor (unitless)
att_val = 0.0002
dem_res = 30

att_bool = True
connectivity_bool = True

def get_water_array(n = 10,slr = 1.0):
    """
    set as 1 meter SLR
    """
    return np.zeros((n,n),dtype=float) + slr

def get_dem_array(n = 10, std = 1, max_elev = 5):
    '''
    Generates a n x n matrix with a centered gaussian 
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    
    # gaussian1D = signal.gaussian(n, std)
    # gaussian2D = np.outer(gaussian1D, gaussian1D)
    ax = np.linspace(-(n - 1) / 2., (n - 1) / 2., n)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(std))
    kernel = np.outer(gauss, gauss)
    kernel = kernel / np.sum(kernel)
    return kernel/np.max(kernel)*max_elev

def get_protection_array(n = 10):
    return np.zeros((n,n),dtype = float)

if __name__ == "__main__":

    land_thresh = 0.2
    domain_size = 50
    std = 7
    slr = 0.4

    dem_arr = get_dem_array(n = domain_size, std=std)
    land_arr = dem_arr >= land_thresh
    water_arr = get_water_array(dem_arr.shape[0], slr = slr)
    protection_arr = get_protection_array(dem_arr.shape[0])

    arr_list = [dem_arr,land_arr,water_arr,protection_arr]
    title_list = ['DEM (float)','Land (bool)',f'SLR = {slr:.2f} (float)', 'Protection (float)']

    #Exploit minwater to get a starting point for flooding efficiently.
    start_set_min = getStartSet(water_arr, dem_arr, protection_arr, land_arr) # The start_set indicates the pixel coordinates (int tuples) to continue progressive flooding.
    print('start_set_min:',start_set_min)
    bolean_min = np.zeros(dem_arr.shape, dtype = bool)
    
    fig, axes = plt.subplots(1,len(arr_list), figsize = (12,5))
    
    for i,ax in enumerate(axes.flatten()):
        im = ax.imshow(arr_list[i])
        ax.set_title(title_list[i])
        plt.colorbar(im, ax=ax)
    
    plt.show()

    fig, axes = plt.subplots(1,2, figsize=(10,5))
    

    