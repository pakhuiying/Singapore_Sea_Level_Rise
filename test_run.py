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
from matplotlib import cm
from matplotlib.patches import Rectangle
import argparse

# parse arguments
parser = argparse.ArgumentParser(description='Generate SLR based on bathtub algorithm')
parser.add_argument('--slr',type=float, default=1.0)
parser.add_argument('--att_val',type=float, default=0.0002)
parser.add_argument('--slope', type=float, default=0.5)
args = parser.parse_args()

def get_water_array(n = 10,slr = 1.0):
    """
    set as 1 meter SLR
    """
    return np.zeros((n,n),dtype=float) + slr

def get_dem_array(n = 10, slope = 1, max_elev = 5, min_elev=-1):
    '''
    Generates a n x n matrix with a centered gaussian 
    of standard deviation std centered on it. If normalised,
    its volume equals 1.
    @returns X, Y mesh grid and Z (DEM elev)
    '''
    
    # ax = np.linspace(-(n - 1) / 2., (n - 1) / 2., n)
    # gauss = np.exp(-0.5 * np.square(ax) / np.square(std))
    # kernel = np.outer(gauss, gauss)
    # kernel = kernel / np.sum(kernel)
    # norm_kernel = kernel/np.max(kernel)
    # return norm_kernel*max_elev + min_elev
    
    # defining surface and axes
    def f(x, y):
        return np.sin(np.sqrt(slope*x ** 2 + slope*y ** 2))
    
    # x and y axis
    x = np.linspace(-1, 5, n)
    y = np.linspace(-1, 5, n)
    
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    return X, Y, Z/np.max(Z)*max_elev + max_elev + min_elev

def get_protection_array(n = 10):
    return np.zeros((n,n),dtype = float)

def get_seawall_idx():
    """ 
    returns a diagonal seawall on the lower right corner of a square array
    returns a list of tuple of (i,j) array index
    """
    idx1 = [(i,int(9/5*domain_size)-i-1) for i in range(int(4/5*domain_size),domain_size)]
    idx2 = [(i,j-1) for i,j in idx1]
    return idx1 + idx2

def build_seawall(protection_arr, seawall_idx, elev):
    """
    @param seawall_idx (int): array index
    @param elev (int): elevation of the seawall
    """
    for i,j in seawall_idx:
        protection_arr[i,j] = elev
    return protection_arr

def plot_seawall(ax,n):
    ax.add_patch(Rectangle((0, n-10), n-5, 5, fill=True, hatch='+',color='red',alpha=0.5))
    ax.add_patch(Rectangle((n-10, 0), 5, n-5, fill=True, hatch='+',color='red',alpha=0.5))
    return

if __name__ == "__main__":

    #Set global attenuation factor (unitless)
    
    att_bool = True
    connectivity_bool = True

    dem_res = 30
    
    land_thresh = 0.2
    domain_size = 50
    max_elev = 3
    min_elev = -1
    std = 7
    slr = args.slr #0.4
    slope = args.slope

    X, Y, dem_arr = get_dem_array(n = domain_size, max_elev=max_elev, min_elev=min_elev, slope = slope) # float arr
    land_arr = dem_arr >= land_thresh # bool arr
    water_arr = get_water_array(dem_arr.shape[0], slr = slr) # float arr
    protection_arr = get_protection_array(dem_arr.shape[0]) # float arr

    seawall_idx = get_seawall_idx()
    build_seawall(protection_arr, seawall_idx = seawall_idx, elev = max_elev)

    # build seawall
    # protection_arr[np.s_[-10:-5,:-5]] = max_elev
    # protection_arr[np.s_[:-5,-10:-5]] = max_elev

    #Get attenuation arr
    att_val = args.att_val*att_bool
    att_arr = dem_arr*0+att_val # numpy array

    # arr_list = [dem_arr,land_arr,water_arr,protection_arr]
    # title_list = ['DEM (float)',f'Land > {land_thresh:.2f} (bool)',f'SLR = {slr:.2f} (float)', 'Protection (float)']
    
    # print('start_set_min:',start_set_min)
    print(f'min DEM: {dem_arr.min()}, max DEM: {dem_arr.max()}, dem shape: {dem_arr.shape}')

    bolean_min = np.zeros(dem_arr.shape, dtype = bool)
    
    fig, axes = plt.subplots(1,2, subplot_kw={"projection": "3d"})
    for ax in axes:
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        ax.set_zlabel('z')

    print(f'X, Y shape: {X.shape}')
    # plot DEM
    surf = axes[0].plot_surface(X,Y, dem_arr, vmin = dem_arr.min(), cmap = cm.viridis)
    # plot water level plane
    surf_water = np.ones(X.shape)*slr
    axes[0].plot_surface(X,Y, surf_water, cmap = cm.Blues, alpha=0.7)
    # plot protection
    axes[0].plot_surface(X,Y, protection_arr, cmap = cm.Reds,alpha=0.7,label='seawall')

    
    fig.colorbar(surf, shrink = 0.5, aspect=5)
    axes[0].set_title(f'DEM\nmax: {max_elev:.2f}, min: {min_elev:.2f}')
    zlim = axes[0].get_zlim()
    
   
    # """
    # identify the start set min on land array
    # """
    
    #Exploit minwater to get a starting point for flooding efficiently.
    start_set_min = getStartSet(water_arr, dem_arr, protection_arr, land_arr) # The start_set indicates the pixel coordinates (int tuples) to continue progressive flooding.
    checkstack, bolean = getCoastlineContinuationSet(water_arr,dem_arr, bolean_min, start_set_min, protection_arr)
    # bolean (bool) used as integrated attenuation.  If bool, stores 1,0 of whether in sea or not, respectively. This is basically a starting for progressive flooding
    # if float, If float, stores the cumulative attenuation to given point.
    water_depth, att_integral, checkstack = connectedAttenuatedBathtub(water_arr, dem_arr, protection_arr, att_arr, bolean, checkstack, dem_res = dem_res, threshold = 0.01)
    flooded_area = water_depth*land_arr
    # print(f'att array: {att_arr}')
    print(f'att integral min: {att_integral.min()}; att integral max: {att_integral.max()}')

    # checkstack_x = []
    # checkstack_y = []
    # checkstack_z = []
    start_set_min_arr = np.zeros(dem_arr.shape,dtype=bool)
    for i,j in checkstack:
    #     checkstack_x.append(X[i,j])
    #     checkstack_y.append(Y[i,j])
    #     checkstack_z.append(dem_arr[i,j])
        start_set_min_arr[i,j] = True

    # axes[0].scatter(checkstack_x, checkstack_y, checkstack_z, color='red',alpha=0.5, label = 'to be checked for flooding')

    axes[1].plot_surface(X,Y, -water_depth) # invert water depth to represent flooded waters visually
    axes[1].set_title(f'Water depth\nSLR: {slr}, max: {water_depth.max():.2f}, min: {water_depth.min():.2f}')
    axes[1].set_zlim(zlim[0],zlim[1])
    # plot protection
    axes[1].plot_surface(X,Y, protection_arr, cmap = cm.Reds)

    plt.legend()
    plt.show()
    ax_width = [1, 0.1, 1, 1, 1, 1, 0.1]
    fig, axes = plt.subplots(1,7, figsize=(15,5), gridspec_kw={'width_ratios': ax_width})
    # remove axis outline
    [ax.axis('off') for i, ax in enumerate(axes.flatten()) if ax_width[i]>0.1]
    # add seawall to plot
    # [plot_seawall(ax,domain_size) for i, ax in enumerate(axes.flatten()) if ax_width[i]>0.1]
    # plot DEM
    im = axes[0].imshow(dem_arr)
    cbar = plt.colorbar(im, cax=axes[1])
    cbar.ax.set_ylabel('DEM (m)')
    axes[0].set_title(f'DEM')
    # plot coords to be checked for flooding
    axes[2].imshow(start_set_min_arr, cmap='Greys_r') # contains coordinates to be checked for flooding
    axes[2].set_title('Coords to be checked for flooding')
    # plot coords that have been checked for flooding
    axes[3].imshow(bolean,cmap='Greys_r') # indicates if a pixel is checked for flooding or not # boolean array
    axes[3].set_title('Checked for flooding')
    # plot water depth
    im = axes[4].imshow(water_depth, cmap='Blues')
    axes[4].set_title('Water depth')
    vmin, vmax = im.get_clim()
    # plot flooded area
    axes[5].imshow(flooded_area, cmap='Blues', vmin=vmin, vmax=vmax)
    axes[5].set_title('Flooded land area')
    cbar = plt.colorbar(im, cax=axes[6])
    cbar.ax.set_ylabel('flood depth (m)')

    fig.suptitle(f'lowest DEM: {dem_arr.min():.3f}, SLR: {slr:.3f}\natt_val: {args.att_val:.5f}, att_integral: {att_integral.max():.2f}\n max water depth: {water_depth.max():.3f}')
    plt.tight_layout()
    plt.show()

    # plot flooded land area

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    # surf = ax.plot_surface(X,Y, -flooded_area, vmin = flooded_area.min(),alpha=0.5)
    # # ax.plot_surface(X,Y, dem_arr, vmin = dem_arr.min())
    # # ax.plot_surface(X,Y, water_depth,alpha=0.5)
    # fig.colorbar(surf, shrink = 0.5, aspect=5)
    # ax.set_title('Flooded area')
    # plt.show()




    