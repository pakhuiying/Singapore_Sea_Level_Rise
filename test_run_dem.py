from bathtub import *
import numpy as np
import sys
import time
# from scipy import signal
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib import cbook, cm
from matplotlib.patches import Rectangle
from matplotlib.colors import LightSource
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

def get_dem_array(min_elev=-1):
    # Load and format data
    dem = cbook.get_sample_data('jacksboro_fault_dem.npz')
    z = dem['elevation']
    nrows, ncols = z.shape
    x = np.linspace(dem['xmin'], dem['xmax'], ncols)
    y = np.linspace(dem['ymin'], dem['ymax'], nrows)
    x, y = np.meshgrid(x, y)

    region = np.s_[5:50, 5:50] # returns a tuple of slice objects, useful to indx array afterwards
    x, y, z = x[region], y[region], z[region]
    z = z - z.min() + min_elev
    return x,y,z

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

def plot_dem(ax,x,y,z):
    # Set up plot
    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                        linewidth=0, antialiased=False, shade=False)

    return surf

def plot_water(ax, water_arr):
    surf = ax.plot_surface(X,Y, water_arr, cmap = cm.Blues, alpha=0.7)
    return surf

if __name__ == "__main__":

    #Set global attenuation factor (unitless)
    
    att_bool = True
    connectivity_bool = True

    dem_res = 30
    
    land_thresh = 0.0
    min_elev = -50
    slr = args.slr #0.4

    X, Y, dem_arr = get_dem_array(min_elev=min_elev) # float arr
    land_arr = dem_arr >= land_thresh # bool arr
    water_arr = get_water_array(dem_arr.shape[0], slr = slr) # float arr
    protection_arr = get_protection_array(dem_arr.shape[0]) # float arr
    bolean_min = np.zeros(dem_arr.shape, dtype = bool)

    # seawall_idx = get_seawall_idx()
    # build_seawall(protection_arr, seawall_idx = seawall_idx, elev = max_elev)

    # build seawall
    # protection_arr[np.s_[-10:-5,:-5]] = max_elev
    # protection_arr[np.s_[:-5,-10:-5]] = max_elev

    #Get attenuation arr
    att_val = args.att_val*att_bool
    att_arr = dem_arr*0+att_val # numpy array

    # start 3D plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    surf = plot_dem(ax, X, Y, dem_arr)
    _ = plot_water(ax,water_arr=water_arr)
    ax.set_title('DEM (m)')
    fig.colorbar(surf, shrink = 0.5, aspect=5)
    ax.set_title(f'DEM\nmax: {dem_arr.max():.2f}, min: {min_elev:.2f}')
    plt.show()

    # start 2D plot
    #Exploit minwater to get a starting point for flooding efficiently.
    start_set_min = getStartSet(water_arr, dem_arr, protection_arr, land_arr) # The start_set indicates the pixel coordinates (int tuples) to continue progressive flooding.
    checkstack, bolean = getCoastlineContinuationSet(water_arr,dem_arr, bolean_min, start_set_min, protection_arr)
    # bolean (bool) used as integrated attenuation.  If bool, stores 1,0 of whether in sea or not, respectively. This is basically a starting for progressive flooding
    # if float, If float, stores the cumulative attenuation to given point.
    water_depth, att_integral, checkstack = connectedAttenuatedBathtub(water_arr, dem_arr, protection_arr, att_arr, bolean, checkstack, dem_res = dem_res, threshold = 0.01)
    flooded_area = water_depth*land_arr
    # print(f'att array: {att_arr}')
    print(f'att integral min: {att_integral.min()}; att integral max: {att_integral.max()}')


    start_set_min_arr = np.zeros(dem_arr.shape,dtype=bool)
    for i,j in checkstack:
        start_set_min_arr[i,j] = True

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

    fig.suptitle(f'lowest DEM: {dem_arr.min():.3f}, SLR: {slr:.3f}\n max water depth: {water_depth.max():.3f}')
    plt.tight_layout()
    plt.show()
