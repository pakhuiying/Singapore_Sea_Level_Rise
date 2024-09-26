"""22/09/2023
Source of this script: Kasmalkar et al. (2023)
Run script for improved bathtub inundation model as per Kasmalkar et al. (2023)

Usage:
python run.py <folder>

folder: str. Folder name of one of the folders, Bac_lieu, Bay_are, Dutch_islands, Hamburg, Misrata.

Upon execution, a water_depth_RP1000.tif will be generated in the respective folder in 10-12 minutes. 
This .tif file (which can be opened in QGIS), will represent the water depth in meters.

to run this script in the command line:
	python run_sg_30m.py (this command runs the default arguments, where scenario is SSP1, att_val is 0.0002)
	python run_sg_30m.py --scenario SSP2 (this command specifies the type of SSP scenario)
	python run_sg_30m.py --slr (this command specifies the slr level, which will override the scenario argument, which allows for better customisation of SLR)
"""

from bathtub import *
import numpy as np
import sys
import time
import os
import argparse

# parse arguments
parser = argparse.ArgumentParser(description='Generate SLR based on bathtub algorithm')
parser.add_argument('--scenario',type=str, default='SSP1')
parser.add_argument('--att_val',type=float, default=0.0002)
parser.add_argument('--slr',type=float, required=False, default=None)
args = parser.parse_args()

#Set global attenuation factor (unitless)
att_val = 0.0002
att_bool = True
RCP = '26'
year = '2100'

# projected SLR for each SSP scenarios for Singapore
SSP1 = 0.45 # SSP1-2.6
SSP2 = 0.57 # SSP2-4.5
SSP5 = 0.79 #SSP5-8.5
# high impact low likelihood event
"""
Sea levels could rise by even as high as 4m to 5m, if we take into account all the factors, 
including coastal surges, extreme high tides and land subsidence, 
said Ms Hazel Khoo, director of national water agency PUB's Coastal Protection Department, during an ST webinar
"""
HILL_low = 4.0
HILL_high = 5.0

# return period e.g. 1 in 1000 years
RP = '1000'
connectivity_bool = True
store_dir = 'Singapore'
land_thresh = 0.0

#Set up basic files.
time_other = time.time()
dem =  Raster(r"C:\Users\hypak\OneDrive - Singapore Management University\Documents\Data\FABDEM_SG_30m\FABDEM_SG_30m_Clipped.tif")
dem_res = 30
if dem.resolution[0] < 0.0001:
	dem_res = 10
	
dem[dem.isna()] = 0.0
dem [(dem <= 1e-37)*(dem >= 0)] = 0.0
dem [dem <= -1e5] = 0.0

# protection
protection = dem*0

global_input = 'global_inputs'
dem_arr = dem.data() # numpy array

#Assume zero protection.
protection_arr = dem*0 -100
	  
#Minwater file is the minimum of RCP26_2020, RCP45_2020, and RCP85_2020 and RP1, as a computational starting point of flood extent.
minwater = Raster(os.path.join(global_input,'minwater.tif'))
  
#Sea file indicates raster of the global sea, used to remove places in the sea.
local_land = Raster(os.path.join(global_input,'global_sea_layer.tif')).matchRefMap(dem, 'cutoff') # global sea layer is a binary file, 1 = sea, 0 = land
local_land = -local_land+1
local_land = local_land > 0.5 # boolean array


dem_arr = dem.data() # numpy array
# no data mask
noDataMask = dem_arr < -98 # boolean array, where True = noData = -99
validSGMask = dem_arr > -98 # boolean array, where True = SG land, False = outside SG
print(f'validSGmask dtype: {validSGMask.dtype}')
# if noData is -99, set as 0.0
dem_arr[dem_arr<-98] = 0.0

protection_arr = protection.data() # numpy array

#Get attenuation arr
att_val = 0.0002*att_bool
att_arr =dem_arr*0+att_val # numpy array

#Get water array.
# scenario = Raster(global_input+'RP{}.tif'.format(RP)).matchRefMap(dem, 'cutoff')
# SLR = Raster(global_input+'RCP{}_{}.tif'.format(RCP,year)).matchRefMap(dem, 'cutoff')
if args.scenario == 'SSP1':
	slr = SSP1
elif args.scenario == 'SSP2':
	slr = SSP2
elif args.scenario == "SSP5":
	slr = SSP5
elif args.scenario == "HILL_low":
	slr = HILL_low
else:
	slr = HILL_high

if args.slr is not None:
	# if slr is supplied, override the slr provided by scenarios
	slr = args.slr

water_arr = np.zeros(dem_arr.shape) + slr
# current sea level
local_minwater = minwater.matchRefMap(dem, 'cutoff')
print(f'local land arr min & max: {local_minwater.min():.3f}m, {local_minwater.max():.3f}')
# water = scenario+SLR
# water_arr = water.data() # numpy array

#Exploit minwater to get a starting point for flooding efficiently.
start_set_min = getStartSet(local_minwater.data(), dem_arr, protection_arr, local_land.data())
bolean_min = np.zeros(dem_arr.shape, dtype = bool)

print('Other input data prepared in {:.1f} secs.'.format(time.time() - time_other))

time_prop = time.time()

if connectivity_bool:

	local_land_arr = local_land.data() #bool data
	# print(f'local land arr dtype: {local_land_arr.dtype}')
	
    #intersect SG mask with land area mask
	local_land_arr = local_land_arr*validSGMask
	
	#We first find the edge of the DEM.

	checkstack, bolean = getCoastlineContinuationSet(water_arr,dem_arr, bolean_min, start_set_min, protection_arr)
	water_depth, att_integral, checkstack = connectedAttenuatedBathtub(water_arr, dem_arr, protection_arr, att_arr, bolean, checkstack, dem_res = dem_res, threshold = 0.01)



else:
	time_prop = time.time()
	water_depth = np.maximum(water_arr - dem_arr - 0.01, 0)*(water_arr > protection_arr)

floodmap = dem.copy()
floodmap[:] = water_depth*local_land_arr # creates a copy. Clips the land boolean area with the floodmap to show which the water depth on the land array
# floodmap[np.where(noDataMask)] = 0 # assign -99 to noData mask
floodmap.writeNodata(0)

fn = os.path.join(store_dir,f'water_depth_year{year}_{args.scenario}.tif')
if args.slr is not None:
	# if slr is supplied, override the slr provided by scenarios
	fn = os.path.join(store_dir,f'water_depth_{int(slr)}.tif')

floodmap.writeRaster(filename = fn)
# floodmap.writeRaster(filename = fldr+'/water_depth_RP{}.tif'.format(RP))

print('Raster prepared and stored in {:.1f} secs\nRasters stored in: {}'.format(time.time() - time_prop, store_dir))