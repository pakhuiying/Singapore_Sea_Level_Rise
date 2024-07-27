"""22/09/2023
Source of this script: Kasmalkar et al. (2023)
Run script for improved bathtub inundation model as per Kasmalkar et al. (2023)

Usage:
python run.py <folder>

folder: str. Folder name of one of the folders, Bac_lieu, Bay_are, Dutch_islands, Hamburg, Misrata.

Upon execution, a water_depth_RP1000.tif will be generated in the respective folder in 10-12 minutes. 
This .tif file (which can be opened in QGIS), will represent the water depth in meters.


"""

from bathtub import *
import numpy as np
import sys
import time


#Set global attenuation factor (unitless)
att_val = 0.0002
att_bool = True
RCP = '26'
year = '2020'
RP = '1000'
fldr = sys.argv[1]+'/'
connectivity_bool = True

#Set up basic files.
time_other = time.time()
dem =  Raster(fldr+'dem.tif')
dem_res = 30
if dem.resolution[0] < 0.0001:
	dem_res = 10
dem[dem.isna()] = 0.0
dem [(dem <= 1e-37)*(dem >= 0)] = 0.0
dem [dem <= -1e5] = 0.0
protection = dem*0

global_input = 'global_inputs/'
dem_arr = dem.data() # numpy array

#Assume zero protection.
protection_arr = dem*0 -100
	  
#Minwater file is the minimum of RCP26_2020, RCP45_2020, and RCP85_2020 and RP1, as a computational starting point of flood extent.
minwater = Raster(global_input+'minwater.tif')
  
#Sea file indicates raster of the global sea, used to remove places in the sea.
local_land = Raster(global_input+'global_sea_layer.tif').matchRefMap(dem, 'cutoff') # global sea layer is a binary file, 1 = sea, 0 = land
local_land = -local_land+1
local_land = local_land > 0.5 # boolean array


dem_arr = dem.data() # numpy array

protection_arr = protection.data() # numpy array

#Get attenuation arr
att_val = 0.0002*att_bool
att_arr =dem_arr*0+att_val # numpy array

#Get water array.
scenario = Raster(global_input+'RP{}.tif'.format(RP)).matchRefMap(dem, 'cutoff')
SLR = Raster(global_input+'RCP{}_{}.tif'.format(RCP,year)).matchRefMap(dem, 'cutoff')
local_minwater = minwater.matchRefMap(dem, 'cutoff')

water = scenario+SLR
water_arr = water.data() # numpy array

#Exploit minwater to get a starting point for flooding efficiently.
start_set_min = getStartSet(local_minwater.data(), dem_arr, protection_arr, local_land.data())
bolean_min = np.zeros(dem_arr.shape, dtype = bool)

print('Other input data prepared in {:.1f} secs.'.format(time.time() - time_other))

time_prop = time.time()

if connectivity_bool:

	local_land_arr = local_land.data()
	#We first find the edge of the DEM.

	checkstack, bolean = getCoastlineContinuationSet(SLR.data(),dem_arr, bolean_min, start_set_min, protection_arr)
	water_depth, att_integral, checkstack = connectedAttenuatedBathtub(water_arr, dem_arr, protection_arr, att_arr, bolean, checkstack, dem_res = dem_res, threshold = 0.01)



else:
	time_prop = time.time()
	water_depth = np.maximum(water_arr - dem_arr - 0.01, 0)*(water_arr > protection_arr)


		
floodmap = dem.copy()
floodmap[:] = water_depth*local_land_arr
floodmap.writeNodata(0)
floodmap.writeRaster(filename = fldr+'/water_depth_RP{}.tif'.format(RP))

print('Raster prepared and stored in {:.1f} secs'.format(time.time() - time_prop))