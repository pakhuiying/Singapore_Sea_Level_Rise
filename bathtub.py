"""
02/01/2023
Author: Neel
All copyright and exclusive use belongs to Ventrx.

Algorithm script for bathtub inundation model to propagate flooding.
"""

"""Load local.py"""
import os

from itertools import chain
from scipy.ndimage import convolve as convolveBF
import numpy as np
import rioxarray as rio
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import pyproj
import xarray


DEM_FLOOR = -10000.0
FLOOD_FLOOR = 0.0

def connectedBathtub(water_arr,dem_arr,protection_arr,bolean, checkstack):
	"""Function to perform connected bathtub inundation.
	We say a pixel is 'apparently flooded' if the difference between the water height and the DEM height at that pixel is positive. 
		Apparent water level = water height - DEM height.
	If protection levels are involved, we say a pixel is 'apparently flooded' if the difference betweej water height and the DEM height at that pixel exceeds the protection height.
		Apparent water level = water height - DEM height.
	We consider an edge pixel to be 'flooded' if a certain number of neighboring edge pixels are apparently flooded. 
		The number of neighbors to each side is determined by @a window.
		The window is useful for filtering out rivers and other narrow troughs intersecting the edge that may be apparently flooded.
	We consider an internal pixel to be flooded if there is a path of flooded pixels from the given pixel to a flooded edge pixel.


	To see how the algorithm works, read the inline comments.
	****************************************
	@param water_arr: Numpy float array. Flood map water elevation values.
	@param dem_arr: Numpy float array. DEM values.
	@param protection_arr: Numpy float array or None. Protection values.
	@pre water_arr.shape == dem_arr.shape == protection_arr.shape == bolean.shape
	@param bolean: boolean array. Indicates if a point has been previously checked or not. 
								  Does not include non-flooded points.
	@param checkstack: set or None. If not None, set of tuples of raster pixel index coordinates.
						If None, we assume all the edge pixels might be potentially connected to the sea.
	@returns numpy float array:the flood map array, and a set of int tuples: @a resstack, 
			indicating pixels that were not apparently flooded but which neighbored flooded pixels.
			@a resstack is used for continuing connected bathtub flooding for the next water level.
	"""

	r = water_arr.shape[0]
	s = water_arr.shape[1]

	resstack = set()
	water_depth = np.maximum(water_arr-dem_arr,0)

	water_depth[water_arr <= protection_arr] = 0.


	"""Iterate over the stack. 
	If water level is above zero for a given pixel, add to the stack all its 8 neighbors,
	but only if they have not already been iterated over (bolean stores that information).
	"""
	while len(checkstack) > 0:
		i,j = checkstack.pop()
		if i< 0 or i >= r:
			continue
		if j < 0 or j >= s:
			continue
		if bolean[i,j] == 0: #if point has not been checked
			bolean[i,j] = 1 # mark it as checked
			if water_depth[i,j] > 0: # if current location's water depth is +ve, add its neighbour for checking
				checkstack.add((i,j+1))
				checkstack.add((i,j-1))
				checkstack.add((i+1,j))
				checkstack.add((i-1,j))
				checkstack.add((i+1,j+1))
				checkstack.add((i-1,j+1))
				checkstack.add((i+1,j-1))
				checkstack.add((i-1,j-1))
			else:
				resstack.add((i,j))


	"""Multiply by bolean so that 'flooded' pixels that are unreachable from the borders are removed."""
	water_depth = water_depth*bolean # to obtain existing water depth that has been checked
	return water_depth, resstack

def getStartSet(water_arr, dem_arr, protection_arr = None, land_arr = None, window = 2):
	"""Function to get an initial start set, which can simplify the checking of future maps.
	****************************************
	@param water_arr: Numpy array. Indicates a baseline water level map for identifing flooded pixels.
	@param dem_arr: Numpy array. Indicates topographic elevation.
	@param protection_arr: Numpy array or None. If array, indicates protection levels at each pixel.
	@param land_arr: Numpy array or None. If not None, indicates sea extent to identify raster edge pixels that are flooded.
						0 stands for sea.
	@param window: int. If land_arr is None, takes all edge pixels as start points with a continuous window.
	@returns start_set: set of int tuples. 
				The start_set indicates the pixel coordinates (int tuples) to continue progressive flooding.
	"""
	start_set = set()
	if protection_arr is None:
		protection_arr = np.zeros(dem_arr.shape, dtype = float)

	r = dem_arr.shape[0]
	s = dem_arr.shape[1]

	water_depth = np.maximum(water_arr-dem_arr,0)
	water_depth = water_depth*(water_arr > protection_arr)

	#If nothing to distinguish land, just consider edge pixels and see if water depth exceeds elevation and protection.
	if land_arr is None:

		if np.min(water_depth[0:window,0])>0 and np.min(water_depth[0,0:window])>0:
			start_set.add((0,0))

		if np.min(water_depth[r-1-window:r-1,0])>0 and np.min(water_depth[r-1,0:window])>0:
			start_set.add((r-1,0))

		if np.min(water_depth[0:window,s-1])>0 and np.min(water_depth[0,s-1-window:s-1])>0:
			start_set.add((0,s-1))

		if np.min(water_depth[r-1-window:r-1,s-1])>0 and np.min(water_depth[r-1,s-1-window:s-1])>0:
			v.add((r-1,s-1))

		"""Add the non-corner edge points."""
		for i in range(1,r-1):
			if np.min(water_depth[max(i-window+1,0):min(i+window,r),0]) > 0:
				start_set.add((i,0))

			if np.min(water_depth[max(i-window+1,0):min(i+window,r),s-1]) > 0:
				start_set.add((i,s-1))

		for j in range(1,s-1):
			if np.min(water_depth[0,max(j-window+1,0):min(j+window,s)]) > 0:
				start_set.add((0,j-1))

			if np.min(water_depth[r-1,max(j-window+1,0):min(j+window,s)]) > 0:
				start_set.add((r-1,j-1))
	else:
		start_set = set()
		
		for i in range(r):
			if land_arr[i,0] < 0.5 and water_depth[i,0] > 0:
				start_set.add((i,0))
			if land_arr[i,s-1] < 0.5 and water_depth[i,s-1] > 0:
				start_set.add((i,s-1))

		for j in range(s):
			if land_arr[0,j] < 0.5 and water_depth[0,j] > 0:
				start_set.add((0,j))
			if land_arr[r-1,j] < 0.5 and water_depth[r-1,j] > 0:
				start_set.add((r-1,j))

	return start_set


def getCoastlineContinuationSet(water_arr, dem_arr, bolean, checkstack, protection_arr = None):
	"""CPU Function to compute sea levels up to the coastline using a minimal water level map.
	****************************************
	@param water_arr: numpy float array. Flood water level values.
	@param dem_arr: numpy float array. Terrain elevation values.
	@param bolean: numpy boolean array. Indicates if a pixel is checked or not for flooding.
	@param checkstack: set. Contains pixel coordinates to be checked for flooding.
	@param protection_arr: numpy float array or None. Protection adds a threshold to flooding.
	@pre water_arr.shape == dem_arr.shape == bolean.shape.
	@pre If protection_arr is not None, water_arr.shape = protection_arr.shape.
	@returns resstack, bolean: set, numpy boolean array. 
			 resstack indicates frontier points not flooded. 
			 bolean is the set of flooded, checked points.
	"""
	if protection_arr is None:
		protection_arr = np.zeros(dem_arr.shape,dtype = np.float32)

	water_depth, resstack = connectedBathtub(water_arr, dem_arr, protection_arr = protection_arr, bolean = bolean.copy(), checkstack = checkstack.copy())
	return resstack, water_depth > 0


def connectedAttenuatedBathtub(water_arr, dem_arr, protection_arr, att_arr, att_integral, scstack, dem_res = 30, threshold = 0.01):
	"""Function to do connected bathtub inundation with attenuation given distance from coast.
	****************************************
	@param water_arr: Numpy float array. Water height values for a given scenario.
	@param dem_arr: Numpy float array. DEM values.
	@param protection_arr: Numpy float array. If float, height of protection infrastructure.
	@param att_arr: float. Attenuation. No unit. x meters of water height reduction per x meters of distance traveled by water.
	@param att_integral: Numpy float or bool array. If float, stores the cumulative attenuation to given point.
											  If bool, stores 1,0 of whether in sea or not, respectively. This is basically a starting for progressive flooding.
	@param scstack: SuperCheckStack object or set. If former, stores the pixels that should be checked for flood propagation.
													If latter, a new object is reated to cast the set into.
	@param dem_res: float. Resolution of the DEM in meters/pixel.
	@param threshold: float. Threshold water depth to be considered flooded.
	@returns: water_depth: Numpy float array with water depth values;
			  att_integral: Numpy float array with total path-attenuation to the point;
			  scstack: SuperCheckStack object with residual points for next round.
	"""

	srstack = SuperCheckStack()
	MAX_ATT = np.inf

	"""This is for compatibility. If scstack is a set, convert it to a SuperCheckStack object, assuming the set corresponds to points at distance 0."""
	if type(scstack) == set:
		temp = SuperCheckStack()
		temp.populate(scstack)
		scstack = temp

	"""If att_integral is boolean, 1 value corresponds to sea so set att_integral as 0, and 0 value as land, so set initial att_integration value to MAX_ATT."""
	if att_integral.dtype == bool or att_integral.dtype == np.uint8:
		att_integral = (1-att_integral).astype(float)
		att_integral[att_integral > 0] = MAX_ATT
	
	"""Refresh the dist values on the starting stack so that iterations can occur."""
	r = dem_arr.shape[0]
	s = dem_arr.shape[1]

	"""idx indicates which stack we are on. idx = 0 corresponds to distance 0, idx = 1 to distance 0.5, and linearly on."""
	idx = 0

	while(len(scstack)) > 0:
		checkstack = scstack.checklist[idx]
		m = max(int(2*(scstack.lower_list[idx]+1.5 - scstack.lower_list[-1])),0)
		scstack.extend(m)
		d = scstack.lower_list[idx]
		"""checkstack is the set with pixels at distance d.
		We flood them before farther away pixels."""
		while len(checkstack) > 0:
			x = checkstack.popitem()
			i,j = x[0]
			att = x[1]
			if i < 0 or i >= r:
				continue
			if j < 0 or j >= s:
				continue
			
			
			if att_integral[i,j] > att:
				# print(f'IF ATT_INTEGRAL {att_integral[i,j]} > ATT {att}')

				"""If the flood water exceeds the protection, and attentuated flood level exceeds the DEM, then propagate."""
				if water_arr[i,j] > protection_arr[i,j] and water_arr[i,j] >= threshold + dem_arr[i,j] + att:
					residual = {(i,j+1): att+ dem_res*att_arr[i,j],
								(i,j-1): att+ dem_res*att_arr[i,j],
								(i+1,j): att+ dem_res*att_arr[i,j],
								(i-1,j): att+ dem_res*att_arr[i,j]}
					scstack.populate(residual,d+1)
					residual = {(i+1,j+1): att+ 1.5*dem_res*att_arr[i,j],
								(i-1,j+1): att+ 1.5*dem_res*att_arr[i,j],
								(i+1,j-1): att+ 1.5*dem_res*att_arr[i,j],
								(i-1,j-1): att+ 1.5*dem_res*att_arr[i,j]}
					scstack.populate(residual,d+1.5)
					att_integral[i,j] = att
				else:
					srstack.populate({(i,j):att},d)		
		idx+=1


	"""Adjust for distance and protection_arr so that 'flooded' pixels that are unreachable from the borders are removed."""
	water_depth = np.maximum(0,water_arr-att_integral - dem_arr)
	water_depth = water_depth*(att_integral < MAX_ATT)
	return water_depth, att_integral, srstack

"""Data structure to perform attenuated connected bathtub inundation."""
class SuperCheckStack:
	def __init__(self):
		"""Class for holding Dijkstra information when performing attenuation based flood propagation.""" 
		self.checklist = [dict()]
		self.lower_list = [0.0]

	def populate(self,residual, dist = 0.0):
		"""Function to populate a SuperCheckStack from a set or dictionary.
		@param self: object.
		@param residual: set or dict. Contains pixels to put in checkstack, and if dict, contains attentuation value.
		@param dist: float. Starting distance for adding the residuals.
		@returns None.
		@post: fills self.checklist and self.lower_list.
		"""
		"""If residual is a set, assume it is the set of pixels at distance 0."""
		if type(residual) == set:
			self.checklist[0] = {(i,j):0.0 for (i,j) in residual}
			return

		"""If dictionary, we have attentuation values."""
		for (i,j) in residual:
			att = residual[(i,j)]
			ind = int(2*dist)
			if ind >= len(self.lower_list):
				spots = int(2*(dist - self.lower_list[-1]))
				self.extend(spots)
			if (i,j) not in self.checklist[ind]:
				self.checklist[ind][(i,j)] = att
			else:
				self.checklist[ind][(i,j)] = min(att,self.checklist[ind][(i,j)])

	
	def shift(self):
		"""Function to move ahead to next checklist."""
		self.checklist = self.checklist[1:]
		self.lower_list = self.lower_list[1:]

	def mergeDicts(self):
		"""Function to merge checklists to get minimum attenuation values.
		@param self: object.
		@returns vals: dict. It is the merging of all checklist dicts, keeping only the minimum values.
		"""
		vals = dict()
		for i in range(len(self.checklist)):
			for (x,y) in self.checklist[i]:
				if (x,y) in vals:
					vals[(x,y)] = min(vals[(x,y)], self.checklist[i][(x,y)])
				else:
					vals[(x,y)] = self.checklist[i][(x,y)]
		return vals


	def __iter__(self):
		"""Iteration representation."""
		it = iter(self.checklist[0])
		for st in self.checklist[1:]:
			it = chain(it, iter(st))
		return it

	def __len__(self):
		return sum(len(x) for x in self.checklist)

	def extend(self,k):
		"""Function to extend checklist so that more values can be appended to it.
		****************************************
		@param self: object.
		@param k: int. How much to append.
		@returns None.
		@post: updates self.checklist and self.lower_list.
		"""
		last_val = self.lower_list[-1]
		for i in range(k):
			self.checklist.append(dict())
			self.lower_list.append(last_val+0.5*(i+1))

"""Dictionary to translate strings to the Resampling module."""
resample_dict = {'bilinear': Resampling.bilinear,
				 'cubic': Resampling.cubic,
				 'mean': Resampling.average,
				 'median': Resampling.med,
				 'nearest': Resampling.nearest,
				 'sum': Resampling.sum}
		
def changeToNan(raster):
	"""Function to change float Xarray nodata to np.nan
		****************************************
		@param raster. xarray.DataArray of dtype np.float32.
		@returns xarray.DataArray nodata values are changed to np.nan.
		"""
	if not sisnan(raster.rio.nodata):
		raster = raster.where(raster != raster.rio.nodata, np.nan)
		raster.rio.write_nodata(np.nan, inplace = True)
	return raster	

def sisnan(x):
	if x is None:
		return False
	return np.isnan(x)

class Raster:
	"""
	Class to emulate the R raster package using Python modules, cupy, geopandas, matplotlib, numpy, pyproj, rasterio, rioxarray, scipy, xarray.
	"""
	
	def __init__(self, filename, nan = False):
		"""Constructor of Raster.
		****************************************
		@param self: object.
		@param filename: str or xarray.core.dataarray.DataArray. 
						 If str, Location of the raster.
						 if xarray, it is the raster data itself.
		@returns None.
		@post self.raster is xarray object with the raster at the location @a filename.
		@post self.crs, self.nodata, self.dim set from raster information.
		Exceptions if no file at @a filename, or if @a filename does not hold raster.
		"""

		#Check if filename is str of xarray.
		if type(filename) == str:
			if os.path.exists(filename):
				self.raster = rio.open_rasterio(filename)
			else:
				raise Exception('No such raster file: {}.'.format(filename))
		elif type(filename) == xarray.core.dataarray.DataArray:
			self.raster = filename.copy()
		else:
			raise Exception('Incorrect raster input\n {}'.format(filename))

		#Collect all raster metadata.
		self.bounds = self.raster.rio.bounds()
		self.crs = self.raster.rio.crs
		self.nodata = self.raster.rio.nodata
		self.raster = self.raster.squeeze() # to remove extra dimension so we only need a 2D array instead of a default 3D array
		self.resolution = self.raster.rio.resolution()
		self.transform = self.raster.rio.transform()
		self.shape = self.raster.shape

		#If nan, change data to nan.
		if nan:
			self.raster = changeToNan(self.raster)
			self.nodata = np.nan
		# if nan and self.raster.dtype == np.float32:
		# 	self.nan = True
		# else:
		# 	self.nan = False
		self.nan = False
	
	def __add__(self, other):
		"""Addition operation. Add underlying rasters.
		****************************************
		@param self: object.
		@param other: float or object.
		@returns object.
		"""

		#self.lazyNaN()


		#If float
		if type(other) in (float, int, np.float32, np.float64):
			other = float(other)
			r= Raster(self.raster + other)
		else:
			#other.lazyNaN()
			r = Raster(self.raster + other.raster)
		if sisnan(self.nodata):
			r.writeNodata(np.nan)
		return r

	def __eq__(self, other):
		"""Equality operation.
		****************************************
		@param self: object.
		@param other: float or object.
		@returns object with boolean raster, which is 1 where values are equal.
		"""
		#self.lazyNaN()

		if type(other) in (float, int, bool, np.float32, np.float64):
			val = self.raster == other
			return Raster(val)
		else:
			#other.lazyNaN()
			val = self.raster == other.raster
		return Raster(val)

	def __ge__(self, other):
		"""Greater than or equal operation.
		****************************************
		@param self: object.
		@param other: float or object.
		@returns object with boolean raster, and NaNs where either input is Nan.
		"""
		#self.lazyNaN()

		if type(other) in (float, int, np.float32, np.float64):
			other = float(other)
			val = self.raster >= other
		else:
			#other.lazyNaN()
			val = self.raster >= other.raster
		return Raster(val)

	def __getitem__(self, slicer):
		"""Get slice of Raster object.
		****************************************
		@param self: object.
		@param slicer. slice object or tuple or bool object.
						If slice object, returns single value array.
						If tuple, returns the slice of underlying array as a raster.
						If object, returns array with original values where bool object is True, and Nodata otherwise.
		@returns array. Warning. If no nodata, then slcing by a raster yields no change.
		"""
		#self.lazyNaN()

		if type(slicer) == slice:
			return self.raster[slicer].values
		elif type(slicer) == tuple:
			return self.raster[slicer].values
		elif type(slicer) == Raster:
			assert(self.shape == slicer.shape)
			if self.nodata is not None:
				return self.raster.where(slicer.raster, self.nodata).values
			else:
				return self.raster.copy()
		elif type(slicer) in (np.ndarray, xarray.core.dataarray.DataArray):
			assert(self.shape == slicer.shape)
			if self.nodata is not None:
				return self.raster.where(slicer, self.nodata).values
			else:
				return self.raster.copy()
		else:
			raise Exception('Incorrect slicing')

	def __gt__(self, other):
		"""Greater than operation.
		****************************************
		@param self: object.
		@param other: float or object.
		@returns object with boolean raster, and NaNs where either input is Nan.
		"""
		#self.lazyNaN()
		

		if type(other) in (float, int, np.float32, np.float64):
			other = float(other)
			val = self.raster > other
		else:
			#other.lazyNaN()
			val = self.raster > other.raster
		return Raster(val)

	def __invert__(self):
		"""Logical not operation.
		Makes sure nodata values are respected.
		****************************************
		@param self: boolean object.
		@returns object with negated raster values.
		"""
		return Raster(~self.raster)

	def __mul__(self, other):
		"""Multiplication operation. Multiply underlying rasters.
		Makes sure nodata values are respected.
		****************************************
		@param self: object.
		@param other: float or object.
		@returns object.
		"""
		#self.lazyNaN()

		#if float
		if type(other) in (float, int, np.float32, np.float64):
			other = float(other)
			r= Raster(self.raster * other)
		else:
			#other.lazyNaN()
			r= Raster(self.raster * other.raster)
		if sisnan(self.nodata):
			r.writeNodata(np.nan)
		return r

	def __ne__(self, other):
		"""Not equal than operation.
		****************************************
		@param self: object.
		@param other: float or object.
		@returns object with boolean raster, which is 1 where values are unequal.
		"""
		#self.lazyNaN()

		if type(other) in (float, int, bool, np.float32, np.float64):
			val = self.raster != other
			return Raster(val)
		else:
			#other.lazyNaN()
			val = self.raster != other.raster
		return Raster(val)

	def __neg__(self):
		"""Negative operation.
		Makes sure nodata values are respected.
		****************************************
		@param self: object.
		@returns object with negative raster values.
		"""
		#self.lazyNaN()

		return self*(-1.0)

	def __le__(self, other):
		"""Greater than operation.
		****************************************
		@param self: object.
		@param other: float or object.
		@returns object with boolean raster, and NaNs where either input is NaN.
		"""
		#self.lazyNaN()

		if type(other) in (float, int, np.float32, np.float64):
			other = float(other)
			val = self.raster <= other
		else:
			#other.lazyNaN()
			val = self.raster <= other.raster
		return Raster(val)

	def __lt__(self, other):
		"""Greater than operation.
		Makes sure nodata values are respected.
		****************************************
		@param self: object.
		@param other: float or object.
		@returns object with boolean raster, and NaNs where either input is Nan.
		"""
		#self.lazyNaN()

		if type(other) in (float, int, np.float32, np.float64):
			other = float(other)
			val = self.raster < other
		else:
			#other.lazyNaN()
			val = self.raster < other.raster
		return Raster(val)

	def __sub__(self, other):
		"""Subtraction operation. Subtract underlying rasters.
		Makes sure nodata values are respected.
		****************************************
		@param self: object.
		@param other: float or object.
		@returns object.
		"""
		#self.lazyNaN()

		if type(other) in (float, int, np.float32, np.float64):
			other = float(other)
			r = Raster(self.raster - other)
		else:
			#other.lazyNaN()
			r = Raster(self.raster - other.raster)
		if sisnan(self.nodata):
			r.writeNodata(np.nan)
		return r
	
	def __repr__(self):
		"""Repr function.
		****************************************
		@param self: object.
		@returns repr of self.raster.
		"""

		return self.raster.__repr__()

	def __setitem__(self, slicer, val):
		"""Set slice of Raster object.
		****************************************
		@param self: object.
		@param slicer. slice object or tuple, ndarray, xarray, or bool object.
						If slice object, sets @a val at single pixel
						If tuple, sets the slice of underlying array with @a val.
						If bool object or np.ndarray or xarray.core.dataarray.DataArray, 
						sets array where bool object is True, and Nodata otherwise.
		@pre if type(slicer) in (Raster, np.ndarrayxarray.core.dataarray.DataArray), 
				then slicer.shape == self.shape.
		@param val: int, float, array or xarray or raster.
		@returns None.
		@post self.raster is changed.
		"""
		#self.lazyNaN()

		#If slicer is array or Xarray, just convert it to Raster first.
		if type(slicer) in (np.ndarray, xarray.core.dataarray.DataArray):
			assert(self.shape == slicer.shape)
			r = self.copy()
			r.raster = r.raster.astype(bool)
			r.raster[:] = slicer[:]
			slicer = r

		#If val is Raster or xarray, retrieve the underlying array data.
		if type(val) == Raster:
			val = val.data()
		elif type(val) == xarray.core.dataarray.DataArray:
			val = val.values

		#If slice or tuple, this is standard slice assignment.
		if type(slicer) in (slice, tuple):
			self.raster[slicer] = val

		elif type(slicer) == Raster:
			assert(self.shape == slicer.shape)
			self.raster = self.raster.where(~slicer.raster, val)

		else:
			raise Exception('Incorrect slicing')

	def __str__(self):
		"""Str representation.
		****************************************
		@param self: object.
		@returns str representation of self.raster.
		"""
		return self.raster.__str__()

	def __truediv__(self, other):
		"""Division operation. Divide underlying rasters.
		Makes sure nodata values are respected.
		****************************************
		@param self: object.
		@param other: float or object.
		@returns object.
		"""
		#self.lazyNaN()

		if type(other) in (float, int):
			other = float(other)
			r= Raster(self.raster / other)

		else:
			#other.lazyNaN()
			r= Raster(self.raster / other.raster)

		#Check is self.nodata is np.nan
		if sisnan(self.nodata):
			r.writeNodata(np.nan)
		return r

	def abs(self):
		"""Get Raster with absolute value of all pixels.
		****************************************
		@param self: object.
		@param na_rm: bool. If True, does not average over Nodata values.
		@returns Raster, with np.nan as Nodata, and with values as np.abs(self.raster).
		"""
		#self.lazyNaN()

		return Raster(np.abs(self.raster))

	def aggregate(self, fact, fun = 'mean'):
		"""Aggregate a raster to a lower resolution which is a multiple of its original resolution.
		****************************************
		@param self: object.
		@param fact: int. The aggregation factor.
		@param fun: str. Indicates aggregation function, such as 'sum', 'mean', 'median',.
		@returns object, the aggregated raster.
		"""
		#self.lazyNaN()

		new_shape = (int(self.raster.rio.shape[0]/fact),int(self.raster.rio.shape[1]/fact))
		new_raster = self.raster.rio.reproject(self.crs, shape = new_shape, resampling = resample_dict[fun])
		return Raster(new_raster)

	def astype(self, cast):
		"""
		Function to change dtype of underlying raster.
		****************************************
		@param self: object.
		@param cast: either int, or float, or bool etc..
		@returns object with changed raster dtype.
		"""
		return Raster(self.raster.astype(cast))

	def changeNodata(self, nodata):
		"""Change the Nodata of the underlying raster.
		****************************************
		@param self: object.
		@param nodata: *. Change all existing nodata to this object.
		@returns object with Nodata value changed. 
			And all pixels with original Nodata value changed to new value.
		"""
		#Consider cases where nodata already matches self.nodata.
		if sisnan(self.nodata) and sisnan(nodata):
			return self

		if self.nodata == nodata:
			return self

		#If the nodata value is NaN, change it to nodata.
		if sisnan(self.nodata):
			raster = self.raster.where(~self.isna().raster, nodata)
		else:
			raster = self.raster.where(self.raster != self.nodata, nodata)
		raster.rio.write_nodata(nodata, inplace = True)
		return Raster(raster)

	def reproject(self, resolution = None, shape = None, method = 'bilinear'):
		"""Function to change the dimensions to an input resolution or raster.
		****************************************
		@param self: object.
		@param resolution: float 2-ple or None.
		@param shape: float 2-ple or None. If resolution is None, use shape.
		@returns object, resampled to match inputs.
		"""
		
		#If resolution is None, work with shape.
		if resolution is None:
			r = self.raster.rio.reproject(dst_crs = self.crs, shape = shape, resampling = resample_dict[method])
		else:
			r = self.raster.rio.reproject(dst_crs = self.crs, resolution = resolution, resampling = resample_dict[method])

		return Raster(r)

	def changeToNan(self):
		"""Function to change nodata to NaN safely.
		****************************************
		@param self: object.
		@returns None.
		@post converts nodata to NaN.
		"""
		self.raster = changeToNan(self.raster)
		self.nodata = np.nan

	def checkExtent(self, other):
		"""Function to check if Raster intersects other raster.
		****************************************
		@param self. object.
		@param other. Raster object.
		@returns bool. True of False of intersection.
		"""
		ext1 = self.bounds
		ext2 = other.bounds

		if ext1[0] > ext2[2]:
			return False

		if ext1[2] < ext2[0]:
			return False

		if ext1[1] > ext2[3]:
			return False

		if ext1[3] < ext2[1]:
			return False

		return True

	def contains(self, x,y):
		"""Check if a point x,y lies contains the bounds.
		****************************************
		@param self: object.
		@param x: float. x coordinate.
		@param y: float. y coordinate.
		@returns True if within bounds.
		"""
		return x > self.bounds[0] and x < self.bounds[2] and y > self.bounds[1] and y < self.bounds[3]

	def coordinates(self):
		"""Get the long table coordinates of the raster data.
		****************************************
		@param self: object.
		@returns Pandas DataFrame with 'x' and 'y' columns.
		"""
		return self.long()[['x','y']]

	def copy(self):
		"""Copy operation. Creates a raster copy.
		****************************************
		@param self: object.
		@returns object with the same raster copied.
		"""
		return Raster(self.raster.copy())


	def crop(self, other, snap = 'out', warp_bounds = False):
		"""Crop the raster.
		****************************************
		@param self: object.
		@param other: object or tuple. 
			If object, uses the raster extent for cropping. 
			If tuple, uses (xmin, ymin, xmax, ymax).
		@param snap: str. Either 'out', 'near', or 'ín', 
				indicating if pixels at the border are take or not.
				If 'out', the cropping exceeds the extent.
				If 'in', the cropping is inside the extent.
				If 'near', the cropping depends on whether the 'in' or 'out' is closer.
		@returns object with clipped raster.
		"""

		#Check if raster.
		if type(other) == Raster:
			other_bounds = other.bounds
			if self.crs != other.crs:
				if warp_bounds:
					other_bounds = transform_bounds(other.crs, self.crs, *other_bounds)
				else:
					raise Exception('Rasters do not have same CRS.')
			xmin, ymin, xmax,ymax = other_bounds
		else:
			#If extent, get the extent values from the iterable.
			try:
				xmin = other[0]
				ymin = other[1]
				xmax = other[2]
				ymax = other[3]
			except:
				raise Exception('Object is neither raster nor extent. Cannot crop.')

			if warp_bounds:
				other = transform_bounds(pycrs('EPSG:4326'), self.crs, *other)
				xmin = other[0]
				ymin = other[1]
				xmax = other[2]
				ymax = other[3]
		
		raster = self.raster.rio.clip_box(minx = xmin, miny = ymin, maxx = xmax, maxy = ymax)
		
		bounds = raster.rio.bounds()
		obounds = [xmin, ymin, xmax, ymax]
		resolution = raster.rio.resolution()
		#Default clip_box behavior is snap 'out'.

		if snap == 'near':
			#Check if bounds beyond halfway.
			bounds = raster.rio.bounds()
			if obounds[0] > bounds[0] + abs(resolution[0])/2:
				raster = raster[:,1:]

			if obounds[1] > bounds[1] + abs(resolution[1])/2:
				raster = raster[:-1,:]

			if obounds[2] < bounds[2] - abs(resolution[0])/2:
				raster = raster[:,:-1]

			if obounds[3] < bounds[3] - abs(resolution[1])/2:
				raster = raster[1:,:]
		elif snap == 'in':
			#Check if bounds strictly greater
			bounds = raster.rio.bounds()
			if obounds[0] > bounds[0]:
				raster = raster[:,1:]

			if obounds[1] > bounds[1]:
				raster = raster[:-1,:]

			if obounds[2] < bounds[2]:
				raster = raster[:,:-1]

			if obounds[3] < bounds[3]:
				print('remove top')
				raster = raster[1:,:]

		return Raster(raster)

	def cutOff(self, cutoff, lower = True):
		"""
		Function to set a cutoff to np.nan.
		****************************************
		@param self: object. Currently only supports float data.
		@param cutoff: float. Value of cutoff.
		@param lower: bool. If lower, anything below @a cutoff is set to np.nan.
		@returns object, with cutoff exceeded values set to np.nan. And raster nodata set to np.nan.
		"""
		r = self.copy()
		if lower:
			r[r < cutoff] = np.nan
		else:
			r[r > cutoff] = np.nan
		r.raster.rio.write_nodata(np.nan, inplace = True)
		r.nodata = np.nan
		return r

	def disaggregate(self, fact, method = 'bilinear'):
		"""Disaggregate a raster to a lower resolution which is an integer factor of its original resolution.
		****************************************
		@param self: object.
		@param fact: int. The disaggregation factor.
		@param fun: str. Indicates disaggregation method, such as 'bilinear', 'cubic'.
		@returns object, the disaggregated raster.
		"""
		#self.lazyNaN()

		new_shape = (int(self.raster.rio.shape[0]*fact),int(self.raster.rio.shape[1]*fact))
		new_raster = self.raster.rio.reproject(self.crs, shape = new_shape, resampling = resample_dict[method])
		return Raster(new_raster)

	def data(self, nodata = False):
		"""Get the data matrix of the raster.
		****************************************
		@param self: object.
		@param nodata: bool. If True, converts the Nodata into np.nan and returns the array.
		@returns Numpy array with the data.
		"""
		#self.lazyNaN()

		if nodata:
			if sisnan(self.nodata):
				return self.raster.data.copy()
			else:
				temp_raster = self.raster.where(self.raster != self.nodata, np.nan)
			return temp_raster.data
		return self.raster.data.copy()


	def edge(self):
		"""Function to compute the edge of a boolean raster, where 1 is in and 0 is out.
		****************************************
		@param self: object. Boolean raster
		@returns object, with 1 for edge and 0 otherwise.
		"""
		r = -self.astype(float)+1
		window = np.ones((3,3), dtype = float)
		edges = r.focal(window, na_rm = True, pad = True, padValue = 0., nan = False)
		return (edges > 0)*self


	def get(self,x,y, error = True):
		"""Function to get values at coordinates.
		@param x: float or Numpy float array. x coordinates. Longitude.
		@param y: float or Numpy float array. y coordinate. Latitude.
		@param error: bool. If True, returns error if coordinates outside. If False, returns 0.
		@returns raster value at coordinates.
		"""
		#self.lazyNaN()

		if type(x) in (np.ndarray, pd.Series):
			i,j = self.pixel(x,y, error = error)
			blank_val = 0
			if self.nodata is not None:
				blank_val = self.nodata

			for r in range(max(i)):
				if r not in i:
					break

			change = (i < 0) | (i >= self.shape[0]) | (j < 0) | (j >= self.shape[1])
			i[change] = 0
			j[change] = 0

			temp_val = self[r,0]
			self[r,0] = blank_val
			vals = self.data()[i,j]
			self[r,0] = temp_val
			return vals

		i,j = self.pixel(x,y, error = error)
		if (i >= 0) and (i < self.shape[0]) and (j >= 0) and (j < self.shape[1]):
			return self[i,j][()]
		return 0

		
	def getValues(self):
		"""Get the long table values of the raster data.
		****************************************
		@param self: object.
		@returns Pandas Series with 'z' data.
		"""
		#self.lazyNaN()
		return self.long().z

	def checkIntersection(self, other, snap = 'out', warp_bounds = False, res_safety = True):
		"""Function to check intersections. This is useful prior to cropping.
		****************************************
		@param self: object.
		@param other: object, or 4-tuple of float. In the latter case, it is the bounds.
		@param snap: str. Either 'out', 'near', or 'in'.
		@returns bool. The cropped bounds should have at least 2 pixels in each dimension.
		"""

		if type(other) == Raster:
			bounds = other.bounds
			if self.crs != other.crs:
				if warp_bounds:
					bounds = transform_bounds(other.crs, self.crs, *bounds)
				else:
					raise Exception('Rasters do not have same CRS.')
		
		else:
			bounds = other
			if warp_bounds:
				bounds = transform_bounds(pycrs('EPSG:4326'), self.crs, *bounds)

		x_min = max(self.bounds[0], bounds[0])
		x_max = min(self.bounds[2],bounds[2])
		y_min = max(self.bounds[1], bounds[1])
		y_max = min(self.bounds[3],bounds[3])

		if snap == 'out':
			return (x_min +abs(self.resolution[0])*1.01 < x_max) and (y_min +abs(self.resolution[1])*1.01 < y_max)

		if snap == 'in':
			return (x_min +2*abs(self.resolution[0])*1.01 < x_max) and (y_min +2*abs(self.resolution[1])*1.01 < y_max)

		if snap == 'near':
			return (x_min +1.5*abs(self.resolution[0])*1.01 < x_max) and (y_min +1.5*abs(self.resolution[1])*1.01 < y_max)  

	def isna(self):
		"""Given a raster, compute a boolean raster of whether a pixel is NaN or not.
		****************************************
		@param self: object.
		@returns object with a boolean raster.
		"""
		#self.lazyNaN()

		na_raster = np.isnan(self.raster)

		#Make sure to remove nodata value for the boolean raster.
		na_raster.rio.write_nodata(None, inplace = True)
		return Raster(na_raster)

	def lazyNaN(self):
		"""If np.float32 raster, Change the Nodata of the underlying raster in place.
		****************************************
		@param self: object.
		@returns None.
		@post changes raster to have np.nan as Nodata value. Also changes self.nodata value.
		"""
		return
		if not self.nan:
			return

		if sisnan(self.nodata):
			return

		self.raster = self.raster.where(self.raster != self.nodata, np.nan)
		self.raster.rio.write_nodata(np.nan, inplace = True)
		self.nodata = np.nan
		return


	def matchRefMap(self, other, nan = False, method = 'bilinear'):
		"""Crop and resample one raster based on a template raster.
		****************************************
		@param self: Raster object.
		@param other: Raster object.
		@param nan: str or bool. Handling NaNs. 
				If nan = 'cutoff', set lower cutoff of 1e-30.
				If nan = 'changeNodata', use changeNoData function.
		@param method: str. Method of interpolation. See resample_dict.
		@returns Raster object which is cropped and resampled, with Nodata values as NaN.
		"""
		safety = False
		bounds = other.bounds
		if (bounds[2]-bounds[0]) < self.resolution[0]*0.9 or (bounds[3] - bounds[1]) < self.resolution[1]*0.9:
			safety = True
			bounds = [bounds[0] - self.resolution[0]*1.1, bounds[1] + self.resolution[1]*1.1, bounds[2] + self.resolution[0]*1.1, bounds[3] - self.resolution[1]*1.1]

		r = self.crop(bounds)
		if nan != False:
			if nan == 'cutoff':
				r = r.cutOff(-1e33)
			elif nan == 'changeNodata':
				r = r.changeNodata(np.nan) 
		final = r.resample(other, method = method)

		if safety:
			final = final.crop(other)
		return final

	def max(self):
		"""Given a raster, compute its maximum value over non-NaN data.
		****************************************
		@param self: object.
		@returns float.
		@post: if STATISTICS_MAXIMUM not in self.raster.attrs, adds it.
		"""
		#self.lazyNaN()
		mx = np.nanmax(self.raster)
		self.raster.attrs['STATISTICS_MAXIMUM'] = mx
		return mx

	def maximum(self, other):
		"""Given a Raster, compute its maximum against another raster or a value
		****************************************
		@param self: object.
		@param other: int,float or object.
		@pre Assume the two rasters, if both are rasters are in the same datum, coordinate system.
		@returns Raster.
		"""
		#self.lazyNaN()
		if type(other) in (float,int):
			return Raster(np.maximum(self.raster, other))
		else:
			##other.lazyNaN()
			return Raster(np.maximum(self.raster, other.raster))

	def mean(self, na_rm = True):
		"""Get the mean of all pixels.
		****************************************
		@param self: object.
		@param na_rm: bool. If True, does not average over NaN values.
		@returns float.
		"""
		#self.lazyNaN()
		count = ~self.isna()
		if na_rm:
			return np.nansum(self.raster.values)/np.sum(count.raster.values)
		else:
			return np.sum(self.raster.values)/(self.shape[0]*self.shape[1])

	def merge(self, others, method = 'last'):
		if type(others) == Raster:
			others = [others]

		raster_list = [other.raster for other in others if other.verifyDims()]
		if self.verifyDims():
			raster_list = [self.raster]+raster_list		
		raster = merge_arrays(raster_list, method = method)
		return Raster(raster)

	def min(self):
		"""Given a raster, compute its minimum value.
		****************************************
		@param self: object.
		@returns float.
		@post: if STATISTICS_MINIMUM not in self.raster.attrs, adds it.
		"""
		#self.lazyNaN()
		mn = np.nanmin(self.raster)
		self.raster.attrs['STATISTICS_MINIMUM'] = mn
		return mn

	def minimum(self, other):
		"""Given a Raster, compute its minimum against another raster or a value
		****************************************
		@param self: object.
		@param other: int,float or object.
		@pre Assume the two rasters, if both are rasters are in the same datum, coordinate system.
		@returns Raster.
		"""
		#self.lazyNaN()
		if type(other) in (float,int):
			return Raster(np.minimum(self.raster, other))
		else:
			#other.lazyNaN()
			return Raster(np.minimum(self.raster, other.raster))

	def pixel(self,x,y, error = True):
		"""Given a Raster and coordinates, get the pixel indices.
		****************************************
		@param self: object.
		@param x: float or Numpy float array. x coordinate.
		@param y: float or Numpy float array. y coordinate.
		"""

		if type(x) in {np.ndarray, pd.Series}:
			j = np.int32((x - self.transform[2])/self.transform[0])
			i = np.int32((y - self.transform[5])/self.transform[4])

			if error:
				if (i >= self.shape[0]).any() or (i < 0).any():
					raise Exception('Some coordinates out of bounds.')
				if (j >= self.shape[1]).any() or (j < 0).any():
					raise Exception('Some coordinates out of bounds.')

			return i,j
			
		j = int((x - self.transform[2])/self.transform[0])
		i = int((y - self.transform[5])/self.transform[4])
		if error:
			if i >= self.shape[0] or i < 0:
				raise Exception('Coordinates (lon = {0},lat = {1}) out of bounds.'.format(x,y))
			if j >= self.shape[1] or j < 0:
				raise Exception('Coordinates (lon = {0},lat = {1}) out of bounds.'.format(x,y))
		return i,j

		

	def reversePixel(self,i,j):
		"""Given a Raster and indices, get the pixel coordinates.
		****************************************
		@param self: object.
		@param i: int. x coordinate.
		@param j: int. y coordinate.
		"""
		x = j*self.transform[0] + self.transform[2]
		y = i*self.transform[4] + self.transform[5]
		
		return x,y

	def plot(self, title = None, height = 6, width = 6, auto_resolution = True):
		"""Plot the underlying raster data, either single band or multiband.
		****************************************
		@param self: object.
		@param title: str. Title for the plot
		@param height, width: float. The pyplot dimensions.
		@param auto_resolution: bool. If True, adjusts resolution for easier plot.
		@returns None.
		@post creates matplotlib plot with raster data.
		"""

		r = self
		if auto_resolution:
			factor = min(int(max(self.shape[0],self.shape[1])/3000),1)
			r = self.aggregate(factor)


		fig = r.draw(title, height, width)
		fig.show()

	def projectRaster(self, to = None, crs = pyproj.CRS('EPSG:4326'), method = 'bilinear'):
		"""Project raster to different raster or to a given CRS.
		****************************************
		@param self: object.
		@param to: object or None. If None, reproject based on @a crs.
				   If object, reproject @a self to that raster's CRS and dimensions.
		@param crs: pyproj.CRS object. Indicates pyproj coordinate system, e.g. 'EPSG:4326'.
		@param method: str. Indicates method of resampling, e.g. bilinear, cubic.
		@returns object, with new reprojected raster.
		"""
		#self.lazyNaN()
		if to is None:
			new_raster = self.raster.rio.reproject(crs, resampling = resample_dict[method])
		else:
			new_raster = self.raster.rio.reproject_match(to.raster, resampling = resample_dict[method])
		return Raster(new_raster)



	def resample(self, other, method = 'bilinear'):
		"""Resample raster data based on a template raster.
		****************************************
		@param self: object.
		@param other: object. Used as template for resampling @a self to.
		@param method: str. Indicates method of resampling, e.g. bilinear, cubic.
		@returns object, with raster that is resampled.
		"""
		#self.lazyNaN()
		new_raster = self.raster.rio.reproject_match(other.raster, resampling = resample_dict[method])
		return Raster(new_raster)


	def sum(self, na_rm = True):
		"""Get the sum of all pixels.
		****************************************
		@param self: object.
		@param na_rm: bool. If True, does not sum over NaN values.
		@returns float.
		"""
		#self.lazyNaN()
		if na_rm:
			return np.nansum(self.raster.values)
		else:
			return np.sum(self.raster.values)

	def verifyDims(self):
		return len(self.raster.dims) == 2

	def writeNodata(self, nodata):
		"""Change the no data of the raster.
		****************************************
		@param self: object.
		@param nodata: float/bool/int. Writh the nodata of the value to the underlying raster.
		@returns None
		@post changes self.raster.rio.nodata and self.nodata.
		"""
		self.raster.rio.write_nodata(nodata, inplace = True)
		self.nodata = nodata

	def writeRaster(self, filename, driver = 'GTiff', dtype = 'float32', compress = 'LZW'):
		"""Write raster to disk.
		****************************************
		@param self: object.
		@param filename: str. Location to output floods too.
		@param driver: str. Indicates driver used for writing rasters.
		@param dtype: str. Indicates dtype of raster.
		@param compress: str. Either LZW or zlib.
		"""

		if compress == 'LZW':
			self.raster.rio.to_raster(raster_path = filename,
									  driver = driver,
									  compress = 'LZW',
									  dtype = dtype)
		elif compress == 'zlib':
			self.raster.rio.to_raster(raster_path = filename,
									  driver = driver,
									  compress = 'zlib',
									  zlevel = 9,
									  dtype = dtype)