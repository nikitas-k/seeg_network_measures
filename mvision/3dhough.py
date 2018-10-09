#!/usr/bin/env python2

import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt



def 3dhough(file):

	def ind2sub(array_shape, ind):
		rows = (ind.astype('int') / array_shape[1])
		cols = (ind.astype('int') % array_shape[1])
		return rows, cols

	from itertools import product
	def accumarray(accmap, a, func=None, size=None, fill_value=0, dtype=None):
		#if accmap.shape[:a:ndim] != a.shape:
		    #raise ValueError("The initial dimensions of accmap must be the same as a.shape")
		if func is None:
		    func = np.sum
		if dtype is None:
		    dtype = a.dtype
		if accmap.shape == a.shape:
		    accmap = np.expand_dims(accmap, -1)
		adims = tuple(range(a.ndim))
		if size is None:
		    size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
		size = np.atleast_1d(size)
		
		vals = np.empty(size, dtype='O')
		for s in product(*[range(k) for k in size]):
		    vals[s] = []
		for s in product(*[range(k) for k in a.shape]):
		    indx = tuple(accmap[s])
		    val = a[s]
		    vals[indx].append(val)
		    
		out = np.empty(size, dtype=dtype)
		for s in product(*[range(k) for k in size]):
		    if vals[s] == []:
		        out[s] = fill_value
		    else:
		        out[s] = func(vals[s])
		        
		return out

 	## load data

	while True:
		try:
			f = nib.load(file)
			hdr = f.header
			size = hdr.get_zooms
			data = np.array(f.dataobj)
			if size.ndims > 3:
				return ValueError:
					print("Invalid number of dimensions. Must be a 3D nifti image.")

			if data.dtype is not 'uint8':
				data = np.uint8(data)
			
		except ValueError:
			print("Invalid file format. Accepted files are .nii")

	# main parameters
	minr = int(size[0])*2
	maxr = int(size[1])*3
	r_range = np.array([[0,0],[minr,maxr]])
	grdthres = 0.2
	fltrLM_R = 8
	multirad = 0.5
	obj_cint = 0.1

	vap_gradthres = 1
	vap_multirad = 3

	vap_obj_cint = 4

	grdx, grdy, grdz = np.gradient(data)
	grdmag = np.sqrt(grdx**2 + grdy**2 + grdz**2)

	grdthres = grdthres*np.max(grdmag[:])
	grdmasklin = np.array(np.where(grdmag > grdthres))
		    
	grdmask_IdxI, grdmask_IdxJ, grdmask_IdxK = grdmasklin[[0]], grdmasklin[[1]], grdmasklin[[2]]
	grdmask_IdxI = grdmask_IdxI.T
	grdmask_IdxJ = grdmask_IdxJ.T
	grdmask_IdxK = grdmask_IdxK.T

	## Compute the linear indices (as well as the subscripts of
	## all the votings to the accumulation array.
	## A row in matrix 'lin2accum_aJ' contains the J indices (into the
	## accumulation array) of all the votings that are introduced by a
	## same pixel in the image. Similarly with matrix 'lin2accum_aI'.

	rr_4linaccum = np.float64(r_range)
	linaccum_dr = np.array([[-1.5, 1.5]])

	lin2accum_aK = np.floor(np.dot(grdz[grdmask_IdxI, 
		            grdmask_IdxJ, grdmask_IdxK] / grdmag[grdmask_IdxI, 
		            grdmask_IdxJ, grdmask_IdxK], linaccum_dr) + 
		                    np.tile(np.float64(grdmask_IdxK) + 0.5, 
		                           [1, linaccum_dr.shape[1]]))

	lin2accum_aJ = np.floor(np.dot(grdx[grdmask_IdxI, 
		            grdmask_IdxJ, grdmask_IdxK] / grdmag[grdmask_IdxI, 
		            grdmask_IdxJ, grdmask_IdxK], linaccum_dr) +
		                    np.tile(np.float64(grdmask_IdxJ) + 0.5, 
		                             [1, linaccum_dr.shape[1]]))

	lin2accum_aI = np.floor(np.dot(grdy[grdmask_IdxI, 
		            grdmask_IdxJ, grdmask_IdxK] / grdmag[grdmask_IdxI, 
		            grdmask_IdxJ, grdmask_IdxK], linaccum_dr) +
		                    np.tile(np.float64(grdmask_IdxI) + 0.5,
		                            [1, linaccum_dr.shape[1]]))

	## Clip the votings that are out of the accumulation array

	mask_valid_aJaIaK_1 = np.logical_and(lin2accum_aK > 0, 
		                               lin2accum_aK < (np.size(grdmag,2) + 1))

	mask_valid_aJaIaK_2 = np.logical_and(mask_valid_aJaIaK_1, lin2accum_aJ > 0)

	mask_valid_aJaIaK_3 = np.logical_and(mask_valid_aJaIaK_2, lin2accum_aJ < (np.size(grdmag,2) + 1))

	mask_valid_aJaIaK_4 = np.logical_and(mask_valid_aJaIaK, lin2accum_aI > 0)

	mask_valid_aJaIaK = np.logical_and(mask_valid_aJaIaK_4, lin2accum_aI < (np.size(grdmag,1) + 1))

	mask_valid_aJaIaK_inv = ~mask_valid_aJaIaK
	lin2accum_aK = lin2accum_aK * mask_valid_aJaIaK + mask_valid_aJaIaK_inv
	lin2accum_aJ = lin2accum_aJ * mask_valid_aJaIaK + mask_valid_aJaIaK_inv
	lin2accum_aI = lin2accum_aI * mask_valid_aJaIaK + mask_valid_aJaIaK_inv

	## Linear indices of the votings into the accumulation array
	lin2accum = np.ravel_multi_index((lin2accum_aI, lin2accum_aJ, lin2accum_aK), grdmag.shape)
	lin2accum_size = lin2accum.shape
	lin2accum = np.reshape(lin2accum, [lin2accum.size, 1])


	weight4accum = np.tile(grdmag[grdmask_IdxI, grdmask_IdxJ, grdmask_IdxK], [lin2accum_size[1],1]) * mask_valid_aJaIaK.reshape(-1,1)
	accum = accumarray(lin2accum, weight4accum)

	accum = accum[np.newaxis].T
	z = np.zeros((grdmag.size - accum.size, 1))
	accum = np.vstack((accum, z))
	accum = np.reshape(accum, grdmag.shape)

	fltrLM_s = 1.35
	fltrLM_r = np.ceil(fltrLM_R * 0.6)
	fltrLM_npix = 1
	LM_LBra = 0.1
	LM_LB = np.max(accum) * LM_LBra

	s = ndimage.generate_binary_structure(3,1)
	lbl, nlbl = label(accum, return_num=1)
	f = ndimage.measurements.find_objects(lbl)
	sphcen = np.empty((1,3))
	for prop in regionprops(lbl, accum, cache=1):
		cc = [prop.centroid]
		if prop.area > 1:
			if prop.area < 5:
				if prop.max_intensity > obj_cint*np.max(data):
					sphcen = np.concatenate((sphcen, cc), 0)
	sphcen = np.delete(sphcen, (0), axis=0)
	sphcen = np.uint8(sphcen)

	return sphcen

