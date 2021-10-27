#!/usr/bin/env python3

# OPENCL LIBRARY
from numpy.lib.shape_base import get_array_wrap

from vgl_lib import vglImage
from vgl_lib.vglImage import VglImage, vglLoadImage
from vgl_lib.struct_sizes import struct_sizes
from vgl_lib import vglClImage
from PIL import Image
from cl2py_MM import *
import pyopencl as cl



# VGL LIBRARYS
import vgl_lib as vl

# TO WORK WITH MAIN
import numpy as np

# IMPORTING METHODS
from cl2py_shaders import * 

import time as t
import sys
from matplotlib import pyplot


"""
	THIS BENCHMARK TOOL EXPECTS 2 ARGUMENTS:
	ARGV[1]: PRIMARY 2D-IMAGE PATH (COLORED OR GRAYSCALE)
		IT WILL BE USED IN ALL KERNELS
	ARGV[2]: SECONDARY 2D-IMAGE PATH (COLORED OR GRAYSCALE)
		IT WILL BE USED IN THE KERNELS THAT NEED
		TWO INPUT IMAGES TO WORK PROPERLY
	THE RESULT IMAGES WILL BE SAVED AS IMG-[PROCESSNAME].JPG
"""
import matplotlib.pyplot as mp

from matplotlib.image import imread

from matplotlib import image


def imshow(im):
    plot = mp.imshow(im, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
    plot.set_interpolation('nearest')
    mp.show()

def salvando2d(img, name):
	# SAVING IMAGE img
	ext = name.split(".")
	ext.reverse()

	#vl.vglClDownload(img)
	vl.vglCheckContext(img, vl.VGL_RAM_CONTEXT())

	if( ext.pop(0).lower() == 'jpg' ):
		if( img.getVglShape().getNChannels() == 4 ):
			vl.rgba_to_rgb(img)
	
	vl.vglSaveImage(name, img)


img_in_path = "images/driveTresh.png"
img_in_path1 = "images/01_test.png"
img_out_path= "images/"

msg = ""

vl.vglClInit()

img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
vl.vglLoadImage(img_input)
if( img_input.getVglShape().getNChannels() == 3 ):
    vl.rgb_to_rgba(img_input)

vl.vglClUpload(img_input)


img_input1 = vl.VglImage(img_in_path1, None, vl.VGL_IMAGE_2D_IMAGE())
vl.vglLoadImage(img_input1)
if( img_input1.getVglShape().getNChannels() == 3 ):
    vl.rgb_to_rgba(img_input1)

vl.vglClUpload(img_input1)

img_output = vl.create_blank_image_as(img_input)
img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())

e = ""

convolution_window_2d_5x5 = np.array((	(1, 1,  1,  1,  1),
											(1, 1,  1,  1,  1),
											(1, 1,  1,  1,  1),
											(1, 1,  1,  1,  1),
											(1, 1,  1,  1,  1) ), np.float32)

cv = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]    
nsteps = 1000
media = 0.0
total = 0.0
p = 0
inicio = t.time()

from datetime import datetime

n = 1000

t0 = datetime.now()

for i in range( n ):
  vglClConvolution(img_input, img_output,convolution_window_2d_5x5, np.uint32(5), np.uint32(5))

t1 = datetime.now()

diff = t1 - t0

med = (diff.total_seconds() * 1000) / n

print("Tempo de" +str(n)+ "execuções do metódo Convolution:" + str(med) + " ms")



#vglClConvolution(img_input, img_output, cv, np.uint32(3), np.uint32(3))
#salvando2d(img_output, img_out_path+"img-vglClConvolution.jpg")

#vglClErode(img_input,img_output, cv, np.uint32(3), np.uint32(3))
#salvando2d(img_output, img_out_path+"img-vglClErodeCross1.png")

#vglClThreshold(img_input,img_output,np.float32(0.5))
#salvando2d(img_output, img_out_path+"img-vglClTreshold.png")
