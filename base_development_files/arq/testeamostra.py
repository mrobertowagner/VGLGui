#!/usr/bin/env python3

# OPENCL LIBRARY
from numpy.lib.shape_base import get_array_wrap
from vgl_lib import vglImage
from vgl_lib.vglImage import VglImage, vglLoadImage
from vgl_lib.struct_sizes import struct_sizes
from vgl_lib import vglClImage
from PIL import Image

import pyopencl as cl


#cd ~/Documentos/InterpretadorWorkflow/VGLGui/; ./testeamostra.py ./images/img2.png img_f/
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


img_in_path = sys.argv[1]
nSteps		= 100
img_out_path= sys.argv[2]

msg = ""

vl.vglClInit()

img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
vl.vglLoadImage(img_input)
if( img_input.getVglShape().getNChannels() == 3 ):
    vl.rgb_to_rgba(img_input)

vl.vglClUpload(img_input)

img_output = vl.create_blank_image_as(img_input)
img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )

img_output1 = vl.create_blank_image_as(img_input)
img_output1.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )


img_output2 = vl.create_blank_image_as(img_input)
img_output2.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )

img_output3 = vl.create_blank_image_as(img_input)
img_output3.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )

#vglClThreshold(img_input,img_output,np.float32(0.8))
#salvando2d(img_output, img_out_path+"img-vglClThresh.png")

#vglClInvert(img_input,img_output1)
#salvando2d(img_output1, img_out_path+"img-vglClInvert.png")

#vglClMin(img_input,img_output1,img_output3)
#salvando2d(img_output3, img_out_path+"img-vglClMin.png")

#vglClSum(img_output,img_output1,img_output2)
#salvando2d(img_output3, img_out_path+"img-vglClS.png")