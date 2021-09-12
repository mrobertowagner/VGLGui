#!/usr/bin/env python3

# OPENCL LIBRARY
from vgl_lib.vglConst import VGL_CL_CONTEXT
from numpy.lib.shape_base import get_array_wrap
from vgl_lib import vglImage, vglShape
from vgl_lib.vglImage import VglImage, create_vglShape, vglLoadImage
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

def rgb2gray(im):
    result = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    result[result > 255] = 255
    np.round(result)
    return np.uint8(result)

def rgb_to_gray(img):
        grayImage = np.zeros(img.shape)
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])

        R = (R *.299)
        G = (G *.587)
        B = (B *.114)

        Avg = (R+G+B)
        grayImage = img.copy()

        for i in range(3):
           grayImage[:,:,i] = Avg
           
        return grayImage       

imggray = rgb_to_gray(VglImage.get_ipl(img_input))

#imshow(imggray)

#salvando2d(gray, img_out_path+"img-vglClTrs.tif")
from skimage import io
#vl.vglCheckContext(f,vl.VGL_RAM_CONTEXT()) 
#vglLoadImage(imggray)
#print(imggray)
#vl.vglCheckContext(imggray, vl.VGL_CL_CONTEXT())
#f = VglImage.get_ipl(imggray)
imggray1 = io.imread(imggray)
#vl.vglAddContext(fa, vl.VGL_RAM_CONTEXT())	
#vglClThreshold(imggray, img_output,np.float32(0.5))
#salvando2d(imggray,img_out_path+"img-vglClTrs.tif")

#vglClInvert(img_input,img_output1)
#salvando2d(img_output1, img_out_path+"img-vglClInvert.png")

#vglClMin(img_input,img_output1,img_output3)
#salvando2d(img_output3, img_out_path+"img-vglClMin.png")

#vglClSum(img_output,img_output1,img_output2)
#salvando2d(img_output3, img_out_path+"img-vglClS.png")


