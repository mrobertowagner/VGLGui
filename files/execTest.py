#!/usr/bin/env python3

# OPENCL LIBRARY
from skimage import io
from vgl_lib.vglConst import VGL_BLANK_CONTEXT, VGL_CL_CONTEXT
from numpy.lib.shape_base import get_array_wrap

from vgl_lib.vglImage import VglImage, create_vglShape, vglLoadImage
from vgl_lib.struct_sizes import struct_sizes
from vgl_lib import vglClImage
from PIL import Image

import pyopencl as cl
import os
os.environ['PYOPENCL_NO_CACHE'] = '1'

import cv2
#cd ~/Documentos/InterpretadorWorkflow/VGLGui/; ./testeamostra.py ./images/teste.jpeg img_f/
# VGL LIBRARYS
import vgl_lib as vl

# TO WORK WITH MAIN
import numpy as np

# IMPORTING METHODS
from cl2py_shaders import * 

import time as t
import sys
from matplotlib import pyplot

import matplotlib.pyplot as mp

from matplotlib.image import imread

from matplotlib import image

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

def rgb2gray(im):
    result = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    result[result > 255] = 255
    np.round(result)
    return np.uint8(result) 

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


img_in_path1 = 'images/demo1/img2_Convolution.png'
img_in_path2 = 'images/demo1/img2_BlackHat.png'
img_out_path= 'images/demo1/'

msg = ""



# [0, 1, 0, 1, 1, 1, 0, 1, 0]
#[0.125,  0.500, 0.125, 0.500, 1.000, 0.500, 0.125,  0.500, 0.125]
#[0,0,0.125,0,0, 0,0,0.500,0,0,.125,0.500,1.000,0.500,.125,0,0,0.500,0,0,0,0,0.125,0,0]
#[0,0,0.125,0,0, 0,0,0.500,0,0,.125,0.500,1.000,0.500,.125,0,0,0.500,0,0,0,0,0.125,0,0]


#[0.0030, 0.0133, 0.0219, 0.0133, 0.0030,0.0133, 0.0596, 0.0983, 0.0596, 0.0133,0.0219, 0.0983, 0.1621, 0.0983, 0.0219,0.0133, 0.0596, 0.0983,0.0596, 0.0133,0.0030, 0.0133, 0.0219, 0.0133, 0.0030]

vl.vglClInit()



img_input1 = vl.VglImage(img_in_path1, None, vl.VGL_IMAGE_2D_IMAGE())
vl.vglLoadImage(img_input1)
if( img_input1.getVglShape().getNChannels() == 3 ):
    vl.rgb_to_rgba(img_input1)

vl.vglClUpload(img_input1)


img_input2 = vl.VglImage(img_in_path2, None, vl.VGL_IMAGE_2D_IMAGE())
vl.vglLoadImage(img_input2)
if( img_input2.getVglShape().getNChannels() == 3 ):
    vl.rgb_to_rgba(img_input2)

vl.vglClUpload(img_input2)


img_output = vl.create_blank_image_as(img_input1)
img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input1) )
vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())

imshow(VglImage.get_ipl(img_input1))
imshow(VglImage.get_ipl(img_input2))

vglClSub(img_input2,img_input1,img_output)

salvando2d(img_output, img_out_path+"img-Sub1.png")
imshow(VglImage.get_ipl(img_output))



'''
filename = 'images/demo/'
  
image_path = 'images/demo'


# Getting the kernel to be used in Top-Hat
filterSize =(3, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                   filterSize)
  
print(kernel)
# Reading the image named 'input.jpg'
input_image = cv2.imread("images/img21.png")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
  
# Applying the Top-Hat operation
tophat_img = cv2.morphologyEx(input_image, 
                              cv2.MORPH_TOPHAT,
                                kernel)
    
cv2.imshow("original", input_image)
cv2.imshow("tophat", tophat_img)
status = cv2.imwrite('/home/Documentos/teste11.png',tophat_img)
#print(status)
cv2.waitKey(5000)

input_image = cv2.imread("images/demo/imgf.png")
ret, thresh1 = cv2.threshold(input_image, 3, 255, cv2.THRESH_BINARY) 

cv2.imshow('Binary Threshold', thresh1)
cv2.waitKey(5000)

kernel = np.ones((3,3), np.uint8) 
  
img_erosion = cv2.erode(thresh1, kernel, iterations=1) 
img_dilation = cv2.dilate(thresh1, kernel, iterations=1) 
  
#cv2.imshow('Input', img) 
cv2.imshow('Erosion', img_erosion) 
cv2.waitKey(5000)
cv2.imshow('Dilation', img_dilation) 
cv2.waitKey(5000)
'''