#!/usr/bin/env python3
# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

# TO WORK WITH MAIN
import numpy as np

# IMPORTING METHODS
from cl2py_shaders import * 

#IMPORTING METHODS TO DISPLAY IMAGE
from PIL import Image

# IMPORTING METHODS FROM VGLGui
import sys
import os

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

sys.path.append(os.getcwd())

from readTest import *


import time as t

    #inicio das estruturas vgl

# Program execution
import matplotlib.pyplot as mp

def imshow(im):
    plot = mp.imshow(im, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
    plot.set_interpolation('nearest')
    mp.show()

# Reading the workflow file and loads into memory all glyphs and connections
fileRead(lstGlyph)


convolution_window_2d_3x3 = np.array((	(1/16, 2/16, 1/16),
                                        (2/16, 4/16, 2/16),
                                        (1/16, 2/16, 1/16) ), np.float32) 



#cv = np.array((1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9),np.float32) # Filtro média
#cv = np.array((1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3),np.float32)  # Filtro média
cv = np.array((	(1, 1, 1),(1, 1, 1),(1, 1, 1) ), np.float32) 


#cv = np.ndarray(shape=(1,1),dtype=float)                                         
print(cv)
#cv = np.array([1, 1, 1]).astype(float)
#print(cv)

#newcv = cv.reshape(3,3).astype(float)
#print(newcv)

cw  = np.array(((1, 1, 1),
                                        (1, 1, 1),
                                        (1, 1, 1) ), np.float32)
                                
convolution_window_2d_5x5 = np.array((	(1/256, 4/256,  6/256,  4/256,  1/256),
                                        (4/256, 16/256, 24/256, 16/256, 4/256),
                                        (6/256, 24/256, 36/256, 24/256, 6/256),
                                        (4/256, 16/256, 24/256, 16/256, 4/256),
                                        (1/256, 4/256,  6/256,  4/256,  1/256) ), np.float32)

def uploadFile (filename):  
    
    # Read "-filename" entry from glyph vglLoadImage
    img_in_path = filename               
    #nSteps		= 6

    img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())

    vl.vglLoadImage(img_input)
    if( img_input.getVglShape().getNChannels() == 3 ):
        vl.rgb_to_rgba(img_input)

    vl.vglClUpload(img_input)
    return img_input

def copyFile (filename_input, filename_output):

     # Upload input image
     img_in_path = filename_input               
     img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
     vl.vglLoadImage(img_input)

     ext = img_in_path.split(".")
     ext.reverse()
     if (ext.pop(0).lower() == 'jpg'):
         if( img_input.getVglShape().getNChannels() == 4 ):
             vl.rgba_to_rgb(img_input)

     vl.vglClUpload(img_input)

     # Save "-filename" output from Glyph vglSaveImage
     img_output = vl.create_blank_image_as(img_input)
     img_output.set_oclPtr (vl.get_similar_oclPtr_object(img_input))
     
     vl.vglSaveImage(filename_output, img_output)
     vl.rgb_to_rgba(img_output)

msg = ""
def salvando2d(img, name):
	# SAVING IMAGE img
	ext = name.split(".")
	ext.reverse()

	vl.vglClDownload(img)

	if( ext.pop(0).lower() == 'jpg' ):
		if( img.getVglShape().getNChannels() == 4 ):
			vl.rgba_to_rgb(img)
	
	vl.vglSaveImage(name, img)
vl.vglClInit() 

#Update the status of glyph entries
for vGlyph in lstGlyph:

    if vGlyph.func == 'vglLoadImage':

        # Read "-filename" entry from glyph vglLoadImage
        img_input = uploadFile (vGlyph.lst_par[0].getValue())

    elif vGlyph.func == 'vglCreateImage':
        
        # Read "-filename" entry from glyph vglLoadImage
        img_input = uploadFile (vGlyph.lst_par[0].getValue())

        # Create output image
        img_output = vl.create_blank_image_as(img_input)
        img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
        
        # Save new image
        vl.vglSaveImage(vGlyph.lst_par[1].getValue(), img_output)
        vl.rgb_to_rgba(img_output)

        msg = msg + "Create function applied"
    
    elif vGlyph.func == 'vglClBlurSq3': #Function blur

        # Read "-filename" entry from glyph vglLoadImage
        img_input = uploadFile (vGlyph.lst_par[0].getValue())

        # Create output image

        nSteps = 30
        # Apply BlurSq3 function
        vglClBlurSq3(img_input, img_output)
        media = 0.0
        for i in range(0, 5):
            nSteps = 8
            p = 0
            inicio = t.time()
            while(p < nSteps):
                vglClBlurSq3(img_output, img_output)
                p = p + 1
                fim = t.time()
                media = media + (fim-inicio)
        
        # Save new image
        salvando2d(img_output, vGlyph.lst_par[1].getValue())
        vl.rgb_to_rgba(img_output)
        #vl.vglSaveImage(vGlyph.lst_par[1].getValue(), img_output)
        #vl.rgb_to_rgba(img_output)

        msg = msg + "Blur function applied"

    elif vGlyph.func == 'vglClThreshold': #Function Threshold

        # Read "-filename" entry from glyph vglLoadImage
        img_input = uploadFile (vGlyph.lst_par[0].getValue())

        # Create output image
        
        # Apply Threshold function
        vglClThreshold(img_input, img_output, np.float32(0.5))
                    
        # Save new image
        salvando2d(img_output, vGlyph.lst_par[1].getValue())
        vl.rgb_to_rgba(img_output)
        #vl.vglSaveImage(vGlyph.lst_par[1].getValue(), img_output)
        #vl.rgb_to_rgba(img_output)

        msg = msg + "Thershold function applied"
    
    elif vGlyph.func == 'vglClInvert':
        img_input = uploadFile(vGlyph.lst_par[0].getValue())

        vglClInvert(img_input,img_output)

        salvando2d(img_output,vGlyph.lst_par[1].getValue())
        vl.rgb_to_rgba(img_output)

        msg = msg + "Invert functioun applied"

    elif vGlyph.func == 'vglClConvolution':
        img_input = uploadFile(vGlyph.lst_par[0].getValue())


        vglClConvolution(img_input,img_output, cw, np.uint32(5), np.uint32(5))

        salvando2d(img_output,vGlyph.lst_par[1].getValue())
        vl.rgb_to_rgba(img_output)

        msg = msg + "Convolution function applied"


    elif vGlyph.func == 'vglClErode':
        img_input = uploadFile(vGlyph.lst_par[0].getValue())



        vglClErode(img_input,img_output, cv, np.uint32(5), np.uint32(5))

        salvando2d(img_output,vGlyph.lst_par[1].getValue())
        vl.rgb_to_rgba(img_output)

        msg = msg + "Erode function applied"

    elif vGlyph.func == 'vglClDilate':
        img_input = uploadFile(vGlyph.lst_par[0].getValue())



        vglClDilate(img_input,img_output, cv, np.uint32(5), np.uint32(5))

        salvando2d(img_output,vGlyph.lst_par[1].getValue())
        vl.rgb_to_rgba(img_output)

        msg = msg + "Dilate function applied"

    elif vGlyph.func == 'vglSaveImage':

        copyFile (vGlyph.lst_par[0].getValue(), vGlyph.lst_par[1].getValue())
         
# Shows the content of the Glyphs
print("-------------------------------------------------------------")
print(msg)
print("-------------------------------------------------------------")
img_input = None
img_output = None
