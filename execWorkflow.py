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

from readWorkflow import *


import time as t

    #inicio das estruturas vgl

# Program execution

# Reading the workflow file and loads into memory all glyphs and connections
fileRead(lstGlyph, lstConnection)

def uploadFile (filename):
    
    # Read "-filename" entry from glyph vglLoadImage
    img_in_path = filename               
    nSteps		= 1

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

vl.vglClInit() 

vGlyph_FuncExec = ''                    #Function name to execution
vGlyph_IndexProx = 0                    #Index next glyph to run
vGlyph_Id = 0                           #Temporary identification next program block
vGlyph_IndexExec = 0                 #Glyph index to run

#Update the status of glyph entries
for vConnection in lstConnection:

    #vConnection.output_glyph_id         #glyph identifier code output
    #vConnection.output_varname          #variable name output
    #vConnection.input_glyph_id          #glyph identifier code input
    #vConnection.input_varname           #variable name input
    #vConnection.image                   #image
    #vConnection.ready                   #False = unread or unexecuted image; True = image read or executed 

    #Identifies connection input and output glyph
    Index = 0

    for vGlyph in lstGlyph:

        if vGlyph.glyph_id == vConnection.output_glyph_id:
            vGlyph_IndexOut = Index
        
        if vGlyph.glyph_id == vConnection.input_glyph_id:
            vGlyph_IndexIn = Index

        Index += 1

    #Execution of the first glyph of program block
    if vGlyph_IndexExec == 0 or vGlyph_Id == 1:
        vGlyph_IndexExec = vGlyph_IndexOut
    else:
        vGlyph_IndexExec = vGlyph_IndexIn

    #Get information from the glyph to execute
    vGlyph = lstGlyph[vGlyph_IndexExec]
    vGlyph_Id = vGlyph.glyph_id                     #Temporary identification next program block

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
        img_output = vl.create_blank_image_as(img_input)
        img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
        
        # Apply BlurSq3 function
        #vglClBlurSq3(img_input, img_output)

        # Save new image
        vl.vglSaveImage(vGlyph.lst_par[1].getValue(), img_output)
        vl.rgb_to_rgba(img_output)

        msg = msg + "Blur function applied"

    elif vGlyph.func == 'vglClThreshold': #Function Threshold

        # Read "-filename" entry from glyph vglLoadImage
        img_input = uploadFile (vGlyph.lst_par[0].getValue())

        # Create output image
        img_output = vl.create_blank_image_as(img_input)
        img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
        
        # Apply Threshold function
        #vglClThreshold(img_input, img_output, np.float32(0.5))
                    
        # Save new image
        vl.vglSaveImage(vGlyph.lst_par[1].getValue(), img_output)
        vl.rgb_to_rgba(img_output)

        msg = msg + "Thershold function applied"

    elif vGlyph.func == 'ShowImage':
 
        img = Image.open(vGlyph.lst_par[1].getValue())
        img.show()
        
    elif vGlyph.func == 'vglSaveImage':

        copyFile (vGlyph.lst_par[0].getValue(), vGlyph.lst_par[1].getValue())
         
# Shows the content of the Glyphs
print("-------------------------------------------------------------")
print(msg)
print("-------------------------------------------------------------")
img_input = None
img_output = None
