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
# Rule7: Glyphs have READY (ready to run) and DONE (executed) status, both status start being FALSE
fileRead(lstGlyph, lstConnection)

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

#Update the status of glyph entries
for vConnection in lstConnection:

    #vConnection.output_glyph_id         #glyph identifier code output
    #vConnection.output_varname          #variable name output
    #vConnection.input_glyph_id          #glyph identifier code input
    #vConnection.input_varname           #variable name input
    #vConnection.image                   #image
    #vConnection.ready                   #False = unread or unexecuted image; True = image read or executed 

    #Identifies connection input and output glyph
    Index               = 0
    vGlyph_Index        = 0        #Glyph index to run
    vConnectionIndex    = lstConnection.index

    # Rule6: Edges with images already read or generated have status READY=TRUE (image ready to be processed)
    # If the connection image is not ready for processing, it will go to the next
    try:
        if not lstConnection[vConnectionIndex].getReadyConnection:
            raise Error("Invalid connection: ",{vConnection.output_glyph_id})
    except ValueError:
        print("Image not ready for processing." , {vConnection.output_glyph_id})

    # Search for the glyph for execution.
    # If it is the first glyph, consider its output, otherwise consider its input.

    for vGlyph in lstGlyph:

        if vConnection.output_glyph_id == vGlyph.glyph_id and vConnection.output_glyph_id == 1:
            vGlyph_Index = Index
        elif vConnection.input_glyph_id == vGlyph.glyph_id:
            vGlyph_Index = Index

        Index = +1

    # Only run the glyph if all its entries are
    if vGlyph.getGlyphReady():
        
        if vGlyph.func == 'vglLoadImage':

            # vglLoadImage(img, filename="")

            # Read "-filename" entry from glyph vglLoadImage
            img_in_path = vGlyph.lst_par[0].getValue()               
            nSteps		= 1

            img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())

            vl.vglLoadImage(img_input)
            if( img_input.getVglShape().getNChannels() == 3 ):
                vl.rgb_to_rgba(img_input)

            vl.vglClUpload(img_input)

            # Rule2: Source glyph only has one output (image), it's a parameter
            lstConnection[vConnectionIndex].setImageConnection = img_input

            # Rule6: Edges with images already read or generated have status READY=TRUE (image ready to be processed)
            #        Assign read-ready to connection
            lstConnection[vConnectionIndex].setReadyConnection
                                  
            # Rule10: Glyph becomes DONE = TRUE after its execution
            #         Assign done to glyph
            lstGlyph[vGlyph_Index].setGlyphDone(True)

            # Rule11: The outputs of a Glyph become DONE = TRUE after the execution of the Glyph
            #         Assign done to glyph
            lstGlyph[vGlyph_Index].setReadyGlyphOutputAll(False)

        elif vGlyph.func == 'vglCreateImage':

            # create_blank_image_as(img):

            # Read "-filename" entry from glyph vglLoadImage
            # img_input = uploadFile (vGlyph.lst_par[0].getValue())

            # Create output image
            img_output = vl.create_blank_image_as(img_input)
            img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
            
            # Save new image
            vl.vglSaveImage(vGlyph.lst_par[1].getValue(), img_output)
            vl.rgb_to_rgba(img_output)

            msg = msg + "Create function applied"
        
        elif vGlyph.func == 'vglClBlurSq3': #Function blur

            # vglClBlurSq3(img_input, img_output)

            # Read "-filename" entry from glyph vglLoadImage
            # img_input = uploadFile (vGlyph.lst_par[0].getValue())

            # Create output image
            img_output = vl.create_blank_image_as(img_input)
            img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
            
            # Apply BlurSq3 function
            #vglClBlurSq3(img_input, img_output)


            # Save new image
            #salvando2d(img_output, vGlyph.lst_par[1].getValue())
            
            vl.rgb_to_rgba(img_output)
            
            #vl.vglSaveImage(vGlyph.lst_par[1].getValue(), img_output)
            #vl.rgb_to_rgba(img_output)

            msg = msg + "Blur function applied"

        elif vGlyph.func == 'vglClThreshold': #Function Threshold

            # vglClThreshold(src, dst, thresh, top = 1.0)

            # Read "-filename" entry from glyph vglLoadImage
            #img_input = uploadFile (vGlyph.lst_par[0].getValue())

            # Create output image
            img_output = vl.create_blank_image_as(img_input)
            img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
            
            # Apply Threshold function
            vglClThreshold(img_input, img_output, np.float32(0.5))
                        
            # Save new image
            #salvando2d(img_output, vGlyph.lst_par[1].getValue())
            #vl.rgb_to_rgba(img_output)
            
            #vl.vglSaveImage(vGlyph.lst_par[1].getValue(), img_output)
            #vl.rgb_to_rgba(img_output)

            msg = msg + "Thershold function applied"

        elif vGlyph.func == 'ShowImage':
    
            # Rule 3: Sink glyph only has one entry (image)             
            img = Image.open(vGlyph.lst_par[1].getValue())
            img.show()
            
        elif vGlyph.func == 'vglSaveImage':

            # SAVING IMAGE img
            ext = vGlyph.lst_par[0].getValue().split(".")
            ext.reverse()

            img = vGlyph.lst_par[1].getValue()

            vl.vglClDownload(vGlyph.lst_par[1].getValue())

            if( ext.pop(0).lower() == 'jpg' ):
                if( img.getVglShape().getNChannels() == 4 ):
                    vl.rgba_to_rgb(img)

            # Rule 3: Sink glyph only has one entry (image)             
            vl.vglSaveImage(ext, img)
        
# Shows the content of the Glyphs
print("-------------------------------------------------------------")
print(msg)
print("-------------------------------------------------------------")
img_input = None
img_output = None
