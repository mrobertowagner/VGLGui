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

#Show info
'''def procShowInfo():
    for vGlyph in lstGlyph:
        print("Library:", vGlyph.library, "Function:", vGlyph.func, "Localhost:", vGlyph.localhost, "Glyph_Id:", vGlyph.glyph_id, 
            "Position_Line:", vGlyph.glyph_x, "Position_Column:", vGlyph.glyph_y)#, "Parameters:", vGlyph.lst_par)

        #Shows the list of glyph inputs
        for vGlyphIn in vGlyph.lst_input:
            print("Glyph_Id:", vGlyph.glyph_id, "Glyph_In:", vGlyphIn)

        #Shows the list of glyph outputs
        for vGlyphOut in vGlyph.lst_output:
            print("Glyph_Id:", vGlyph.glyph_id, "Glyph_Out:", vGlyphOut)

    # Shows the content of the Connections
    for vConnection in lstConnection:
        print("Conexão:", vConnection.type, "Glyph_Output_Id:", vConnection.output_glyph_id, "Glyph_Output_Varname:", vConnection.output_varname,
            "Glyph_Input_Id:", vConnection.input_glyph_id, "Glyph_Input_Varname:", vConnection.input_varname)
'''
# Program execution

# Reading the workflow file and loads into memory all glyphs and connections
fileRead(lstGlyph)

def salvando2d(img, name):
	# SAVING IMAGE img
	ext = name.split(".")
	ext.reverse()

	vl.vglClDownload(img)

	if( ext.pop(0).lower() == 'jpg' ):
		if( img.getVglShape().getNChannels() == 4 ):
			vl.rgba_to_rgb(img)
	
	vl.vglSaveImage(name, img)

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

#Update the status of glyph entries
for vGlyph in lstGlyph:

    if vGlyph.func == 'vglLoadImage':

        # Read "-filename" entry from glyph vglLoadImage
        img_input = uploadFile (vGlyph.lst_par[0].getValue())

    elif vGlyph.func == 'vglClBlurSq3': #Function blur

        # Read "-filename" entry from glyph vglLoadImage
        img_input = uploadFile (vGlyph.lst_par[0].getValue())

        # Create output image
        img_output = vl.create_blank_image_as(img_input)
        img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
        
        # Apply BlurSq3 function
        vglClBlurSq3(img_input, img_output)

        # Save new image
        vl.vglSaveImage(vGlyph.lst_par[1].getValue(), img_output)
        #vl.rgb_to_rgba(img_output)

        msg = msg + "Blur function applied"

    elif vGlyph.func == 'vglClCopy_TIRAR': #Function copy
        sys.argv[1] = 'tmp/testes/img-vglClBlurSq3.jpg'
        img_in_path = sys.argv[1]
        nSteps		= int(sys.argv[2])
        img_out_path= sys.argv[3]
        img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
        vl.vglLoadImage(img_input)
        if( img_input.getVglShape().getNChannels() == 3 ):
            vl.rgb_to_rgba(img_input)
        vl.vglClUpload(img_input)

        img_output = vl.create_blank_image_as(img_input)
        img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
        vglClCopy(img_input, img_output)
        media = 0.0
        for i in range(0, 5):
            p = 0
            inicio = t.time()
            while(p < nSteps):
                vglClCopy(img_input, img_output)
                p = p + 1
                fim = t.time()
                media = media + (fim-inicio)

        salvando2d(img_output, img_out_path+"img-vglClCopy.jpg")
        vl.rgb_to_rgba(img_output)
        msg = msg + "Tempo de execução do método vglClCopy:\t\t\t" +str( round( (media / 5), 9 ) ) +"s\n"

    elif vGlyph.func == 'vglClThreshold_TIRAR': #Function Threshold
        sys.argv[1] = 'tmp/testes/img-vglClBlurSq3.jpg'
        img_in_path = sys.argv[1]
        nSteps		= int(sys.argv[2])
        img_out_path= sys.argv[3]
        img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
        vl.vglLoadImage(img_input)
        if( img_input.getVglShape().getNChannels() == 3 ):
            vl.rgb_to_rgba(img_input)
        vl.vglClUpload(img_input)

        img_output = vl.create_blank_image_as(img_input)
        img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
        vglClThreshold(img_input, img_output, np.float32(0.5))
        media = 0.0
        for i in range(0, 5):
            p = 0
            inicio = t.time()
            while(p < nSteps):
                vglClThreshold(img_input, img_output, np.float32(0.5))
                p = p + 1
                fim = t.time()
                media = media + (fim-inicio)
                    
        salvando2d(img_output, img_out_path+"img-vglClThreshold.jpg")
        vl.rgb_to_rgba(img_output)
        msg = msg + "Tempo de execução do método vglClThreshold:\t\t" +str( round( (media / 5), 9 ) ) +"s\n"

    elif vGlyph.func == 'ShowImage':
 
        #Show image
        img = Image.open(vGlyph.lst_par[0].getValue())
        img.show()
        
    elif vGlyph.func == 'vglSaveImage':

        copyFile (vGlyph.lst_par[0].getValue(), vGlyph.lst_par[1].getValue())

    #Glyph execute
            
# Shows the content of the Glyphs
#procShowInfo()
print("-------------------------------------------------------------")
print(msg)
print("-------------------------------------------------------------")
img_input = None
img_output = None
