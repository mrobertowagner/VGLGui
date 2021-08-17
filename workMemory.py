#!/usr/bin/env python3

from vgl_lib.vglClImage import create_blank_image_as
import pyopencl as cl       # OPENCL LIBRARY
import vgl_lib as vl        # VGL LIBRARYS
import numpy as np          # TO WORK WITH MAIN
from cl2py_shaders import * # IMPORTING METHODS
from PIL import Image       # IMPORTING METHODS TO DISPLAY IMAGE
import os
import sys                  # IMPORTING METHODS FROM VGLGui
from readWorkflow import *


from matplotlib.image import imread

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
sys.path.append(os.getcwd())
import matplotlib.pyplot as mp

def imshow(im):
            plot = mp.imshow(im, cmap=mp.gray(), origin="upper", vmin=0, vmax=255)
            plot.set_interpolation('nearest')
            mp.show()


# Actions after glyph execution
def GlyphExecutedUpdate(vGlyph_Index, image):

    # Rule8: Glyphs have a list of entries. When all entries are READY=TRUE, the glyph changes status to READY=TRUE (function ready to run)
    #lstGlyph[vGlyph_Index].setGlyphReady(True)

    # Rule10: Glyph becomes DONE = TRUE after its execution. Assign done to glyph
    lstGlyph[vGlyph_Index].setGlyphDone(True)

    # Rule6: Edges whose source glyph has already been executed, and which therefore already had their image generated, have READY=TRUE (image ready to be processed).
    #        Reading the image from another glyph does not change this status. Check the list of connections
    for i_Con, vConnection in enumerate(lstConnection):

        # Checks if the executed glyph is the origin of any glyph
        if lstGlyph[vGlyph_Index].glyph_id == vConnection.output_glyph_id:

            # Rule2: In a source glyph, images (one or more) can only be output parameters.
            if image is not None:
                lstConnection[i_Con].setImageConnection(image)
            
            # Assign read-ready to connection
            lstConnection[i_Con].setReadyConnection

            # Finds the glyphs that originate from the executed glyph
            for i_Gli, vGlyph in enumerate(lstGlyph):

                if vGlyph.glyph_id == vConnection.input_glyph_id:

                    # Set READY = TRUE to the Glyph input
                    for vGlyphIn in lstGlyph[i_Gli].lst_input:
                        lstGlyph[i_Gli].setGlyphReadyInput(True, vConnection.input_varname)
                        lstGlyph[i_Gli].setGlyphReady(True)
                    break
                
# Program execution
def salvando2d(img, name):
	# SAVING IMAGE img
	ext = name.split(".")
	ext.reverse()

	vl.vglClDownload(img)

	if( ext.pop(0).lower() == 'jpg' ):
		if( img.getVglShape().getNChannels() == 4 ):
			vl.rgba_to_rgb(img)
	
	vl.vglSaveImage(name, img)
# Reading the workflow file and loads into memory all glyphs and connections
# Rule7: Glyphs have READY (ready to run) and DONE (executed) status, both status start being FALSE
fileRead(lstGlyph, lstConnection)

vl.vglClInit() 

# Update the status of glyph entries
for vGlyph_Index, vGlyph in enumerate(lstGlyph):

    # Rule9: Glyphs whose status is READY=TRUE (ready to run) are executed. Only run the glyph if all its entries are
    try:
        if not vGlyph.getGlyphReady():
            raise Error("Rule9: Glyph not ready for processing.", {vGlyph.glyph_id})
    except ValueError:
        print("Rule9: Glyph not ready for processing: ", {vGlyph.glyph_id})


    if vGlyph.func == 'vglLoadImage':

        # Read "-filename" entry from glyph vglLoadImage
        img_in_path = vGlyph.lst_par[0].getValue()               
        img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())

        vl.vglLoadImage(img_input)
        if( img_input.getVglShape().getNChannels() == 3 ):
            vl.rgb_to_rgba(img_input)

        vl.vglClUpload(img_input)

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph_Index, img_input)
                                
    elif vGlyph.func == 'vglCreateImage':

        # Search the input image by connecting to the source glyph
        for i_Con, vConnection in enumerate(lstConnection):
            if vGlyph.glyph_id == vConnection.input_glyph_id and vConnection.image is not None:
                img = vConnection.image

        RETVAL = vl.create_blank_image_as(img)
        #RETVAL.set_oclPtr( vl.get_similar_oclPtr_object(img) )

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph_Index, RETVAL)

    elif vGlyph.func == 'vglClBlurSq3': #Function blur

        # Search the input image by connecting to the source glyph
        for i_Con, vConnection in enumerate(lstConnection):
            if vGlyph.glyph_id == vConnection.input_glyph_id and vConnection.image is not None:
                img_input = vConnection.image

            img_output = RETVAL
        
            # Apply BlurSq3 function
            vglClBlurSq3(img_input,img_output)
            img_out_path = 'imgtst/'
            salvando2d(img_output,img_out_path+"imgTst.png")

            
            

            #imshow(img_output)

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph_Index, img_output)

    elif vGlyph.func == 'vglClThreshold': #Function Threshold


        img_output= vl.create_blank_image_as(img_output)
        vglClThreshold(img_input,img_output, np.float32(0.5))
        img_out_path = 'imgtst/'
        salvando2d(img_output,img_out_path+"imgTst1.png")

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph_Index, img_output)



    elif vGlyph.func == 'vglSaveImage':

        if img_input is not None:

            # SAVING IMAGE img
            vpath = vGlyph.lst_par[0].getValue()

            # Rule3: In a sink glyph, images (one or more) can only be input parameters            
            vl.vglSaveImage(vpath, img_output)

            # Actions after glyph execution
            GlyphExecutedUpdate(vGlyph_Index, None)
       
img_input = None
img_output = None