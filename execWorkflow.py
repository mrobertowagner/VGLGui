#!/usr/bin/env python3

import pyopencl as cl       # OPENCL LIBRARY
import vgl_lib as vl        # VGL LIBRARYS
import numpy as np          # TO WORK WITH MAIN
from cl2py_shaders import * # IMPORTING METHODS
from PIL import Image       #IMPORTING METHODS TO DISPLAY IMAGE
import os
import sys                  # IMPORTING METHODS FROM VGLGui
from readWorkflow import *
import time as t


os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
sys.path.append(os.getcwd())

# Actions after glyph execution
def GlyphExecutedUpdate(vGlyph_Index):

    # Rule8: Glyphs have a list of entries. When all entries are READY=TRUE, the glyph changes status to READY=TRUE (function ready to run)
    lstGlyph[vGlyph_Index].setGlyphReady(True)

    # Rule10: Glyph becomes DONE = TRUE after its execution
    #         Assign done to glyph
    lstGlyph[vGlyph_Index].setGlyphDone(True)

    # Rule6: Edges whose source glyph has already been executed, and which therefore already had their image generated, have READY=TRUE (image ready to be processed).
    #        Reading the image from another glyph does not change this status.
    # Check the list of connections
    for i_Con, vConnection in lstConnection:

        # Checks if the executed glyph is the origin of any glyph
        if lstGlyph[vGlyph_Index].glyph_id == vConnection.output_glyph_id:

            # Assign read-ready to connection
            lstConnection[vConnection].setReadyConnection

            # Finds the glyphs that originate from the executed glyph
            for i_Gli, vGlyph in lstGlyph:

                if vGlyph.glyph_id == vConnection.output_glyph_id:

                    # Set READY = TRUE to the Glyph input
                    for vGlyphIn in lstGlyph[i_Gli].lst_input:
                        lstGlyph[i_Gli].input_glyph_id.setGlyphReadyInput(True, vConnection.input_varname)

# Program execution

# Reading the workflow file and loads into memory all glyphs and connections
# Rule7: Glyphs have READY (ready to run) and DONE (executed) status, both status start being FALSE
fileRead(lstGlyph, lstConnection)

vl.vglClInit() 

# Update the status of glyph entries
for vGlyph_Index, vGlyph in enumerate(lstGlyph):

    # Rule9: Glyphs whose status is READY=TRUE (ready to run) are executed. Only run the glyph if all its entries are
    try:
        if not vGlyph.getGlyphReady():
            raise Error("Rule9: Invalid Glyph: ",{vGlyph.glyph_id})
    except ValueError:
        print("Rule9: Glyph not ready for processing." , {vGlyph.glyph})

    if vGlyph.func == 'vglLoadImage':

        # Read "-filename" entry from glyph vglLoadImage
        img_in_path = vGlyph.lst_par[0].getValue()               
        img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())

        vl.vglLoadImage(img_input)
        if( img_input.getVglShape().getNChannels() == 3 ):
            vl.rgb_to_rgba(img_input)

        vl.vglClUpload(img_input)

        # Rule2: In a source glyph, images (one or more) can only be output parameters.
        for i_Con in lstConnection:
            if lstConnection[i_Con].output_glyph_id == vGlyph.Glyph_Id:
                lstConnection[i_Con].setImageConnection = img_input

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph_Index)
                                
    elif vGlyph.func == 'vglCreateImage':

        # Create output image
        img_output = vl.create_blank_image_as(img_input)
        img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
        
        # Save new image
        vl.vglSaveImage(vGlyph.lst_par[1].getValue(), img_output)

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph_Index)

    elif vGlyph.func == 'vglClBlurSq3': #Function blur

        # Create output image
        img_output = vl.create_blank_image_as(img_input)
        img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
        
        # Apply BlurSq3 function
        vglClBlurSq3(img_input, img_output)

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph_Index)

    elif vGlyph.func == 'vglClThreshold': #Function Threshold

        # Create output image
        img_output = vl.create_blank_image_as(img_input)
        img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
        
        # Apply Threshold function
        vglClThreshold(img_input, img_output, np.float32(0.5))

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph_Index)

    elif vGlyph.func == 'ShowImage':

        # Rule3: In a sink glyph, images (one or more) can only be input parameters             
        img = Image.open(vGlyph.lst_par[1].getValue())
        img.show()

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph_Index)

    elif vGlyph.func == 'vglSaveImage':

        # SAVING IMAGE img
        ext = vGlyph.lst_par[0].getValue().split(".")
        ext.reverse()

        img = vGlyph.lst_par[1].getValue()

        vl.vglClDownload(vGlyph.lst_par[1].getValue())

        # Rule3: In a sink glyph, images (one or more) can only be input parameters             
        vl.vglSaveImage(ext, img)

        # Actions after glyph execution
        GlyphExecutedUpdate(vGlyph_Index)
       
img_input = None
img_output = None