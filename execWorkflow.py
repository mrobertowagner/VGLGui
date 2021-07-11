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

#sys.argv[1] = 'lena_1024.tif'
#img_in_path = sys.argv[1]
#nSteps		= int(sys.argv[2])
#img_out_path= sys.argv[3]

msg = ""

vl.vglClInit() 

'''img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
vl.vglLoadImage(img_input)
if( img_input.getVglShape().getNChannels() == 3 ):
    vl.rgb_to_rgba(img_input)

vl.vglClUpload(img_input)

img_output = vl.create_blank_image_as(img_input)
img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )'''



#Update the status of glyph entries
for vGlyph in lstGlyph:
    if vGlyph.func == 'vglLoadImage':
        sys.argv[1] = 'lena_1024.tif'
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
                   

    elif vGlyph.func == 'vglClBlurSq3': #Function blur
        sys.argv[1] = 'lena_1024.tif'
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
 
        vglClBlurSq3(img_input, img_output)
        media = 0.0
        for i in range(0, 5):
            p = 0
            inicio = t.time()
            while(p < nSteps):
                vglClBlurSq3(img_input, img_output)
                p = p + 1
                fim = t.time()
                media = media + (fim-inicio)
        salvando2d(img_output, img_out_path+"img-vglClBlurSq3.jpg")
        vl.rgb_to_rgba(img_output)
        msg = msg + "Tempo de execução do método vglClBlurSq3:\t\t" +str( round( ( media / 5 ), 9 ) ) +"s\n"

    elif vGlyph.func == 'vglClCopy': #Function copy
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

    elif vGlyph.func == 'vglClThreshold': #Function Threshold
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
         

    elif vGlyph.func == 'vglShowImage':
        #img = Image.open('tmp/testes/img-vglClThreshold.jpg')
        #img1 = Image.open('tmp/testes/img-vglClCopy.jpg')
        #img2 = Image.open('tmp/testes/img-vglClBlurSq3.jpg')

        #img.show()
        #img1.show()
        #img2.show()
        '''
    if vGlyph.func == 'vglSaveImage':
        
        salvando2d(img_output, img_out_path+"img-vglClBlurSq3.jpg")
        vl.rgb_to_rgba(img_output)
        msg = msg + "Tempo de execução do método vglClBlurSq3:\t\t" +str( round( ( media / 5 ), 9 ) ) +"s\n"

        
        salvando2d(img_output, img_out_path+"img-vglClCopy.jpg")
        vl.rgb_to_rgba(img_output)
        msg = msg + "Tempo de execução do método vglClCopy:\t\t\t" +str( round( (media / 5), 9 ) ) +"s\n"

        
        salvando2d(img_output, img_out_path+"img-vglClThreshold.jpg")
        vl.rgb_to_rgba(img_output)
        msg = msg + "Tempo de execução do método vglClThreshold:\t\t" +str( round( (media / 5), 9 ) ) +"s\n"
    '''

            

    #Glyph execute
    '''
    if vGlyph.getGlyphReady() and vGlyph.getGlyphDone() == True:
        #Image Input
        if vGlyph.func == 'in':
            print("pass")

        elif vGlyph.func == 'vglBlurSq3': #Structuring element 
            vglClBlurSq3(img_input, img_output)
            media = 0.0
            for i in range(0, 5):
                p = 0
                inicio = t.time()
                while(p < nSteps):
                    vglClBlurSq3(img_input, img_output)
                    p = p + 1
                    fim = t.time()
                    media = media + (fim-inicio)
            salvando2d(img_output, img_out_path+"img-vglClBlurSq3.jpg")
            vl.rgb_to_rgba(img_output)
            msg = msg + "Tempo de execução do método vglClBlurSq3:\t\t" +str( round( ( media / 5 ), 9 ) ) +"s\n"

        #elif vGlyph.func == 'out': #Image Output
        #    img_output = vl.create_blank_image_as(img_input)
        #    img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
        #    vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())

        elif vGlyph.func == 'vglClCopy': #Function copy
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

        elif vGlyph.func == 'vglClThreshold': #Function erosion
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
        
        elif vGlyph.func == 'vglShowImage':
            img = Image.open('images/belchior.jpg')
            img.show()
       '''

            
# Shows the content of the Glyphs
#procShowInfo()
print("-------------------------------------------------------------")
print(msg)
print("-------------------------------------------------------------")
img_input = None
img_output = None
