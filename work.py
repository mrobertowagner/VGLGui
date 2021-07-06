#!/usr/bin/env python3
# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

# TO WORK WITH MAIN
import numpy as np

# IMPORTING METHODS
from cl2py_shaders import * 

from PIL import Image

# IMPORTING METHODS FROM VGLGui
import sys
import os

sys.path.append(os.getcwd())

from readWorkflow import *


import time as t

# Reading the workflow file and loads into memory all glyphs and connections
fileRead(lstGlyph)



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
nSteps		= int(sys.argv[2])
img_out_path= sys.argv[3]

msg = ""

vl.vglClInit()

img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
vl.vglLoadImage(img_input)
if( img_input.getVglShape().getNChannels() == 3 ):
    vl.rgb_to_rgba(img_input)

vl.vglClUpload(img_input)

img_output = vl.create_blank_image_as(img_input)
img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())


for vGlyph in lstGlyph:
    print(vGlyph.func)
    if vGlyph.func == 'kvision': #imagem de entrada
        print("")
    
    elif vGlyph.func == 'vstrflat': #elementro estruturante 
        print("")
    
    elif vGlyph.func == 'vglBlurSq3': 
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
        

    elif vGlyph.func == 'vglClCopy': 
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

    elif vGlyph.func == 'vglClThreshold': 
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
        

print("-------------------------------------------------------------")
print(msg)
print("-------------------------------------------------------------")

PATH = 'tmp/testes'
for image in os.listdir(PATH):
    if image.endswith('.jpg'):
        img = Image.open(image,"r")

#im= Image.open('tmp/testes/img-vglClCopy.jpg')
#im.show()

img_input = None
img_output = None
