#!/usr/bin/env python3

# OPENCL LIBRARY
from numpy.lib.shape_base import get_array_wrap
from vgl_lib import vglImage
from vgl_lib.vglImage import VglImage, vglLoadImage
from vgl_lib.struct_sizes import struct_sizes
from vgl_lib import vglClImage
from PIL import Image

import pyopencl as cl



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


img_in_path = "images/01_test.png"
img_in_path1 = "images/01_test.png"
img_out_path= "images/"

msg = ""

vl.vglClInit()

img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
vl.vglLoadImage(img_input)
if( img_input.getVglShape().getNChannels() == 3 ):
    vl.rgb_to_rgba(img_input)

vl.vglClUpload(img_input)


img_input1 = vl.VglImage(img_in_path1, None, vl.VGL_IMAGE_2D_IMAGE())
vl.vglLoadImage(img_input1)
if( img_input1.getVglShape().getNChannels() == 3 ):
    vl.rgb_to_rgba(img_input1)

vl.vglClUpload(img_input1)

img_output = vl.create_blank_image_as(img_input)
img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())



vglClEqual(img_input1,img_input,img_output)
print(img_output)

'''


fileRead(lstGlyph,lstConnection)
for vGlyph in lstGlyph:
    if vGlyph.func == "vglClErode":
        print(vGlyph.lst_par[0].getValue())
        #print(type(tratnum(vGlyph.lst_par[0].getValue())))
        #print(tratnum(vGlyph.lst_par[0].getValue()))
        #print(vGlyph.lst_par[1].getName())
        #print(type(vGlyph.lst_par[1].getValue()))




#f = '11111'
#g = tratnum(f)
#print(g)

#process(liststr)
#r = re.compile(r'[^\d ]')
#f = '-1,1,1,1@@,1,1!#!#!@!,1,1,1#!#!@!'

#s = re.findall('-?\d+\.?\d*',f)
#print(tratnum(s))
#numeros  =  "".join(re.findall("\d+",f))
#a = (r.sub('',f))
#print(s)
#print(numeros)

fileRead(lstGlyph,lstConnection)
for vGlyph in lstGlyph:
    if vGlyph.func == "vglClErode":
        print(vGlyph.lst_par[0].getValue())
        txt = vGlyph.lst_par[0].getValue()
        liststr = txt[1:-1]
        print(liststr)
        dados = []
        for linha in liststr:
            dados.append(int(linha))
        print(dados)
        print(type(dados))
        w = np.array(dados,np.float32)
        print(w)
        print(type(w))
print(w)
'''

#vglClConvolution(img_input, img_output, cv, np.uint32(3), np.uint32(3))

#salvando2d(img_output, img_out_path+"img-vglClConvolution.jpg")
#vl.rgb_to_rgba(img_output)


#vglClErode(img_input,img_output, cv, np.uint32(3), np.uint32(3))

#salvando2d(img_output, img_out_path+"img-vglClErodeCross1.png")


#vglClThreshold(img_input,img_output,np.float32(0.5))
#salvando2d(img_output, img_out_path+"img-vglClTreshold.png")
