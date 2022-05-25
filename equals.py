#!/usr/bin/env python3

# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

# TO WORK WITH MAIN
import numpy as np

# IMPORTING METHODS
from cl2py_shaders import * 
from vglClUtil import * 

import time as t
import sys
from datetime import datetime
"""
  THIS BENCHMARK TOOL EXPECTS 2 ARGUMENTS:
  ARGV[1]: PRIMARY 2D-IMAGE PATH (COLORED OR GRAYSCALE)
    IT WILL BE USED IN ALL KERNELS
  ARGV[2]: SECONDARY 2D-IMAGE PATH (COLORED OR GRAYSCALE)
    IT WILL BE USED IN THE KERNELS THAT NEED
    TWO INPUT IMAGES TO WORK PROPERLY
  THE RESULT IMAGES WILL BE SAVED AS IMG-[PROCESSNAME].JPG
"""


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

if __name__ == "__main__":
  
  """
    CL.IMAGE OBJECTS
  """

  if (len(sys.argv) < 4):
    img_in_path  = "images/standard_test_images/lena_color_512.tif"
    nSteps       = 1000
    img_out_path = "images/output"
  else:
    img_in_path = sys.argv[1]
    nSteps    = int(sys.argv[2])
    img_out_path= sys.argv[3]

  msg = ""

  CPU = cl.device_type.CPU #2
  GPU = cl.device_type.GPU #4
  vl.vglClInit(CPU)

  img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
  vl.vglLoadImage(img_input)
  if( img_input.getVglShape().getNChannels() == 3 ):
    vl.rgb_to_rgba(img_input)

  #img_input.list()
  vl.vglClUpload(img_input)
  #img_input.list()
  
  img_output = vl.create_blank_image_as(img_input)
  img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
  vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())
  
  convolution_window_2d_3x3 = np.array((  (1/16, 2/16, 1/16),
                                          (2/16, 4/16, 2/16),
                                          (1/16, 2/16, 1/16) ), np.float32)
  convolution_window_2d_5x5 = np.array((  (1/256, 4/256,  6/256,  4/256,  1/256),
                                          (4/256, 16/256, 24/256, 16/256, 4/256),
                                          (6/256, 24/256, 36/256, 24/256, 6/256),
                                          (4/256, 16/256, 24/256, 16/256, 4/256),
                                          (1/256, 4/256,  6/256,  4/256,  1/256) ), np.float32)


  #print("BEFORE vglClBlurSq3: shape = " + str(img_input.shape))
  #print(type(img_input))
  #vglClThreshold(img_input,img_output,1)
  img_output = img_input
  result = vglClEqual(img_input, img_output)
  print(result)
  #Runtime
  
  

  print("-------------------------------------------------------------")
  print(msg)
  print("-------------------------------------------------------------")

  img_input = None
  img_output = None
  
  convolution_window_2d_5x5 = None
  convolution_window_2d_3x3 = None
