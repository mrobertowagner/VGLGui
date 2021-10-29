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
    img_in_path  = "standard_test_images/lena_color_512.tif"
    nSteps       = 1000
    img_out_path = "output"
  else:
    img_in_path = sys.argv[1]
    nSteps    = int(sys.argv[2])
    img_out_path= sys.argv[3]

  msg = ""

  CPU = cl.device_type.CPU #2
  GPU = cl.device_type.GPU #4
  vl.vglClInit(GPU)

  img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
  vl.vglLoadImage(img_input)
  if( img_input.getVglShape().getNChannels() == 3 ):
    vl.rgb_to_rgba(img_input)

  img_input.list()
  vl.vglClUpload(img_input)
  img_input.list()
  
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


  #Blur
  vglClBlurSq3(img_input, img_output)
  t0 = datetime.now()
  for i in range( nSteps ):
    vglClBlurSq3(img_input, img_output)
  t1 = datetime.now()
  t = t1 - t0
  media = (t.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClBlurSq3: " + str(media) + " ms\n"

  salvando2d(img_output, img_out_path+"img-vglClBlurSq3.jpg")
  vl.rgb_to_rgba(img_output)



  vglClConvolution(img_input, img_output, convolution_window_2d_3x3, np.uint32(3), np.uint32(3))
   #Runtime
  t0 = datetime.now()
  for i in range( nSteps ):
    vglClConvolution(img_input, img_output, convolution_window_2d_3x3, np.uint32(3), np.uint32(3))
  t1 = datetime.now()
  diff = t1 - t0
  med = (diff.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClConvolution_3x3: " + str(med) + " ms\n"
  salvando2d(img_output, img_out_path+"img-vglClConvolution_3x3.jpg")
  vl.rgb_to_rgba(img_output)

  vglClConvolution(img_input, img_output, convolution_window_2d_5x5, np.uint32(5), np.uint32(5))
  #Runtime
  t0 = datetime.now()
  for i in range( nSteps ):
    vglClConvolution(img_input, img_output, convolution_window_2d_5x5, np.uint32(5), np.uint32(5))
  t1 = datetime.now()
  diff = t1 - t0
  med = (diff.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClConvolution_5x5: " + str(med) + " ms\n"
  
  salvando2d(img_output, img_out_path+"img-vglClConvolution_5x5.jpg")
  vl.rgb_to_rgba(img_output)


  vglClInvert(img_input, img_output)
  #Runtime
  t0 = datetime.now()
  for i in range( nSteps ):
    vglClInvert(img_input, img_output)
  t1 = datetime.now()
  diff = t1 - t0
  med = (diff.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClInvert: " + str(med) + " ms\n"

  salvando2d(img_output, img_out_path+"img-vglClInvert.jpg")
  vl.rgb_to_rgba(img_output)

  vglClThreshold(img_input, img_output, np.float32(0.5))
 #Runtime
  t0 = datetime.now()
  for i in range( nSteps ):
    vglClThreshold(img_input, img_output,np.float32(0.5))
  t1 = datetime.now()
  diff = t1 - t0
  med = (diff.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClThreshold: " + str(med) + " ms\n"
  
  salvando2d(img_output, img_out_path+"img-vglClThreshold.jpg")
  vl.rgb_to_rgba(img_output)

  vglClCopy(img_input, img_output)
  #Runtime
  t0 = datetime.now()
  for i in range( nSteps ):
    vglClCopy(img_input, img_output)
  t1 = datetime.now()
  diff = t1 - t0
  med = (diff.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClCopy: " + str(med) + " ms\n"
  salvando2d(img_output, img_out_path+"img-vglClCopy.jpg")
  vl.rgb_to_rgba(img_output)

  vglClRgb2Gray(img_input, img_output)
  #Runtime
  t0 = datetime.now()
  for i in range( nSteps ):
    vglClRgb2Gray(img_input, img_output)
  t1 = datetime.now()
  diff = t1 - t0
  med = (diff.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClRgb2Gray: " + str(med) + " ms\n"

  salvando2d(img_output, img_out_path+"img-vglClRgb2Gray.jpg")
  vl.rgb_to_rgba(img_output)
  
  result = vglClEqual(img_input, img_output)
  #Runtime
  t0 = datetime.now()
  for i in range( nSteps ):
    result = vglClEqual(img_input, img_output)
  t1 = datetime.now()
  diff = t1 - t0
  med = (diff.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClEqual: " + str(med) + " ms\n"
  msg = msg + "    Imagens iguais, result = %d" % result +"\n" 

  result = vglClEqual(img_input, img_output)
  #Runtime
  t0 = datetime.now()
  for i in range( nSteps ):
    result = vglClEqual(img_input, img_output)
  t1 = datetime.now()
  diff = t1 - t0
  med = (diff.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClEqual: " + str(med) + " ms\n"
  msg = msg + "    Imagens diferentes, result = %d" % result +"\n" 
  
  
  
  

  print("-------------------------------------------------------------")
  print(msg)
  print("-------------------------------------------------------------")

  img_input = None
  img_output = None
  
  convolution_window_2d_5x5 = None
  convolution_window_2d_3x3 = None
