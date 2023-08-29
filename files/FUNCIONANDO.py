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
import time as t
import gc
from datetime import datetime
from areaopen import *
import ia870 as ia
import my
"""
  THIS BENCHMARK TOOL EXPECTS 2 ARGUMENTS:
  ARGV[1]: PRIMARY 2D-IMAGE PATH (COLORED OR GRAYSCALE)
    IT WILL BE USED IN ALL KERNELS
  ARGV[2]: SECONDARY 2D-IMAGE PATH (COLORED OR GRAYSCALE)
    IT WILL BE USED IN THE KERNELS THAT NEED
    TWO INPUT IMAGES TO WORK PROPERLY
  THE RESULT IMAGES WILL BE SAVED AS IMG-[PROCESSNAME].JPG
"""


def iase2off(Bc,option='neigh'):
    '''Converts structuring element to list of neighbor offsets in graph image'''
    if len(Bc.shape) == 2:
        h,w = Bc.shape
        hc,wc = h//2,w//2
        B = Bc.copy()
        B[hc,wc] = 0  # remove origin
        off = np.transpose(B.nonzero()) - np.array([hc,wc])
        if option == 'neigh':
            return off  # 2 columns x n. of neighbors rows
        elif option == 'fw':
            i = off[:,0] * w + off[:,1]
            return off[i>0,:]  # only neighbors higher than origin in raster order
        elif option == 'bw':
            i = off[:,0] * w + off[:,1]
            return off[i<0,:]  # only neighbors less than origin in raster order
        else:
            assert 0,'options are neigh, fw or bw. It was %s'% option
            return None
    elif len(Bc.shape) == 3:
        d,h,w = Bc.shape
        dc,hc,wc = d//2,h//2,w//2
        B = Bc.copy()
        B[dc,hc,wc] = 0  # remove origin
        off = np.transpose(B.nonzero()) - np.array([dc,hc,wc])
        if option == 'neigh':
            return off  # 2 columns x n. of neighbors rows
        elif option == 'fw':
            i = off[:,0] * h*w + off[:,1] * w + off[:,2]
            return off[i>0,:]  # only neighbors higher than origin in raster order
        elif option == 'bw':
            i = off[:,0] * h*w + off[:,1] * w + off[:,2]
            return off[i<0,:]  # only neighbors less than origin in raster order
        else:
            assert 0,'options are neigh, fw or bw. It was %s'% option
            return None
    else:
        print('2d or 3d only. Shape was', len(Bc.shape))
        return None

def iasecross(r=1):
    from ia870.iasesum import iasesum
    from ia870.iabinary import iabinary

    B = iasesum( iabinary([[0,1,0],
                           [1,1,1],
                           [0,1,0]]),r)
    return B


def iaNlut(s,offset):
    '''Precompute array of neighbors. Optimized by broadcast.
    s - image shape
    offset - offset matrix, 2 columns (dh,dw) by n. of neighbors rows
    '''
    if len(s)== 2:
        H,W = s
        n = H*W
        hi = np.arange(H).reshape(-1,1)
        wi = np.arange(W).reshape(1,-1)
        hoff = offset[:,0]
        woff = offset[:,1]
        h = hi + hoff.reshape(-1,1,1)
        w = wi + woff.reshape(-1,1,1)
        h[(h<0) | (h>=H)] = n
        w[(w<0) | (w>=W)] = n
        Nlut = np.clip(h * W + w,0,n)
        return Nlut.reshape(offset.shape[0],-1).transpose()
    elif len(s)== 3:
        D,H,W = s
        n = D*H*W
        di = np.arange(D).reshape(-1, 1, 1)
        hi = np.arange(H).reshape( 1,-1, 1)
        wi = np.arange(W).reshape( 1, 1,-1)
        doff = offset[:,0]
        hoff = offset[:,1]
        woff = offset[:,2]
        d = di + doff.reshape(-1,1,1,1)
        h = hi + hoff.reshape(-1,1,1,1)
        w = wi + woff.reshape(-1,1,1,1)
        d[(d<0) | (d>=D)] = n
        h[(h<0) | (h>=H)] = n
        w[(w<0) | (w>=W)] = n
        Nlut = np.clip(d * H*W + h * W + w,0,n)
        return Nlut.reshape(offset.shape[0],-1).transpose()
    else:
        print('s must have 2 or 3 dimensions')
    return None
      
def iaareaopen_eq(f, a, Bc=iasecross()):

    
    y = np.zeros_like(f.ipl)
    k1 = f.ipl.min()
    k2 = f.ipl.max()
    for k in range(k1,k2+1):   # gray-scale, use thresholding decomposition
      fk = (f.ipl >= k)
      fo = iaareaopen(fk,a,Bc)
      if not fo.any():
        break
      y = iaiaunion(y, MT.iagray(fo,f.dtype,k))
    return y
'''
def iaareaopen(f,a,Bc=iasecross()):
    a = -a
    s = f.shape
    g = np.zeros_like(f).ravel()
    f1 = np.concatenate((f.ravel(),np.array([0])))
    area = -np.ones((f1.size,), np.int32)
    N = iaNlut(s, iase2off(Bc))
    pontos = f1.nonzero()[0]
    pontos = pontos[np.lexsort((np.arange(0,-len(pontos),-1),f1[pontos]))[::-1]]
    for p in pontos:
        for v in N[p]:
            if f1[p] < f1[v] or (f1[p] == f1[v] and v < p):
                rv = find_area(area, v)
                if rv != p:
                    if area[rv] > a or f1[p] == f1[rv]:
                        area[p] = area[p] + area[rv]
                        area[rv] = p
                    else:
                        area[p] = a
    for p in pontos[::-1]:
        if area[p] >= 0:
            g[p] = g[area[p]]
        else:
            if area[p] <= a:
                g[p] = f1[p]
    return g.reshape(s)
'''
def iaareaopen(f,a,Bc=iasecross()):
    a = -a
    s =f.oclPtr.shape
    print("Shape=",s)
    #g = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    g= np.zeros(100)
    print("Zeros=",g)
    sd = f.ipl
    fravel = [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,255,255,255,255,255,255,255,255,255,0,0,0,255,255,255,255,255,255,255,255,255,0,255,255,255,255,255,255,255,0,255,255,255,255,255,255,255,255,255,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0]
    print("TAMANHO",len(fravel))
    f1 = np.concatenate((fravel,np.array([0])))
    print("F1=",f1)
    size = f.oclPtr.shape[0]*f.oclPtr.shape[1]
    area = -np.ones((size,), np.int32)
    print("Area=",area)
    N = iaNlut(s, iase2off(Bc))
    print("N=",N)
    pontos = f1.nonzero()[0]
    print("Pontos=",pontos)
    pontos = pontos[np.lexsort((np.arange(0,-len(pontos),-1),f1[pontos]))[::-1]]
    print("Pontos1=",pontos)
    
    for p in pontos:
        for v in N[p]:
            if f1[p] < f1[v] or (f1[p] == f1[v] and v < p):
                #print(len(N[p]))
                rv = find_area(area, v)
                if rv != p:
                    if area[rv] > a or f1[p] == f1[rv]:
                        area[p] = area[p] + area[rv]
                        area[rv] = p
                    else:
                        area[p] = a
    for p in pontos[::-1]:
        if area[p] >= 0:
            g[p] = g[area[p]]
        else:
            if area[p] <= a:
                g[p] = f1[p]
    #print(g.reshape(s))
    return g.reshape(s)
  
def find_area(area, i):
    lista = []
    while area[i] >= 0:
        lista.append(i)
        i = area[i]
    area[lista] = i
    return i
  
def find_area(area, i):
    lista = []
    while area[i] >= 0:
        lista.append(i)
        i = area[i]
    area[lista] = i
    return i

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
    img_in_path  = "images/index.png"
    nSteps       = 1000
    img_out_path = "output"
  msg = ""

  CPU = cl.device_type.CPU #2
  GPU = cl.device_type.GPU #4
  vl.vglClInit(GPU)

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

  img_output1= vl.create_blank_image_as(img_input)
  img_output1.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
  vl.vglAddContext(img_output1, vl.VGL_CL_CONTEXT())
  
  vglClRgb2Gray(img_input,img_output)
  salvando2d(img_output, img_out_path+"img-vgd3x3.jpg")
  vl.rgb_to_rgba(img_output)
  #my.imshow(img_output.ipl)
  vglClThreshold(img_output,img_output1,0.00784)
  salvando2d(img_output1, img_out_path+"img-vgd3x3.jpg")
  vl.rgb_to_rgba(img_output1)
  #my.imshow(img_output1.ipl)
  
  def func (img):
    
    #print("RODOU")
    #my.imshow(img_output.ipl)
    
    d =iaareaopen(img_output,1000,ia.iasebox())
    print("rodou")
    my.imshow(d)    
    
    #print(img.ipl)

  func(img_output)
    

'''

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

  vglClDilate(img_input, img_output, convolution_window_2d_3x3, np.uint32(3), np.uint32(3))
   #Runtime
  t0 = datetime.now()
  for i in range( nSteps ):
    vglClDilate(img_input, img_output, convolution_window_2d_3x3, np.uint32(3), np.uint32(3))
  t1 = datetime.now()
  diff = t1 - t0
  med = (diff.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClDilate_3x3: " + str(med) + " ms\n"
  salvando2d(img_output, img_out_path+"img-vglClDilate_3x3.jpg")
  vl.rgb_to_rgba(img_output)

  vglClDilate(img_input, img_output, convolution_window_2d_5x5, np.uint32(5), np.uint32(5))
  #Runtime
  t0 = datetime.now()
  for i in range( nSteps ):
    vglClDilate(img_input, img_output, convolution_window_2d_5x5, np.uint32(5), np.uint32(5))
  t1 = datetime.now()
  diff = t1 - t0
  med = (diff.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClDilate_5x5: " + str(med) + " ms\n"
  
  salvando2d(img_output, img_out_path+"img-vglClDilate_5x5.jpg")
  vl.rgb_to_rgba(img_output)

  vglClErode(img_input, img_output, convolution_window_2d_3x3, np.uint32(3), np.uint32(3))
   #Runtime
  t0 = datetime.now()
  for i in range( nSteps ):
    vglClErode(img_input, img_output, convolution_window_2d_3x3, np.uint32(3), np.uint32(3))
  t1 = datetime.now()
  diff = t1 - t0
  med = (diff.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClErode_3x3: " + str(med) + " ms\n"
  salvando2d(img_output, img_out_path+"img-vglClErode_3x3.jpg")
  vl.rgb_to_rgba(img_output)

  vglClErode(img_input, img_output, convolution_window_2d_5x5, np.uint32(5), np.uint32(5))
  #Runtime
  t0 = datetime.now()
  for i in range( nSteps ):
    vglClErode(img_input, img_output, convolution_window_2d_5x5, np.uint32(5), np.uint32(5))
  t1 = datetime.now()
  diff = t1 - t0
  med = (diff.total_seconds() * 1000) / nSteps
  msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo vglClErode_5x5: " + str(med) + " ms\n"
  
  salvando2d(img_output, img_out_path+"img-vglClErode_5x5.jpg")
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
'''
