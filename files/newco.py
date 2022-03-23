# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import my
import cv2
import numpy as np
import time as t
from datetime import datetime
from skimage import io, color, morphology

import skimage
def close(im, kernel, iterations=1):
  imdil = cv2.dilate(im, kernel, iterations)
  result = cv2.erode(imdil, kernel, iterations)
  return result

def openth (im, kernel, iterations=1):
  imerod = cv2.erode(im,kernel,iterations)
  imdil = cv2.dilate(imerod,kernel,iterations)
  result = im - imdil
  return result

def rgb2gray(rgb):
  r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray

def blackhat(im, kernel, iterations=1):
  result = close(im, kernel, iterations)
  return result

def smooth(im, diam=3):
  result = cv2.GaussianBlur(im, (diam, diam), 0)
  return(result)

def image_equal(im0, im1):
  return (sum(sum(im0 != im1)) == 0)


def reconstruct(im):
  kernel = np.ones((17, 17), np.uint8)
  imero =  cv2.dilate(im, kernel)
  c = 0
  imt0 = imero
  imt1 = cv2.dilate(imt0, kernel)
  is_equal = image_equal(imt0, imt1)
  while (not is_equal):
    print(c)
    imt0 = imt1
    imdil = cv2.erode(imt0, kernel)
    imt1 = np.minimum(imdil, im)
    is_equal = image_equal(imt0, imt1)
    c = c + 1
  return imt1


def recbydil(im):
  kernel = np.ones((17,17), np.uint8)
  imdil = cv2.dilate(im,kernel)
  immin = np.minimum(imdil,im)

  return immin

def infrerec1(im,imgaux):
  kernel = np.ones((17,17),np.uint8)
  imero = cv2.erode(imgaux,kernel)
  c = 0
  imt0 = imero
  imt1 = cv2.dilate(imt0,kernel)
  is_equal = image_equal(imt0,imt1)
  while (not is_equal):
    #print(c)
    imt0 = imt1
    imdil = cv2.dilate(imt0,kernel)
    imt1 = np.minimum(imdil,im)
    is_equal = image_equal(imt0, imt1)
    c = c + 1
  return imt1


  
  
  
'''
01 a = mmreadgray(’galeao.jpg’)
02 b = mmopenth( a, mmsedisk(5))
03 c = mmaddm(b, 180) # just for visualization
04 d = mmthreshad( b,30)
05 e = mmthin(d)
06 f = mmareaopen(e, 1000, mmsebox())
07 g = mminfrec( mmgray(f), b)
08 h = mmthreshad( g, 20)
09 i = mmunion(a, mmgray(h))
'''

msg = ""
media = 0.0
nSteps = 10

TEST1 = False
TEST2 = False
TEST3 = True

if __name__ == "__main__":
  filename = "images/1_good.jpg"
  img = my.imread(filename)
  imgray = my.imreadgray(filename)
  

  if (TEST3):
    msg = ""
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = datetime.now()
    for i in range( nSteps ):
      imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do metódo Rgb2Gray: " + str(media) + " ms\n"
    #my.imshow(imgray)
    #kernel = np.ones((51, 51), np.uint8)
    disk = cv2.getStructuringElement(2, (51, 51))

    #PASSO 1 - OPENTH
    imopenth = openth(imgray,disk,1)
    my.imshow(imopenth)


    #PASSO 2 - THREHSOLD

    imthresh = my.thresh(imopenth, 0.117664)

    my.imshow(imthresh)


    #PASSO 3 - THINNING
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (51,51))
    hitmiss = cv2.morphologyEx(imthresh, cv2.MORPH_HITMISS, kernel1)
    #thinned = imthresh - hitmiss
    #th = morphology.thin(imthresh)
    #thinned = hitmiss
    
    '''
    skel = np.zeros(imthresh.shape, np.uint8)
    while True:
      open1 = cv2.morphologyEx(imthresh, cv2.MORPH_OPEN, kernel1)
      temp = cv2.subtract(imthresh, open1)
      eroded = cv2.erode(imthresh,kernel1)
      skel = cv2.bitwise_or(skel,temp)
      imthresh= eroded.copy()

      if cv2.countNonZero(imthresh) == 0:
        break
    '''

    my.imshow(hitmiss)

    kernel17 = np.ones((17,17), np.uint8)

    thin = cv2.subtract(imthresh,hitmiss)
    my.imshow (thin)
    #PASSO 4- INFRE - REC
    iminfrec = recbydil(thin)
    my.imshow(iminfrec)
    

    #PASSO 5 - THRESHOLD

    imthresh1 = my.thresh(iminfrec,0.078431)
    my.imshow(imthresh1)


    #PASSO 6- SUM

    imsum = cv2.max(imgray,imthresh1)
    my.imshow(imsum)
    
    

print("-------------------------------------------------------------")            
print(msg)
print("-------------------------------------------------------------")
