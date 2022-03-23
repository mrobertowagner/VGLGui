# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import my
import cv2
import numpy as np
import time as t
from datetime import datetime
from skimage import io, color, morphology
import ia870 as ia
import skimage

def close(im, kernel, iterations=1):
  imdil = cv2.erode(im, kernel, iterations)
  result = cv2.dilate(imdil, kernel, iterations)
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
  kernel = np.ones((25, 25), np.uint8)
  imero =  cv2.erode(im, kernel)
  c = 0
  imt0 = imero
  imt1 = cv2.dilate(imt0, kernel)
  is_equal = image_equal(imt0, imt1)
  while (not is_equal.all()):
    #print(c)
    imt0 = imt1
    imdil = cv2.dilate(imt0, kernel)
    imt1 = np.minimum(imdil, im)
    is_equal = image_equal(imt0, imt1)
    c = c + 1
  return imt1




def reconstruct1(im,aux):
  kernel = np.ones((29,29), np.uint8)
  imero =  aux
  c = 0
  imt0 = imero
  imt1 = cv2.erode(imt0, kernel)
  is_equal = image_equal(imt0, imt1)
  while (not is_equal.all()):
    #print(c)
    imt0 = imt1
    imdil = cv2.erode(imt0, kernel)
    imt1 = np.maximum(imdil, im)
    is_equal = image_equal(imt0, imt1)
    c = c + 1
  return imt1

def conddil(im):
  kernel = np.ones((11,11), np.uint8)
  imdil = cv2.dilate(im,kernel)
  immin = imdil - im
  return immin


def dilcond(img):
  kernel = np.ones((7, 7), np.uint8)
  imgdil = cv2.dilate(img,kernel)
  result = cv2.min(imgdil,img)

  return result



def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()
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
  recimg = "images/rec.png" 
  

  if (TEST3):
    msg = ""
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    openth = ia.iacloseth(imgray,ia.iasedisk(51))
    my.imshow(openth)

    imthresh = my.thresh(openth, 0.117664)
    my.imshow(imthresh)

    m1= ia.iathin(binth)
    m2= ia.iathin(m1,ia.iaendpoints())

    m = ia.iaareaopen(m2,1000,ia.iasebox())


    g = ia.iainfrec(ia.iagray(m),openth)

    final = ia.iathreshad(g,20)
    my.imshow(final)

    


    
