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
import matplotlib.pyplot as plt
from PIL import Image
def close(im, kernel, iterations=1):
  imdil = cv2.erode(im, kernel, iterations)
  result = cv2.dilate(imdil, kernel, iterations)
  return result

def closeth (im, kernel, iterations=1):
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
  kernel = np.ones((11,11), np.uint8)
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




msg = ""
media = 0.0
nSteps = 10

TEST1 = False
TEST2 = False
TEST3 = True

if __name__ == "__main__":
  filename = "images/galeao.jpg"
  f_pil = Image.open('images/galeao.jpg').convert('L') # must be read as grayscale
  img = my.imread(filename)
  imgray = my.imreadgray(filename)
  recimg = "images/rec.png" 
  

  if (TEST3):
    
    f = np.array(f_pil)
    (fig, axes) = plt.subplots(nrows=1, ncols=1,figsize=(7, 7))
    axes.set_title('f')
    axes.imshow(f, cmap='gray')
    axes.axis('off')
    


    th=ia.iaopenth(f,ia.iasedisk(5))
    (fig, axes) = plt.subplots(nrows=1, ncols=1,figsize=(7, 7))
    axes.set_title('th')
    axes.imshow(th, cmap='gray')
    axes.axis('off')
    my.imshow(th)


    bin1=ia.iathreshad(th,30)
    (fig, axes) = plt.subplots(nrows=1, ncols=1,figsize=(7, 7))
    axes.set_title('f, bin1')
    axes.imshow(ia.iagshow(f, bin1).transpose(1, 2, 0), cmap='gray')
    axes.axis('off')
    my.imshow(bin1)


    m1=ia.iathin(bin1)
    m2=ia.iathin(m1,ia.iaendpoints())
    m=ia.iaareaopen(m2,1000,ia.iasebox())
    (fig, axes) = plt.subplots(nrows=1, ncols=1,figsize=(7, 7))
    axes.set_title('f, m1, m2, m')
    axes.imshow(ia.iagshow(f, m1, m2, m).transpose(1, 2, 0), cmap='gray')
    axes.axis('off')
    my.imshow(m)


    g=ia.iainfrec(ia.iagray(m), th)
    (fig, axes) = plt.subplots(nrows=1, ncols=1,figsize=(7, 7))
    axes.set_title('g')
    axes.imshow(g, cmap='gray')
    axes.axis('off')
    my.imshow(g)


    final=ia.iathreshad(g, 20)
    (fig, axes) = plt.subplots(nrows=1, ncols=1,figsize=(7, 7))
    axes.set_title('f, final')
    axes.imshow(ia.iagshow(f, final).transpose(1, 2, 0), cmap='gray')
    axes.axis('off')
    my.imshow(final)

















    
    '''
    msg = ""
    f_pil = Image.open('images/1_good.jpg').convert('L') # must be read as grayscale
    f = np.array(f_pil)

    disk = cv2.getStructuringElement(2, (51,51))

    #PASSO 1 - OPENTH
    #imopenth = closeth(imgray,disk,1)
    imopenth = ia.iaopenth(imgray,ia.iasedisk(15))
    my.imshow(imopenth)
    #th=ia.iacloseth(f,ia.iasedisk(31))


    

    bin1=my.thresh(imopenth,0.0711)
    #bin1 = ia.iathreshad(imopenth,30)
    my.imshow(bin1)



    
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    hitmiss = cv2.morphologyEx(bin1, cv2.MORPH_HITMISS, kernel1)

    m=ia.iaareaopen(hitmiss,1000,ia.iasebox())
    plt.savefig('areaclose.png', format='png')
    my.imshow(m)
    kernel = np.ones((17,17), np.uint8)
    tophat_img = cv2.morphologyEx(m, cv2.MORPH_BLACKHAT, kernel)
    my.imshow(tophat_img)

    


    #Ag = ia.iainfrec(ia.iagray(tophat_img),imopenth)
    #my.imshow(g)

    #final = my.thresh(rec,0.078431)
    #my.imshow(final)


    #my.imshow(rec)
    final = cv2.max(tophat_img,imgray)
    my.imshow(final)
    '''
    


    
  
    


