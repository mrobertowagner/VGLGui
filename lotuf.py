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
  
  imdil = cv2.dilate(im,kernel,iterations)
  imerod = cv2.erode(imdil,kernel,iterations)
  result = imerod - im
  return result


def openth (im, kernel, iterations=1):
  
  
  imerod = cv2.erode(im,kernel,iterations)
  imdil = cv2.dilate(imerod,kernel,iterations)
  result = imdil - im
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

TEST1 = False
TEST2 = False
TEST3 = True

if __name__ == "__main__":
  filename = "images/galeao.jpg"
  f_pil = Image.open('images/1_good.jpg').convert('L') # must be read as grayscale
  #f_pil = Image.open('images/1_good.jpg').convert('L') # must be read as grayscale
  img = my.imread(filename)
  imgray = my.imreadgray(filename)
  recimg = "images/rec.png" 
  

  if (TEST3):
    
    f = np.array(f_pil)
    disk = cv2.getStructuringElement(2, (51,51))

    #PASSO 1 - OPENTH
    #imopenth = closeth(imgray,disk,1)
    imcloseth = closeth(f,disk)
    
    my.imshow(imcloseth)
    print("CloseTh")
    #th=ia.iacloseth(f,ia.iasedisk(31))


    

    bin1=my.thresh(imcloseth,7)
    #bin1 = ia.iathreshad(imopenth,30)
   
    my.imshow(bin1)
    print("Threshold ")

    
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    hitmiss = cv2.morphologyEx(bin1, cv2.MORPH_HITMISS, kernel1)
    
    my.imshow(hitmiss)
    print("Hit-Miss")

    
    m=ia.iaareaopen(hitmiss,1000,ia.iasebox())
    my.imshow(m)
    print("AreaOpen")
    #kernel = np.ones((17,17), np.uint8)
    #tophat_img = cv2.morphologyEx(m, cv2.MORPH_BLACKHAT, kernel)
    #my.imshow(tophat_img)

  
    
    g = ia.iainfrec(ia.iagray(m),imcloseth)
    my.imshow(g)
    print("Infrec")


    
    h = my.thresh(g, 4.5)
    my.imshow(h)
    print("Thresh")

    #my.imshow(rec)
    final = cv2.max(f,ia.iagray(h))
    my.imshow(final)
    


    
  
    


