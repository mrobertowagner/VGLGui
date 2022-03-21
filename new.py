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

    #my.imshow(imgray)
    #kernel = np.ones((51, 51), np.uint8)
    disk = cv2.getStructuringElement(2, (41,41))

    #PASSO 1 - OPENTH
    
    close =  cv2.morphologyEx(imgray, cv2.MORPH_CLOSE, disk)
    tophat = cv2.morphologyEx(close, cv2.MORPH_TOPHAT,disk)
    imopenth = tophat
    print("openth")
    my.imshow(imopenth)

    #PASSO 2 - THREHSOLD

    imthresh = my.thresh(imopenth, 0.117664)
    print("thresh")
    my.imshow(imthresh)
    


    #PASSO 3 - THINNING
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    hitmiss = cv2.morphologyEx(imthresh, cv2.MORPH_HITMISS, kernel1)
    #thinned = imthresh - hitmiss
    
    
    print("hitmiss")
    my.imshow(hitmiss)
    kernel17 = np.ones((17,17), np.uint8)
    kernel5151 = np.ones((51,51), np.uint8)

    #thin = imthresh - hitmiss
    #print("thin")
    #my.imshow(thin)

    #thin = cv2.subtract(imthresh,hitmiss)
    #my.imshow (thin)

    #reconstruc = reconstruct(hitmiss)
    #my.imshow(reconstruc)    




    kernel7 = np.ones((3,3), np.uint8)
    '''
    open_3 = cv2.morphologyEx(hitmiss, cv2.MORPH_OPEN, kernel7)
    close_3 = cv2.morphologyEx(open_3, cv2.MORPH_CLOSE, kernel7) 

    kernel11 = np.ones((5,5), np.uint8)
    open_5 = cv2.morphologyEx(close_7x7, cv2.MORPH_OPEN, kernel11)
    close_11x11 = cv2.morphologyEx(open_11x11, cv2.MORPH_CLOSE, kernel11)
    '''
    

    kernel11 = np.ones((3,3), np.uint8)
    open_11 = cv2.morphologyEx(hitmiss, cv2.MORPH_OPEN, kernel11)
    close_11 = cv2.morphologyEx(open_11, cv2.MORPH_CLOSE, kernel11)
    #my.imshow(close_11)

    kernel15 = np.ones((5,5),np.uint8)
    open_15  = cv2.morphologyEx(close_11, cv2.MORPH_OPEN, kernel15)
    close_15 = cv2.morphologyEx(open_15, cv2.MORPH_CLOSE, kernel15)
   # my.imshow(close_15)
    
    kernel19 = np.ones((7,7),np.uint8)
    open_19  = cv2.morphologyEx(close_15, cv2.MORPH_OPEN, kernel19)
    close_19 = cv2.morphologyEx(open_19 , cv2.MORPH_CLOSE, kernel19)    
    #my.imshow(close_19)

    kernel11 = np.ones((11,11), np.uint8)
    open1 = cv2.morphologyEx(close_19, cv2.MORPH_OPEN, kernel11)
    close1 = cv2.morphologyEx(open1, cv2.MORPH_CLOSE, kernel11)

    print("cond dil")
    cond = conddil(close1)
    resu = cv2.max(cond,close_15)
    my.imshow(resu)

    rresu = reconstruct(resu)
    my.imshow(rresu)
    
    #re = reconstruct(resu)




    

    '''
    kernel23 = np.ones((3,3),np.uint8)
    open_23  = cv2.morphologyEx(close_19, cv2.MORPH_OPEN, kernel17)
    close_23 = cv2.morphologyEx(open_23, cv2.MORPH_CLOSE, kernel17)
    my.imshow(close_23)
    kernel27 = np.ones((5,5),np.uint8)
    open_27  = cv2.morphologyEx(close_23, cv2.MORPH_OPEN, kernel27)
    close_27 = cv2.morphologyEx(open_27, cv2.MORPH_CLOSE, kernel27)
    my.imshow(close_27)
    kernel29 = np.ones((7,7),np.uint8)
    open_29  = cv2.morphologyEx(close_27, cv2.MORPH_OPEN, kernel29)
    close_29 = cv2.morphologyEx(open_29, cv2.MORPH_CLOSE, kernel29)
    my.imshow(close_29)
    '''
    
    


    
    '''
    print("dilatacao condicional")
    cond = conddil(close29)
    resu = cv2.max(cond,hitmiss)
    #re = reconstruct(resu)
    my.imshow(resu)

    #cond = conddil(close_23x23)
    #print("dilatacao condicional")
    #resu = cv2.max(cond,imthresh)



    
    kernel41x41 = np.ones((41,41),np.uint8)
    open_41x41 = cv2.morphologyEx(close_31x31, cv2.MORPH_OPEN, kernel41x41)
    close_41x41 = cv2.morphologyEx(open_41x41, cv2.MORPH_CLOSE, kernel41x41)
    print("aqui")
    my.imshow(close_41x41)
    
    
    
    
    dilr= cv2.dilate(close_17x17, kernel17)
    resu = cv2.max(dilr,thin)

    dilcond = cv2.dilate(resu,kernel17)
    resulcond = np.minimum(dilcond,resu)
    my.imshow(resulcond)

    

    #ecresul = reconstruct(dilr)
    #my.imshow(dilr)
    print("areaopen")
    #my.imshow(dilr)
    
    
    
    #iminfrec = reconstruct1(resu,imopenth)
    print("infrec")
    
    iminfrec = reconstruct1(resu, imopenth)
    my.imshow(iminfrec)
    
    

    #PASSO 5 - THRESHOLD
    print("thresh")
    imthresh1 = my.thresh(iminfrec,0.078431)
    
    #my.imshow(imthresh1)

    #recf = reconstruct(imthresh1)
    #re = reconstruct1(close_31x31,im)
    
    
    #PASSO 6- 
    print("sum")
    imsum = cv2.max(imthresh1,imgray)
    my.imshow(imsum)
    
    

print("-------------------------------------------------------------")            
print(msg)
print("-------------------------------------------------------------")
'''
