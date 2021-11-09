# -*- coding: utf-8 -*-
#!/usr/bin/env python2
import my
import cv2
import numpy as np
import time as t
from datetime import datetime

def close(im, kernel, iterations=1):
  imdil = cv2.dilate(im, kernel, iterations)
  result = cv2.erode(imdil, kernel, iterations)
  return result

def blackhat(im, kernel, iterations=1):
  result = close(im, kernel, iterations)
  return result

def smooth(im, diam=3):
  result = cv2.GaussianBlur(im, (diam, diam), 0)
  return(result)

def image_equal(im0, im1):
  return (sum(sum(im0 != im1)) == 0)


def reconstruct(im):
  kernel = np.ones((3, 3), np.uint8)
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
    #c = c + 1
  return imt1



TEST1 = False
TEST2 = False
TEST3 = True
total = 0.0
msg = ""
if __name__ == "__main__":
  nSteps = 1000
  filename = "images/1_good.jpg"
  img = my.imread(filename)
  
  msg = ""
  if (TEST1):
    h = my.hist(img)
    my.showhist(h, 10)

  if (TEST2):
    my.imshow(my.histeq(imgray))

  if (TEST3):

    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #my.imshow(imgray)
    t0 = datetime.now()
    for i in range( nSteps ):
      imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execucoes do metodo Rgb2Gray: " + str(media) + " ms\n"
    total = total + media

    #2 suavização
    imsmooth = smooth(imgray, 5)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imsmooth = smooth(imgray, 5)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execuções do metodo Convolution: " + str(media) + " ms\n"
    total = total + media

    #my.imshow(imsmooth)

    #3 black hat
    kernel = np.ones((5, 5), np.uint8)
    imbh = blackhat(imsmooth, kernel, iterations=2)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imbh = blackhat(imsmooth, kernel, iterations=2)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execucoes do metodo Closing: " + str(media) + " ms\n"
    total = total + media

    #my.imshow(imbh)

    #4 black hat menos imagem de entrada
    result = imbh - imsmooth

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      result = imbh - imsmooth
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execucoes do metodo Sub: " + str(media) + " ms\n"
    total = total + media

    #my.imshow(my.histeq(result))

    #5 threshold
    imthresh = my.thresh(result, 3)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imthresh = my.thresh(result, 3)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execucoes do metodo Threshold: " + str(media) + " ms\n"
    total = total + media

    #my.imshow(imthresh)

    #6 opening by reconstruction: erosão seguida da dilatação condicional até estabilização
    imopenrec = reconstruct(imthresh)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imopenrec = reconstruct(imthresh)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo médio de " +str(nSteps)+ " execucoes do metodo Reconstruct: " + str(media) + " ms\n"
    total = total + media

    #my.imshow(imopenrec)

with open('files/PYTHON_TEST.txt', 'w') as arquivo:
    #print(msg)
    print(msg, file=arquivo)
    msg1 = "Valor total do tempo médio: "+str(total)
    print(msg1, file=arquivo)
print("-------------------------------------------------------------")
print(msg)
print("-------------------------------------------------------------")
print("Valo total médio: "+ str(total))
