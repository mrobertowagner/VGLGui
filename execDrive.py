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
  while (not is_equal):
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
total = 0.0
if __name__ == "__main__":
  filename = "images/01_test.png"
  img = my.imread(filename)
  imgray = my.imreadgray(filename)
  media = 0.0
  nSteps = 1000
  msg = ""
  if (TEST1):
    h = my.hist(img)
    my.showhist(h, 10)

  if (TEST2):
    my.imshow(my.histeq(imgray))

  if (TEST3):
    #1 extração do canal verde
    imgreen = img[:,:,1]

    #2 suavização
    imsmooth = smooth(img, 5)

    #Runtime
    t0 = datetime.now()
    for i in range( nSteps ):
      imsmooth = smooth(img, 5)
    t1 = datetime.now()
    t = t1 - t0
    media = (t.total_seconds() * 1000) / nSteps
    msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo Convolution: " + str(media) + " ms\n"

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
    msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo Black Hat: " + str(media) + " ms\n"

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
    msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo Sub: " + str(media) + " ms\n"

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
    msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo Threshold: " + str(media) + " ms\n"

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
    msg = msg + "Tempo de " +str(nSteps)+ " execuções do metódo Threshold: " + str(media) + " ms\n"

    #my.imshow(imopenrec)


print("-------------------------------------------------------------")            
print(msg)
print("-------------------------------------------------------------")
