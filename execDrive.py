import my
import cv2
import numpy as np
import time as t

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
    print(c)
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
  filename = "images/01_test.png"
  img = my.imread(filename)
  imgray = my.imreadgray(filename)
  media = 0.0
  nsteps = 1000
  msg = ""
  if (TEST1):
    h = my.hist(img)
    my.showhist(h, 10)

  if (TEST2):
    my.imshow(my.histeq(imgray))

  if (TEST3):
    #1 extração do canal verde
    imgreen = img[:,:,1]

    for i in range(0, 5):
            p = 0
            inicio = t.time()
            while(p<nsteps):
                imgreen = img[:,:,1]
                p = p + 1
            fim = t.time()
            media = media + (fim - inicio)
    msg = msg + "Extraction runtime\t "+str( round((media/5), 9) ) +"s\n"

    #2 suavização
    imsmooth = smooth(imgreen, 5)

    for i in range(0, 5):
            p = 0
            inicio = t.time()
            while(p<nsteps):
                imsmooth = smooth(imgreen, 5)
                p = p + 1
            fim = t.time()
            media = media + (fim - inicio)
    msg = msg + "Convolution runtime\t "+str( round((media/5), 9) ) +"s\n"

    my.imshow(imsmooth)

    #3 black hat
    kernel = np.ones((5, 5), np.uint8)
    imbh = blackhat(imsmooth, kernel, iterations=2)

    for i in range(0, 5):
            p = 0
            inicio = t.time()
            while(p<nsteps):
                imbh = blackhat(imsmooth, kernel, iterations=2)
                p = p + 1
            fim = t.time()
            media = media + (fim - inicio)
    msg = msg + "Closing runtime\t "+str( round((media/5), 9) ) +"s\n"

    my.imshow(imbh)

    #4 black hat menos imagem de entrada
    result = imbh - imsmooth

    for i in range(0, 5):
            p = 0
            inicio = t.time()
            while(p<nsteps):
                result = imbh - imsmooth
                p = p + 1
            fim = t.time()
            media = media + (fim - inicio)
    msg = msg + "Sub Runtime\t "+str( round((media/5), 9) ) +"s\n"  

    my.imshow(my.histeq(result))

    #5 threshold    
    imthresh = my.thresh(result, 3)

    for i in range(0, 5):
            p = 0
            inicio = t.time()
            while(p<nsteps):
                imthresh = my.thresh(result, 3)
                p = p + 1
            fim = t.time()
            media = media + (fim - inicio)
    msg = msg + "Treshold Runtime\t "+str( round((media/5), 9) ) +"s\n" 

    my.imshow(imthresh)

    #6 opening by reconstruction: erosão seguida da dilatação condicional até estabilização
    imopenrec = reconstruct(imthresh)

    for i in range(0, 5):
            p = 0
            inicio = t.time()
            while(p<nsteps):
                imopenrec = reconstruct(imthresh)
                p = p + 1
            fim = t.time()
            media = media + (fim - inicio)
    msg = msg + "Reconstruct Runtime\t "+str( round((media/5), 9) ) +"s\n" 

    my.imshow(imopenrec)

print(msg)