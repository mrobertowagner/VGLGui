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


msg = ""
media = 0.0
nsteps = 1000

TEST1 = False
TEST2 = False
TEST3 = True

if __name__ == "__main__":
  filename = "images/01_test.png"
  img = my.imread(filename)
  imgray = my.imreadgray(filename)


  if (TEST3):
    msg = ""
    media = 0.0
    nsteps = 1000
    total = 0.0
    kernel = np.ones((5, 5), np.uint8)

    #1 Convolution
    imconv = smooth(img, 5)

    for i in range(0, 5):
            p = 0
            inicio = t.time()
            while(p<1000):
                imconv = smooth(img, 5)
                p = p + 1
            fim = t.time()
            media = media + (fim - inicio)
    total = total + ((media/5)*1000)
    msg = msg + "Convolution runtime\t "+str( round((media/5)*1000, 4) ) +"ms\n"

    #my.imshow(imconv)

    #2 Dilate    
    imdil = cv2.dilate(imconv, kernel, 1)

    for i in range(0, 5):
            p = 0
            inicio = t.time()
            while(p<1000):
                imdil = cv2.dilate(imconv, kernel, 1)
                p = p + 1
            fim = t.time()
            media = media + (fim - inicio)
    total = total + ((media/5)*1000)
    msg = msg + "Dilate runtime\t "+str( round((media/5)*1000, 4) ) +"ms\n"

    #my.imshow(imdil)

    #3 Erode
    imerode = cv2.erode(imdil, kernel, 1)

    for i in range(0, 5):
            p = 0
            inicio = t.time()
            while(p<1000):
                imerode = cv2.erode(imdil, kernel, 1)
                p = p + 1
            fim = t.time()
            media = media + (fim - inicio)
    total = total + ((media/5)*1000)
    msg = msg + "Erode runtime\t "+str( round((media/5)*1000,4) ) +"ms\n"
    #my.imshow(imerode)


print("-------------------------------------------------------------")            
print(msg)
print("-------------------------------------------------------------")
print("Total runtime "+str(round(total,2))+"ms")
print("-------------------------------------------------------------")