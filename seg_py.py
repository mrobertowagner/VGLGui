import my
import cv2
import numpy as np


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
  filename = "images/eye.png"
  img = my.imread(filename)
  imgray = my.imreadgray(filename)

  if (TEST1):
    h = my.hist(img)
    my.showhist(h, 10)

  if (TEST2):
    my.imshow(my.histeq(imgray))

  if (TEST3):
    kernel = np.ones((5, 5), np.uint8)

    imsmooth = smooth(imgray, 5)    
    imbh = blackhat(imsmooth, kernel, iterations=2)
    result = imbh - imsmooth
  
    my.imshow(imsmooth)
    my.imshow(my.histeq(result))
    imthresh = my.thresh(result, 3)
    my.imshow(imthresh)
    my.imshow(reconstruct(imthresh))
