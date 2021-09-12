

'''import numpy as np
radius = 3
sigma = radius/2.

k = np.arange(2*radius +1)
row = np.exp( -(((k - radius)/(sigma))**2)/2.)
col = row.transpose()
out = np.outer(row, col)
out = out/np.sum(out)
for line in out:
    print(["%.3f" % x for x in line])'''

from math import exp
import re
def gaussian(x, mu, sigma):
  return exp( -(((x-mu)/(sigma))**2)/2.0 )

# 1 =3x3 2 = 5x5 3 = 7x7 4 = 9x9 5 = 11x11 6 = 13x13 = 7 = 15x15 = 8 = 17x17
#kernel_height, kernel_width = 7, 7
kernel_radius = 7 # for an 7x7 filter
sigma = kernel_radius/2. # for [-2*sigma, 2*sigma]

# compute the actual kernel elements
hkernel = [gaussian(x, kernel_radius, sigma) for x in range(2*kernel_radius+1)]
vkernel = [x for x in hkernel]
kernel2d = [[xh*xv for xh in hkernel] for xv in vkernel]

# normalize the kernel elements
kernelsum = sum([sum(row) for row in kernel2d])
kernel2d = [[x/kernelsum for x in row] for row in kernel2d]

#for line in kernel2d:
    #print(["%.4f" % x for x in line])

#[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#print(len(m))