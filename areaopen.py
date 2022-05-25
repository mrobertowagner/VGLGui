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
import queue
import time as t
import gc
from datetime import datetime
from areaopen import *
import ia870 as ia


from numpy import *

def seshow(B, option="NORMAL"):
    from ia870 import iaisbinary, iaintersec, iagray, iabinary
    from ia870 import iasedil, iaero, iabshow

    option = option.upper()
    if option=='NON-FLAT':
        y = array([0],int32)
        if iaisbinary(B):
            B = iaintersec( iagray(B,'int32'),0)
    elif option=='NORMAL':
        if iaisbinary(B):    y=iabinary([1])
        else:
           y=array([0],int32)
    elif option=='EXPAND':
        assert iaisbinary(B), 'This option is only available with flat SE'
        y = iasedil( iabinary([1]),B)
        b1= iabinary(y>=0)
        b0= b1<0
        b0[shape(b0)[0]//2, shape(b0)[1]//2] = 1
        y = iabshow(b1,y,b0)
        return y
    else:
        print('iaseshow: not a valid flag: NORMAL, EXPAND or NON-FLAT')

    y = iasedil(y,B)
    return y

def blob(f, measurement, output="image"):
    #import numpy
    from numpy import newaxis, ravel, zeros, sum, nonzero, array, asanyarray
    from string import lower

    measurement = lower(measurement)
    output      = lower(output)
    if len(f.shape) == 1: f = f[newaxis,:]
    assert measurement in ('area', 'centroid', 'boundingbox'), 'pymorph.blob: Unknown measurement type \'%s\'' % measurement
    if output == 'data':
        y = []
    elif measurement == 'centroid':
        y = zeros(f.shape,numpy.bool)
    else:
        y = zeros(f.shape,numpy.int32)
    for obj_id in range(f.max()):
        blob = (f == (obj_id+1))
        if measurement == 'area':
            area = blob.sum()
            if output == 'data': y.append(area)
            else               : y += area*blob
        elif measurement == 'centroid':
            indy, indx  = nonzero(blob)
            cy = sum(indy) // len(indy)
            cx = sum(indx) // len(indx)
            if output == 'data': y.append( (cy, cx) )
            else               : y[cy, cx] = 1
        elif measurement == 'boundingbox':
            col, = nonzero(blob.any(0))
            row, = nonzero(blob.any(1))
            if output == 'data': y.append([col[0],row[0],col[-1]+1,row[-1]+1])
            else:
                y[row[0]:row[-1], col[0]] = 1
                y[row[0]:row[-1],col[-1]] = 1
                y[row[0], col[0]:col[-1]] = 1
                y[row[-1],col[0]:col[-1]] = 1
    return asanyarray(y)


def label(f, Bc=None):
    if Bc is None: Bc = secross()
    f = pad4n(f, Bc, 0)
    neighbours = se2flatidx(f,Bc)
    labeled = f * 0
    f = f.ravel()
    labeledflat=labeled.ravel()
    label = 1
    queue = []
    for i in range(f.size):
        if f[i] and labeledflat[i] == 0:
            labeledflat[i]=label
            queue=[i+bi for bi in neighbours]
            while queue:
                ni=queue.pop()
                if f[ni] and labeledflat[ni] == 0:
                    labeledflat[ni]=label
                    for n in neighbours+ni:
                        queue.append(n)
            label += 1
    return labeled[1:-1,1:-1]



def binary(f, k=1):
   
    from numpy import asanyarray
    f = asanyarray(f)
    return (f >= k)


def union(f1, f2, *args):
 
    from numpy import maximum
    y = maximum(f1,f2)
    for f in args:
        y = maximum(y,f)
    return y.astype(f1.dtype)
  
def secross(r=1):
    
    return sesum(binary([[0,1,0],
                         [1,1,1],
                         [0,1,0]]),
                 r)

def isbinary(f):
    return f.dtype == bool


def threshad(f, f1, f2=None):
 
    if f2 is None:
        return f1 <= f
    return (f1 <= f) & (f <= f2)


def sesum(B=None, N=1):
    
    if B is None: B = secross()
    if N==0:
        if isbinary(B): return binary([[1]])
        else:           return to_int32([[0]]) 
    NB = B
    for i in range(N-1):
        NB = sedilate(NB,B)
    return NB

def areaopen(f, a, Bc=None):
    if Bc is None: Bc = secross()
    fr = label(f,Bc)    
    g = blob(fr,'area')
    y = threshad(g,a)
    
    for k in range(k1,k2+1):
      fk = threshad(f,k)
      fo = areaopen(fk,a,Bc)
      if isequal(fo,zero):
        break
      y = union(y, gray(fo,datatype(f),k))
    return y
def pad4n(f, Bc, value, scale=1):
    
    from numpy import ones, array

    if type(Bc) is not array:
      Bc = seshow(Bc)
    Bh, Bw = Bc.shape
    assert Bh%2 and Bw%2, 'structuring element must be odd sized'
    ch, cw = scale * Bh/2, scale * Bw/2
    g = value * ones( f.shape + scale * (array(Bc.shape) - 1))
    g[ ch: -ch, cw: -cw] = f
    return g.astype(f.dtype)

def mat2set(A):
    
    from numpy import take, ravel, nonzero, transpose, newaxis

    if len(A.shape) == 1: A = A[newaxis,:]
    offsets = nonzero(ravel(A) - limits(A)[0])[0]
    if len(offsets) == 0: return ([],[])
    h,w = A.shape
    x = [0,1]
    x[0] = offsets//w - (h-1)//2
    x[1] = offsets%w - (w-1)//2
    x = transpose(x)
    return x,take(ravel(A),offsets)

def find_area(area, i):
    lista = []
    while area[i] >= 0:
        lista.append(i)
        i = area[i]
    area[lista] = i
    return i




def iasecross(r=1):
    from ia870.iasesum import iasesum
    from ia870.iabinary import iabinary

    B = iasesum( iabinary([[0,1,0],
                           [1,1,1],
                           [0,1,0]]),r)
    return B


def iase2off(Bc,option='neigh'):
    '''Converts structuring element to list of neighbor offsets in graph image'''
    if len(Bc.shape) == 2:
        h,w = Bc.shape
        hc,wc = h//2,w//2
        B = Bc.copy()
        B[hc,wc] = 0  # remove origin
        off = np.transpose(B.nonzero()) - np.array([hc,wc])
        if option == 'neigh':
            return off  # 2 columns x n. of neighbors rows
        elif option == 'fw':
            i = off[:,0] * w + off[:,1]
            return off[i>0,:]  # only neighbors higher than origin in raster order
        elif option == 'bw':
            i = off[:,0] * w + off[:,1]
            return off[i<0,:]  # only neighbors less than origin in raster order
        else:
            assert 0,'options are neigh, fw or bw. It was %s'% option
            return None
    elif len(Bc.shape) == 3:
        d,h,w = Bc.shape
        dc,hc,wc = d//2,h//2,w//2
        B = Bc.copy()
        B[dc,hc,wc] = 0  # remove origin
        off = np.transpose(B.nonzero()) - np.array([dc,hc,wc])
        if option == 'neigh':
            return off  # 2 columns x n. of neighbors rows
        elif option == 'fw':
            i = off[:,0] * h*w + off[:,1] * w + off[:,2]
            return off[i>0,:]  # only neighbors higher than origin in raster order
        elif option == 'bw':
            i = off[:,0] * h*w + off[:,1] * w + off[:,2]
            return off[i<0,:]  # only neighbors less than origin in raster order
        else:
            assert 0,'options are neigh, fw or bw. It was %s'% option
            return None
    else:
        print('2d or 3d only. Shape was', len(Bc.shape))
        return None

def iaNlut(s,offset):
    '''Precompute array of neighbors. Optimized by broadcast.
    s - image shape
    offset - offset matrix, 2 columns (dh,dw) by n. of neighbors rows
    '''
    if len(s)== 2:
        H,W = s
        n = H*W
        hi = np.arange(H).reshape(-1,1)
        wi = np.arange(W).reshape(1,-1)
        hoff = offset[:,0]
        woff = offset[:,1]
        h = hi + hoff.reshape(-1,1,1)
        w = wi + woff.reshape(-1,1,1)
        h[(h<0) | (h>=H)] = n
        w[(w<0) | (w>=W)] = n
        Nlut = np.clip(h * W + w,0,n)
        return Nlut.reshape(offset.shape[0],-1).transpose()
    elif len(s)== 3:
        D,H,W = s
        n = D*H*W
        di = np.arange(D).reshape(-1, 1, 1)
        hi = np.arange(H).reshape( 1,-1, 1)
        wi = np.arange(W).reshape( 1, 1,-1)
        doff = offset[:,0]
        hoff = offset[:,1]
        woff = offset[:,2]
        d = di + doff.reshape(-1,1,1,1)
        h = hi + hoff.reshape(-1,1,1,1)
        w = wi + woff.reshape(-1,1,1,1)
        d[(d<0) | (d>=D)] = n
        h[(h<0) | (h>=H)] = n
        w[(w<0) | (w>=W)] = n
        Nlut = np.clip(d * H*W + h * W + w,0,n)
        return Nlut.reshape(offset.shape[0],-1).transpose()
    else:
        print('s must have 2 or 3 dimensions')
    return None


def iaareaopen(f,a,Bc=iasecross()):
    a = -a
    s = f.shape
    g = np.zeros_like(f).ravel()
    f1 = np.concatenate((f.ravel(),np.array([0])))
    area = -np.ones((f1.size,), np.int32)
    N = iaNlut(s, iase2off(Bc))
    pontos = f1.nonzero()[0]
    pontos = pontos[np.lexsort((np.arange(0,-len(pontos),-1),f1[pontos]))[::-1]]
    for p in pontos:
        for v in N[p]:
            if f1[p] < f1[v] or (f1[p] == f1[v] and v < p):
                rv = find_area(area, v)
                if rv != p:
                    if area[rv] > a or f1[p] == f1[rv]:
                        area[p] = area[p] + area[rv]
                        area[rv] = p
                    else:
                        area[p] = a
    for p in pontos[::-1]:
        if area[p] >= 0:
            g[p] = g[area[p]]
        else:
            if area[p] <= a:
                g[p] = f1[p]
    return g.reshape(s)
  
def find_area(area, i):
    lista = []
    while area[i] >= 0:
        lista.append(i)
        i = area[i]
    area[lista] = i
    return i






