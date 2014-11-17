#!/usr/bin/python
#
# This file is part of jetflows.
#
# Copyright (C) 2014, Henry O. Jacobs (hoj201@gmail.com), Stefan Sommer (sommer@di.ku.dk)
# https://github.com/nefan/jetflows.git
#
# jetflows is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# jetflows is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with jetflows.  If not, see <http://www.gnu.org/licenses/>.
#


"""
Matching of simple images of testing 1st order features

Bars, angle

"""

import match as match
import matching.imagesim as imsim

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
#from PIL import Image, ImageDraw
import logging

# match options
scalegrad = True
visualize = True
visualizeIterations = False
gradTol = None # 1e-5
maxIter = 500
logging.basicConfig(level=logging.DEBUG,format="[%(filename)s:%(lineno)s - %(funcName)6s() ] %(message)s")
fixed = '../data/barim2-1.npy'
moving = '../data/barim2-1.npy'
pointsPerAxis = 2 # for 4 points
#pointsPerAxis = 1 # for 1 point
border = 130
order = 0
weights = [0*1e-5, 1]
weights = weights/np.sum(weights)
smoothing = 0.3
splineS = 2**(-8)
SIGMAF = 3./4.
h = 1./(pointsPerAxis+1)


# generate image
Nx = 300
Ny = 300
c = np.floor([Nx/2,Ny/2])

im1 = np.zeros((Ny, Ny))
d1 = 50
d2 = 4
im1[c[0]-d1:c[0]+d1,c[1]-d2:c[1]+d2] = 255
#im2 = np.zeros((Ny, Ny))
#d1 = 15
#d2 = 5
#x = np.arange(c[0]-d1,c[0]+d1,.1)
#y = (x-c[0])*d1/d2+c[1]
## draw
#im2 = Image.fromarray(im2)
#draw = ImageDraw.Draw(im2)
#draw.line(list(np.vstack((x,y)).T.flatten()),fill=255,width=1)
#del draw
#im2 = np.asarray(im2)

## smooth
#var1 = 2
#var2 = 2
#im1 = ndimage.gaussian_filter(im1, sigma=(var1, var1), order=0)
#im2 = ndimage.gaussian_filter(im2, sigma=(var2, var2), order=0)

# save image
plt.imshow(im1, interpolation='nearest')
#plt.show(block=True)
plt.gcf().savefig('../data/barim2-1.png')
np.save(fixed,im1)

#plt.imshow(im2, interpolation='nearest')
#plt.show(block=True)
#plt.gcf().savefig('../data/barim2-2.png')
#np.save(moving,im2)

# imm transform
imm = np.load(moving)
[Nx,Ny] = np.shape(imm)
center = np.array([[Nx/2.,Ny/2.]])
def T(XY):
    phi = 2*np.pi*0.09
    R = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
    res = np.dot(R,XY.T-np.kron(np.ones((1,XY.shape[0])),center.T))+np.kron(np.ones((1,XY.shape[0])),center.T)
    return res.T
immT = T

sim = imsim.get(pointsPerAxis, immname=moving, imfname=fixed, immT=immT, border=border, normalize=True, visualize=visualize, order=order, smoothscaleFactor=smoothing, SIGMAF=SIGMAF, h=h, splineS=splineS)

logging.info("initial point configuration: %s",sim['initial'])

(fstate,res) = match.match(sim,sim['SIGMA'],weights,initial=sim['initial'],gradTol=gradTol,order=order,scalegrad=scalegrad,maxIter=maxIter,visualize=visualize, visualizeIterations=visualizeIterations)

#print res

if True: # res.success:
    match.genStateData(fstate,sim)
