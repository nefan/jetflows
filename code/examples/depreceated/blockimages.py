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
Matching of simple images testing 1st and 2nd order features

Blocks

"""

import __builtin__
#__builtin__.__debug = True

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
fixed = '../data/blockim1.npy'
moving = '../data/blockim2.npy'
pointsPerAxis = 2
border = 110
order = 2
weights = [0*1e-5, 1]
weights = weights/np.sum(weights)
smoothing = 0.20
splineS = 2.**(-1)
SIGMAF = 2.**(-0)
h = 1. / pointsPerAxis


# generate images
Nx = 300
Ny = 300
c = np.floor([Nx/2,Ny/2])

im1 = np.zeros((Nx, Ny))
d1 = 45
d2 = 5
im1[c[0]-d1:c[0]+d1,c[1]-d2:c[1]+d2] = 255

im2 = np.zeros((Nx, Ny))
d1 = 37
d2 = 5
im2[c[0]-d1:c[0]+d1,c[1]-d2:c[1]+d2] = 255
im2[c[0]-1.0*d1:c[0]-.25*d1,c[1]:c[1]+20] = 255

## smooth
#var1 = 2
#var2 = 2
#im1 = ndimage.gaussian_filter(im1, sigma=(var1, var1), order=0)
#im2 = ndimage.gaussian_filter(im2, sigma=(var2, var2), order=0)

# save images
plt.imshow(im1, interpolation='nearest')
#plt.show(block=True)
plt.gcf().savefig('../data/blockim1.png')
np.save('../data/blockim1.npy',im1)

plt.imshow(im2, interpolation='nearest')
#plt.show(block=True)
plt.gcf().savefig('../data/blockim2.png')
np.save('../data/blockim2.npy',im2)

## imm transform
#imm = np.load(moving)
#[Nx,Ny] = np.shape(imm)
#center = np.array([[Nx/2.,Ny/2.]])
#def T(XY):
#    R = np.array([[1./3.,0],[0,1]])
#    res = np.dot(R,XY.T-np.kron(np.ones((1,XY.shape[0])),center.T))+np.kron(np.ones((1,XY.shape[0])),center.T)
#    return res.T
immT = None

sim = imsim.get(pointsPerAxis, immname=moving, imfname=fixed, immT=immT, border=border, normalize=True, visualize=visualize, order=order, smoothscaleFactor=smoothing, SIGMAF=SIGMAF, h=h, splineS=splineS)

logging.info("initial point configuration: %s",sim['initial'])

(fstate,res) = match.match(sim,sim['SIGMA'],weights,initial=sim['initial'],gradTol=gradTol,order=order,scalegrad=scalegrad,maxIter=maxIter,visualize=visualize, visualizeIterations=visualizeIterations)

#print res

if True: # res.success:
    match.genStateData(fstate,sim)
