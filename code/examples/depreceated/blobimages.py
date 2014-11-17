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

Blobs

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
gradTol = None # 1e-5
logging.basicConfig(level=logging.DEBUG,format="[%(filename)s:%(lineno)s - %(funcName)6s() ] %(message)s")
fixed = '../data/blobim1.npy'
moving = '../data/blobim2.npy'
pointsPerAxis = 6
border = 100
order = 0
weights = [0*1e-5, 1]
weights = weights/np.sum(weights)
smoothing = 0.5
SIGMAF = 2.**(0)
h = None # 2.**(-1)


# generate images
Nx = 300
Ny = 300
c = np.floor([Nx/2,Ny/2])

im1 = np.zeros((Nx, Ny))
im1[c[0]-1:c[0]+1,c[1]-1:c[1]+1] = 255

im2 = np.zeros((Ny, Ny))
d1 = 20
d2 = 40
im2[c[0]-d1:c[0]+d1,c[1]-d2:c[1]+d2] = 255

# smooth
var1 = 5
var2 = 5
im1 = ndimage.gaussian_filter(im1, sigma=(var1, var1), order=0)
im2 = ndimage.gaussian_filter(im2, sigma=(var2, var2), order=0)

# save images
plt.imshow(im1, interpolation='nearest')
plt.gcf().savefig('../data/blobim1.png')
#plt.show(block=True)
np.save('../data/blobim1.npy',im1)

plt.imshow(im2, interpolation='nearest')
plt.gcf().savefig('../data/blobim2.png')
#plt.show(block=True)
np.save('../data/blobim2.npy',im2)

## imm transform
#imm = np.load(moving)
#[Nx,Ny] = np.shape(imm)
#center = np.array([[Nx/2.,Ny/2.]])
#def T(XY):
#    R = np.array([[1./3.,0],[0,1]])
#    res = np.dot(R,XY.T-np.kron(np.ones((1,XY.shape[0])),center.T))+np.kron(np.ones((1,XY.shape[0])),center.T)
#    return res.T
immT = None

sim = imsim.get(pointsPerAxis, immname=moving, imfname=fixed, immT=immT, border=border, normalize=True, visualize=visualize, order=order, smoothscaleFactor=smoothing, SIGMAF=SIGMAF, h=h)

logging.info("initial point configuration: %s",sim['initial'])

(fstate,res) = match.match(sim,sim['SIGMA'],weights,initial=sim['initial'],gradTol=gradTol,order=order,scalegrad=scalegrad)

#print res

if True: # res.success:
    match.genStateData(fstate,sim)

