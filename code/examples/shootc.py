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
Deforming at circle to a C

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
maxIter = 0
logging.basicConfig(level=logging.DEBUG,format="[%(filename)s:%(lineno)s - %(funcName)6s() ] %(message)s")
fixed = '../data/shootc1.npy'
moving = '../data/shootc2.npy'
pointsPerAxis = 1
border = 110
order = 2
weights = [0*1e-5, 1]
weights = weights/np.sum(weights)
smoothing = 0.02
splineS = None
SIGMAF = 2.**(-2)
h = 1. / pointsPerAxis
immT = None


# generate image
Nx = 301
Ny = 301
c = np.floor([Nx/2,Ny/2])

# horizontal "bars"
im1 = np.zeros((Nx, Ny))
d1 = 20
d2 = 20
disp = -25
im1[c[0]-d1+disp:c[0]+d1+disp,c[1]-d2:c[1]+d2] = 255

im2 = im1

# save images
plt.imshow(im1, interpolation='nearest')
#plt.show(block=True)
plt.gcf().savefig('../data/barim1-1.png')
np.save(fixed,im1)
plt.imshow(im2, interpolation='nearest')
#plt.show(block=True)
plt.gcf().savefig('../data/barim1-2.png')
np.save(moving,im2)

N = 1
DIM = 2
q = np.array([[0.0,0.0]])
p = np.zeros([N,DIM])
mu_1 = np.zeros([N,DIM,DIM])
mu_2 = np.zeros([N,DIM,DIM,DIM])

# initial conditions
#p = np.array([[1.0,0.0]])
#mu_1 = np.array([[[1.0 , 0.0],[0.0,1.0]]])
mu_2 = -400*np.array([[[[1.0,0.0],[0.0,0.0]] , [[0.0,0.0],[0.0,0.0]]]])

# post process
for d in range(DIM): # make triangular
    mu_2[0,d] = 0.5*(mu_2[0,d] + mu_2[0,d].T)
    print mu_2[0,d]

sim = imsim.get(pointsPerAxis, immname=moving, imfname=fixed, immT=immT, border=border, normalize=True, visualize=visualize, order=order, smoothscaleFactor=smoothing, SIGMAF=SIGMAF, h=h, splineS=splineS)

logging.info("initial point configuration: %s",sim['initial'])

(fstate,res) = match.match(sim,sim['SIGMA'],weights,initial=sim['initial'],initialMomentum=(p,mu_1,mu_2),gradTol=gradTol,order=order,scalegrad=scalegrad,maxIter=maxIter,visualize=visualize, visualizeIterations=visualizeIterations)

#print res

if True: # res.success:
    match.genStateData(fstate,sim)
