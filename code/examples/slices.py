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

Bars, elongated

"""

import match as match
import matching.imagesim as imsim

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import logging

# match options
scalegrad = True
visualize = True
visRes = 70j
visualizeIterations = False
gradTol = None # 1e-5
maxIter = 100
logging.basicConfig(level=logging.DEBUG,format="[%(filename)s:%(lineno)s - %(funcName)6s() ] %(message)s")
fixed = '../data/slices1.npy'
moving = '../data/slices2.npy'
pointsPerAxis = 3
border = 10
order = 2
weights = [5*1e-3, 1] # [5*1e-3, 1] for ppa 4
weights = weights/np.sum(weights)
smoothing = 0.2
splineS = None
SIGMAF = 2.**(-0) * pointsPerAxis / 3. # SIGMA constant, equal to pointsPerAxis = 3 * factor
h = 1. / pointsPerAxis
immT = None

# generate image
import scipy.io
images = scipy.io.loadmat('../data/MRI/2dslices.mat')
im1 = images['im1'][90:160,50:120]
im2 = images['im2'][90:160,50:120]
#im1 = images['im1'] #[90:160,50:120]
#im2 = images['im2'] #[90:160,50:120]

# save images (and plot)
plt.figure(0)
plt.imshow(im1, interpolation='nearest', origin='lower')
plt.gray()
np.save(fixed,im1)
plt.figure(1)
plt.imshow(im2, interpolation='nearest', origin='lower')
plt.gray()
np.save(moving,im2)
import scipy
#scipy.misc.imsave('../data/slices1-full.png', im1*10)
#scipy.misc.imsave('../data/slices2-full.png', im2*10)
#plt.show(block=True)

sim = imsim.get(pointsPerAxis, immname=moving, imfname=fixed, immT=immT, border=border, normalize=False, visualize=visualize, order=order, smoothscaleFactor=smoothing, SIGMAF=SIGMAF, h=h, splineS=splineS, visRes=visRes)

logging.info("initial point configuration: %s",sim['initial'])

(fstate,res) = match.match(sim,sim['SIGMA'],weights,initial=sim['initial'],gradTol=gradTol,order=order,scalegrad=scalegrad,maxIter=maxIter,visualize=visualize, visualizeIterations=visualizeIterations)

#print res

if True: # res.success:
    match.genStateData(fstate,sim)
