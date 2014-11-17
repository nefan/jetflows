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
Generate simple images for testing 2nd order features
"""


import match as match
import matching.imagesim as imsim

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import logging

# match options
scalegrad = True
visualize = True
visualizeIterations = False
gradTol = None # 1e-5
maxIter = 500
logging.basicConfig(level=logging.DEBUG,format="[%(filename)s:%(lineno)s - %(funcName)6s() ] %(message)s")
fixed = '../data/2ndim1.npy'
moving = '../data/2ndim2.npy'
pointsPerAxis = 1
border = 100
order = 1
weights = [0*1.5*1e-3, 1]
weights = weights/np.sum(weights)
smoothing = 0.03 # this looks good!
splineS = 2**4
SIGMAF = 2.**(-0)
h = 1./(pointsPerAxis+1)


# generate image
Nx = 301
Ny = 301
c = np.floor([Nx/2,Ny/2])
print c

#c[0] = c[0]-6

im1 = np.zeros((Ny, Ny))
d1 = 25
d2 = 20
trs = 0
x = np.arange(c[0]-d1,c[0]+d1+1,.1)
y = -0*((x-c[0])/d1)*d2+c[1]+trs
# draw
im1 = Image.fromarray(im1)
draw = ImageDraw.Draw(im1)
draw.line(list(np.vstack((x,y)).T.flatten()),fill=255,width=2)
del draw
im1 = np.asarray(im1)

im2 = np.zeros((Ny, Ny))
d1 = 25
d2 = 20
trs = 0
r = 3.
x = np.arange(c[0]-d1,c[0]+d1+1,.1)
#y = np.sin(x*0.25*2.*np.pi/d1-.5*np.pi)*d2+c[1]+trs-d2
#y = np.sign((x-c[0]))*((x-c[0])/d1)**2*d2+c[1]+trs
#y = ((x-c[0])/d1)**2*d2+c[1]+trs
# draw
im2 = Image.fromarray(im2)
draw = ImageDraw.Draw(im2)
#draw.line(list(np.vstack((x,y)).T.flatten()),fill=255,width=1)
draw.ellipse((c[0]-r,c[1]-r,c[0]+r,c[1]+r), fill=255)
del draw
im2 = np.asarray(im2)

## smooth
#var1 = 2
#var2 = 2
#im1 = ndimage.gaussian_filter(im1, sigma=(var1, var1), order=0)
#im2 = ndimage.gaussian_filter(im2, sigma=(var2, var2), order=0)

plt.imshow(im1, interpolation='nearest')
#plt.show(block=True)
plt.gcf().savefig('../data/2ndim1.png')
np.save(fixed,im1)

plt.imshow(im2, interpolation='nearest')
#plt.show(block=True)
plt.gcf().savefig('../data/2ndim2.png')
np.save(moving,im2)


## sine curves
#fixed = '../data/sinecim1.npy'
#moving = '../data/sinecim2.npy'
#pointsPerAxis = 3
#border = 100
#order = 2
#weights = [1e-5, 1]
#weights = weights/np.sum(weights)
#smoothing = 0.25
#SIGMAF = 1.
#h = 2.**(-3)
#immT = None

# imm transform
imm = np.load(moving)
[Nx,Ny] = np.shape(imm)
center = np.array([[Nx/2.,Ny/2.]])
def T(XY):
    R = np.array([[1.,0.],[0.,1.]])
    res = np.dot(R,XY.T-np.kron(np.ones((1,XY.shape[0])),center.T))+np.kron(np.ones((1,XY.shape[0])),center.T)
    return res.T
immT = None

sim = imsim.get(pointsPerAxis, immname=moving, imfname=fixed, immT=immT, border=border, normalize=True, visualize=visualize, order=order, smoothscaleFactor=smoothing, SIGMAF=SIGMAF, h=h, splineS=splineS)

logging.info("initial point configuration: %s",sim['initial'])

(fstate,res) = match.match(sim,sim['SIGMA'],weights,initial=sim['initial'],gradTol=gradTol,order=order,scalegrad=scalegrad,maxIter=maxIter,visualize=visualize, visualizeIterations=visualizeIterations)

#print res

if True: # res.success:
    match.genStateData(fstate,sim)
