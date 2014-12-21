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
Test convergence of match term for different orders

"""

import __builtin__
#__builtin__.__debug = True

import matching.imagesim as imsim
import two_jets as tj

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image
import logging

logging.basicConfig(level=logging.DEBUG,format="[%(filename)s:%(lineno)s - %(funcName)6s() ] %(message)s")

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

# create .npy files
#moving = '../data/Lenna-bw.png' # smooth approx 0.02
#moving_im = load_image(moving)
#moving = '../data/Lenna-bw.npy'
#moving_im = moving_im[250:300,300:350]

# notes: smoothing 0, border 0 for the below examples
moving = '../data/sine.npy'
(x,y) = np.mgrid[0:512,0:512]
import sympy as sp
sx = sp.Symbol('x')
sy = sp.Symbol('y')
moving_im = np.sin(2.*np.pi*(3.*x+0.*y)/512.)+(x/(512.))**2 # linspace 1,6,24
e = sp.sin(2*sp.pi*(3*sx+0*sy)/512)+(sx/(512))**2
baseline_sp = np.double(sp.integrate(e**2,(sx,0,512),(sy,0,512)).evalf()/512**2)
logging.info("baseline sp: %g", baseline_sp)
#f = lambda x: np.mod(x,2.*np.pi)/np.pi if np.mod(x,2.*np.pi) < np.pi else 2-np.mod(x,2.*np.pi)/np.pi
#moving_im = np.array([f(5.*2.*np.pi*x[i,j]/512.) for i in range(0,512) for j in range(0,512)]).reshape([512,512])
#moving_im = np.sin(((x+y)/(2.*512.))**2)
#moving_im = np.sin((x+y)/(2.*512.))
#moving_im = ((x+y)/(2.*512.))**2 # linspace 0,5,18, for paper
#moving_im = (x/(512.))**2
#moving_im = (x+y)/(2.*512.) # linspace 0,5,18, for paper
#moving_im = x/(512.)
#moving_im = .7*np.ones(x.shape)
np.save(moving,moving_im)

fixed_im = np.zeros(moving_im.shape)
fixed = '../data/zero.npy'
np.save(fixed,fixed_im)

#plt.figure(22)
#plt.imshow(moving_im)
#plt.set_cmap('gray')

# match options
visualize = True
border = 0
smoothing = 0.0
SIGMAF = 2.**(-1)
splineS = 1e2

logging.info("image dimensions: %s", moving_im.shape)

#hk = np.linspace(0,5,18) # for linear and quadratic plots
hk = np.linspace(1,6,24) # np.array([2,3]) for image with points
hs = np.zeros(hk.shape)
ppas = np.zeros(hk.shape)
print "hk: " + str(hk)
res = np.zeros(np.shape(hk))
colors = ['b','r','g']

logfig = plt.figure(21)
ax1 = logfig.add_subplot(111)
ax2 = ax1.twinx()

# get baseline, order 0
# nr points
pointsPerAxis = moving_im.shape[0]
h = 1. / pointsPerAxis
# get similarity
sim = imsim.get(pointsPerAxis, immname=moving, imfname=fixed, immT=None, border=border, normalize=False, visualize=visualize, order=2, smoothscaleFactor=pointsPerAxis*smoothing, SIGMAF=SIGMAF, h=h, splineS=splineS)
DIM = tj.DIM = sim['DIM']
N = tj.N = sim['N']
logging.info("N points: %d", N)
# state
p = np.zeros([N,DIM])
mu_1 = np.zeros([N,DIM,DIM])
mu_2 = np.zeros([N,DIM,DIM,DIM])
(q,q_1,q_2) = (sim['initial'][0], np.outer(np.ones(N),np.eye(DIM)).reshape([N,DIM,DIM]), np.zeros([N,DIM,DIM,DIM]))
state = tj.weinstein_darboux_to_state(q, q_1, q_2, p, mu_1, mu_2)# initial guess + zeros

# runsim
baseline = sim['f'](state, visualize=False)[0]
logging.info("baseline sim: %g", baseline)
#baseline = 0.

for order in [0,2]:
    for i in range(len(hk)):
        # nr points
        h = 1. / 2**hk[i]
        pointsPerAxis = np.ceil(1. / h)
        h = 1. / pointsPerAxis
        hs[i] = h
        ppas[i] = pointsPerAxis

        # get similarity
        sim = imsim.get(pointsPerAxis, immname=moving, imfname=fixed, immT=None, border=border, normalize=False, visualize=visualize, order=order, smoothscaleFactor=pointsPerAxis*smoothing, SIGMAF=SIGMAF, h=h, splineS=splineS)
        DIM = tj.DIM = sim['DIM']
        N = tj.N = sim['N']
        logging.info("N points: %d", N)

        # state
        p = np.zeros([N,DIM])
        mu_1 = np.zeros([N,DIM,DIM])
        mu_2 = np.zeros([N,DIM,DIM,DIM])
        (q,q_1,q_2) = (sim['initial'][0], np.outer(np.ones(N),np.eye(DIM)).reshape([N,DIM,DIM]), np.zeros([N,DIM,DIM,DIM]))
        state = tj.weinstein_darboux_to_state(q, q_1, q_2, p, mu_1, mu_2)# initial guess + zeros

        # runsim
        val = sim['f'](state, visualize=False)[0]
        logging.info("sim: %g", val)

        # save result
        res[i] = val
        logging.info("results, order %d: %s", order, res[i])

    logx = -np.log(hs[:-1])
    logy = np.log(np.abs(baseline-res[:-1]))
    convrate = -np.diff(logy)/(np.diff(logx)+1e-8)
    logging.info("convergence rates, order %d: %s", order, convrate)
    logging.info("mean convergence rates, order %d: %s", order, np.mean(convrate[np.isfinite(convrate)]))

    # plot
    plt.figure(20)
    plt.plot(hk,res,colors[order],label='order '+str(order))
    plt.figure(21)
    ax1.plot(logx,logy,colors[order],label='order '+str(order))
    ax2.plot(logx[1:],convrate,colors[order]+'--',label='order '+str(order))
    ax2.set_ylim(-7,7)
    plt.xlim(-np.log(hs[1]),-np.log(hs[-1]))

# ref
L2ref = 1./np.prod(moving_im.shape)*np.sum( moving_im**2 )
#plt.plot(np.arange(len(res)),np.repeat(L2ref,res.shape),'j',label='pixelwise')
logging.info("L2 ref (non-smoothed): %g", L2ref )

# plots
plt.figure(20)
plt.title('Match Term Approximation')
plt.xlabel('$j=1,\ldots,6$ ($h=\mathrm{ceil}(2^{-j})$)')
#plt.xlabel('$j=0,\ldots,5$ ($h=\mathrm{ceil}(2^{-j})$)')
plt.ylabel('measured $L^2$ dissimilarity')
#plt.ylim(0,2)
plt.legend()
plt.gcf().savefig('../results/simtest.eps')

# plots, log
plt.figure(21)
plt.title('Match Term Approximation, Log-log-scale')
ax1.set_xlabel('$-\mathrm{log}(h)$')
ax1.set_ylabel('$\mathrm{log}-L^2$ dissimilarity error')
ax2.set_ylabel('convergence rate (neg. slope of log-error)')
ax1.legend()
plt.show(block=True)
plt.gcf().savefig('../results/simtest-log.eps')
