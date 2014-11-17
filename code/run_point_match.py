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
Perform a matching of point sets using two jets. Points can either be randomly
generated or pre specified.
"""

import __builtin__
__builtin__.__debug = True

import match as match
import matching.pointsim as ptsim
import numpy as np
import logging

#import pdb
#pdb.set_trace()

DIM = 2
N = 2
SIGMA = 1.0
ORDER = 2
WEIGHTS = [0, 1]
WEIGHTS = WEIGHTS/np.sum(WEIGHTS)

logging.basicConfig(level=logging.DEBUG,format="[%(filename)s:%(lineno)s - %(funcName)6s() ] %(message)s")

random = False

# moving points
if random:
    qm = SIGMA*np.random.randn(N,DIM)
    qm_1 = SIGMA*np.random.randn(N,DIM,DIM)
    qm_2 = np.zeros([N,DIM,DIM,DIM])
    for i in range(0,N):
        for d in range(0,DIM):
            store = (SIGMA**2)*np.random.randn(DIM,DIM)
            qm_2[i,d] = 0.5*(store + store.T)
else:
    #qm = 10.*np.array([[-1.0 , 0.0]])
    qm = 1.*np.array([[-1.0 , 0.0],[1.0,0.0]])
    #qm = 1.*np.array([[-1.0 , 0.0],[0.0,0.0],[1.0,0.0]])
    qm_1 = 1.*np.outer(np.ones(N),np.eye(DIM)).reshape([N,DIM,DIM])
    qm_2 = np.zeros([N,DIM,DIM,DIM])

# fixed points
if random:
    qf = SIGMA*np.random.randn(N,DIM)
    qf_1 = SIGMA*np.random.randn(N,DIM,DIM)
    qf_2 = np.zeros([N,DIM,DIM,DIM])
    for i in range(0,N):
        for d in range(0,DIM):
            store = (SIGMA**2)*np.random.randn(DIM,DIM)
            qf_2[i,d] = 0.5*(store + store.T)
else:
    #qf = 10.*np.array([[-1.0 , 1.0]])
    qf = 1.*np.array([[-1.0 , 1.0],[1.0,1.0]])
    #qf = 1.*np.array([[-1.0 , 1.0],[0.0,1.0],[1.0,1.0]])
    qf_1 = 1.*np.outer(np.ones(N),np.eye(DIM)).reshape([N,DIM,DIM])
    qf_2 = np.zeros([N,DIM,DIM,DIM])

if ORDER == 0:
    moving=(qm, )
    fixed=(qf, )
elif ORDER == 1:
    moving=(qm,qm_1)
    fixed=(qf,qf_1)
else:
    moving=(qm,qm_1,qm_2)
    fixed=(qf,qf_1,qf_2)


sim = ptsim.get(fixed, order=ORDER, moving=moving)

(fstate,res) = match.match(sim,SIGMA,WEIGHTS,initial=moving, order=ORDER, gradTol=1e-6, visualize=False)

#print res

if True: # res.success:
    match.genStateData(fstate,sim)
