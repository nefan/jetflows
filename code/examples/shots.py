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
Produce images of simple deformations

"""


import two_jets as tj
import numpy as np
import plotgrid as pg
import matplotlib.pyplot as plt
import os

DIM = tj.DIM = 2
SIGMA = tj.SIGMA = 1.

def d2zip(grid):
    return np.dstack(grid).reshape([-1,2])

def plotDeformedGrid(grid,Nx,Ny,state):
    (reggrid,Nx,Ny) = grid
    (_,_,mgridts) = tj.integrate(state0,pts=reggrid)
    mgridT = mgridts[-1:].reshape(-1,DIM)
    pg.plotGrid(mgridT,Nx,Ny)

## random shots
#N = tj.N = 4
#q = SIGMA*2*np.random.randn(N,DIM)
##q = SIGMA*2*np.mgrid[-1.5:1.5:np.complex(0,np.sqrt(N)),-1.5:1.5:np.complex(0,np.sqrt(N))] # particles in regular grid
##q = d2zip(q)
#p = SIGMA*np.random.randn(N,DIM)
#mu_1 = SIGMA*np.random.randn(N,DIM,DIM)
#mu_2 = np.zeros([N,DIM,DIM,DIM])
#for i in range(0,N):
#    for d in range(0,DIM):
#        store = (SIGMA**2)*np.random.randn(DIM,DIM)
#        mu_2[i,d] = 0.5*(store + store.T)

## two points
#N = tj.N = 2
#q = np.array([[-1.0 , 0.0],[1.0,0.0]])
#p = np.zeros([N,DIM])
#mu_1 = np.zeros([N,DIM,DIM])
#mu_2 = np.zeros([N,DIM,DIM,DIM])

# one point
N = tj.N = 1
q = np.array([[0.0,0.0]])
p = np.zeros([N,DIM])
mu_1 = np.zeros([N,DIM,DIM])
mu_2 = np.zeros([N,DIM,DIM,DIM])

## translation
#name = 'translation'
#p = np.array([[1.0,0.0]])
## affine
#name = 'expansion'
#mu_1 = np.array([[[1.0 , 0.0],[0.0,1.0]]])
#name = 'contraction'
#mu_1 = np.array([[[-1.0 , 0.0],[0.0,-1.0]]])
#name = 'rotation'
#mu_1 = np.array([[[0.0 , 1.0],[-1.0,0.0]]])
#name = 'shear'
#mu_1 = np.array([[[0.0 , 1.0],[0.0,0.0]]])
name = 'stretch'
mu_1 = np.array([[[1.0 , 0.0],[0.0,-1.0]]])
# 2nd order
#name = '2nd-1000'
#mu_2 = np.array([[[[1.0,0.0],[0.0,0.0]] , [[0.0,0.0],[0.0,0.0]]]])
#name = '2nd-0100'
#mu_2 = np.array([[[[0.0,0.0],[1.0,0.0]] , [[0.0,0.0],[0.0,0.0]]]])
#name = '2nd-0001'
#mu_2 = np.array([[[[0.0,0.0],[0.0,1.0]] , [[0.0,0.0],[0.0,0.0]]]])

# post process
mu_1 = SIGMA**1*mu_1
mu_2 = SIGMA**2*mu_2
for d in range(DIM): # make triangular
    mu_2[0,d] = 0.5*(mu_2[0,d] + mu_2[0,d].T)
    print mu_2[0,d]

# default
q_1 = np.outer(np.ones(N),np.eye(DIM)).reshape([N,DIM,DIM])
q_2 = np.zeros([N,DIM,DIM,DIM])

#tj.test_functions(0)
state0 = tj.weinstein_darboux_to_state(q, q_1, q_2, p, mu_1, mu_2 )
(t_span, y_span) = tj.integrate(state0, T=1.)

print 'initial energy was \n' + str(tj.energy(y_span[0]))
print 'final energy is \n'    + str(tj.energy(y_span[-1]))

# plot
save = True
xlim = (-2.5,2.5)
ylim = (-2.5,2.5)
reggrid = pg.getGrid(xlim[0],xlim[1],ylim[0],ylim[1],xpts=40,ypts=40)

(ggrid,gNx,gNy) = reggrid
(_,_,mgridts) = tj.integrate(state0,pts=ggrid)
mgridT = mgridts[-1:].reshape(-1,DIM)

#plt.figure(1)
#pg.plotGrid(*reggrid)
#pg.axis('equal')
plt.figure(2)
pg.plotGrid(mgridT,gNx,gNy,coloring=True)
plt.plot(y_span[:,0:DIM*N:2],y_span[:,1:DIM*N:2],'r-',linewidth=2)
plt.plot(q[:,0],q[:,1],'bo',markersize=10,markeredgewidth=3)
plt.plot(y_span[-1,0:DIM*N:2],y_span[-1,1:DIM*N:2],'rx',markersize=10,markeredgewidth=3)
plt.axis('equal')
plt.xlim(xlim)
plt.ylim(ylim)

plt.show(block=not save)

# save result
np.save('output/state_data',y_span)
np.save('output/time_data',t_span)
np.save('output/setup',[N,DIM,SIGMA])

# save figures
if save:
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig('output/%s-%s.eps' % (name,os.getpid(),) )
