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
import kernels.pyGaussian as gaussian

DIM = tj.DIM = 2
SIGMA = tj.SIGMA = .75

def d2zip(grid):
    return np.dstack(grid).reshape([-1,2])

def plotDeformedGrid(grid,Nx,Ny,state):
    (reggrid,Nx,Ny) = grid
    (_,_,mgridts) = tj.integrate(state0,pts=reggrid)
    mgridT = mgridts[-1:].reshape(-1,DIM)
    pg.plotGrid(mgridT,Nx,Ny)

# two points
N = tj.N = 2
#q = np.array([[-1.0 , 0.0],[1.0,0.0]])
#gaussian.N = N
#gaussian.DIM = DIM
#gaussian.SIGMA = SIGMA
#G,DG,D2G,D3G,D4G,D5G,D6G = gaussian.derivatives_of_Gaussians(q,q)
##mu_1 = np.zeros([N,DIM,DIM])
#mu_1 = np.array([[[0.0 , 1.0],[-1.0,0.0]],[[0.0 , 1.0],[-1.0,0.0]]])
#mu_2 = np.zeros([N,DIM,DIM,DIM])
##p = np.zeros([N,DIM])
#p = np.vstack( 
#        (np.dot(mu_1[1,:,:],DG[0,1,:])/(G[0,1]**2-1)-np.dot(mu_1[0,:,:],DG[1,0,:])*G[0,1]/(G[0,1]**2-1),
#         np.dot(mu_1[0,:,:],DG[1,0,:])/(G[0,1]**2-1)-np.dot(mu_1[1,:,:],DG[0,1,:])*G[1,0]/(G[0,1]**2-1)) 
#        )
q = np.array([[-2.0*SIGMA , 0.0],[2.0*SIGMA,0.0]])
mu_1 = np.zeros([N,DIM,DIM])
#mu_1 = np.array([[[0.0 , 1.0],[-1.0,0.0]],[[0.0 , 1.0],[-1.0,0.0]]])
mu_1 = np.array([[[1.0 , 0.0],[0.0,1.0]],[[-1.0 , 0.0],[0.0,-1.0]]])
mu_2 = np.zeros([N,DIM,DIM,DIM])
p = np.zeros([N,DIM])
name = 'isotropy2'

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
xlim = (-3.0,3.0)
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
#plt.plot(y_span[-1,0:DIM*N:2],y_span[-1,1:DIM*N:2],'rx',markersize=10,markeredgewidth=3)
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
