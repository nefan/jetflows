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
Similarity measure for matching points with first and second order information.
Sum of squared distances applied to all values.
"""

import numpy as np
import two_jets as tj
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy import sqrt
from functools import partial
import plotgrid as pg

def psim( state, N=None, DIM=None, fixed=None, visualize=False, order=2, state0=None, grid=None):
    qm,qm_1,qm_2,pm,mum_1,mum_2 = tj.state_to_weinstein_darboux( state,N,DIM )
    qf = fixed[0]
    if order >= 1:
        qf_1 = fixed[1]
    if order >= 2:
        qf_2 = fixed[2]

    w = [1,.5,.2]; # weighting between different order terms

    # value
    v0 = qm-qf
    m0 = w[0]*np.einsum('ia,ia',v0,v0) # 1./N ??
    if order >= 1:
        v1 = qm_1-qf_1
        m1 = w[1]*np.einsum('iab,iab',v1,v1) # 1./N ?? 
    if order >= 2:
        v2 = qm_2-qf_2
        m2 = w[2]*np.einsum('iabg,iabg',v2,v2) # 1./N ??
    
    # gradient
    dq0 = w[0]*2.*v0 # 1./N ??
    if order >= 1:
        dq1 = w[1]*2.*v1 # 1./N ??
    if order >= 2:
        dq2 = w[2]*2.*v2 # 1./N ??

    #print "point sim: m0 " + str(m0) + ", m1 " + str(m1) + ", m2 " + str(m2)

    ## visualization
    if visualize:
        plt.figure(1)
        plt.clf()
        plt.plot(qf[:,0],qf[:,1],'bo')
        plt.plot(qm[:,0],qm[:,1],'rx')

        # grid
        if state0 != None and grid != None:
            (reggrid,Nx,Ny) = grid
            (_,_,mgridts) = tj.integrate(state0,pts=reggrid)
            mgridT = mgridts[-1:].reshape(-1,DIM)
            pg.plotGrid(mgridT,Nx,Ny)

        # generate vertices of a circle
        N_vert = 20
        circle_verts = np.zeros( [ 2 , N_vert + 1 ] )
        theta = np.linspace(0,2*np.pi, N_vert )
        circle_verts[0,0:N_vert] = 0.2*np.cos(theta)
        circle_verts[1,0:N_vert] = 0.2*np.sin(theta)
        verts = np.zeros([2, N_vert + 1])
        units = np.ones( N_vert + 1)

        for i in range(0,len(qm)):
            plt.arrow(qm[i,0], qm[i,1], 0.2*pm[i,0], 0.2*pm[i,1],\
                    head_width=0.2, head_length=0.2,\
                    fc='b', ec='b')
            if (qm_1 != None):
                verts = np.dot(qm_1[i,:,:], circle_verts ) \
                        + np.outer(qm[i,:],units)
                plt.plot(verts[0],verts[1],'r-')

        border = 0.4
        plt.xlim(min(np.vstack((qf,qm))[:,0])-border,max(np.vstack((qf,qm))[:,0])+border)
        plt.ylim(min(np.vstack((qf,qm))[:,1])-border,max(np.vstack((qf,qm))[:,1])+border)
        plt.axis('equal')
        plt.draw()

    if order == 0:
        return (m0, (dq0, ))
    elif order == 1:
        return (m0+m1, (dq0,dq1))
    else:
        return (m0+m1+m2, (dq0,dq1,dq2))

def get(fixed=None, visualize=False, order=2, moving=None):
    """
    get point SSD similarity 
    """

    # data
    qf = fixed[0]

    N = qf.shape[0]
    DIM = qf.shape[1]
    
    ## visualization
    reggrid = None
    if visualize:
        plt.figure(0)
        plt.clf()
        plt.plot(qf[:,0],qf[:,1],'bo')

        if moving: # moving points needed in order to find grid size
            qm = moving[0]
            reggrid = pg.getGrid(np.vstack((qf,qm))[:,0].min()-1,np.vstack((qf,qm))[:,0].max()+1,np.vstack((qf,qm))[:,1].min()-1,np.vstack((qf,qm))[:,1].max()+1,xpts=10,ypts=10)
            pg.plotGrid(*reggrid)

    f = partial(psim, fixed=fixed, N=N, DIM=DIM, visualize=visualize, order=order, grid=reggrid)
    sim = {'f': f, 'N': N, 'DIM': DIM}

    return sim
    
