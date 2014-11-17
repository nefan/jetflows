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
Discrete similarity measure for image matching. Built on 2nd order Taylor
expansion of sum of squared distances.
"""

import numpy as np
import two_jets as tj
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import sqrt
from functools import partial
import plotgrid as pg
from scipy import interpolate
from scipy.interpolate import dfitpack
import logging
import os

# order of derivatives
derivs = np.array([(1,0), (0,1)]), np.array([[(2,0), (1,1)], [(1,1), (0,2)]]), np.array([[[(3,0), (2,1)], [(2,1), (1,2)]], [[(2,1), (1,2)], [(1,2), (0,3)]]])

def interp(x, y, z, kind, splineS=None):
    if kind == 'quintic':
        kx = ky = 5
    elif kind == 'cubic':
        kx = ky = 3
    else:
        assert(false)

    if splineS:
        nx, tx, ny, ty, c, fp, ier = dfitpack.regrid_smth(
            x, y, z, None, None, None, None,
            kx=kx, ky=ky, s=splineS)
    else:
        nx, tx, ny, ty, c, fp, ier = dfitpack.regrid_smth(
            x, y, z, None, None, None, None,
            kx=kx, ky=ky)
    tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)],
                kx, ky)

    return tck

def splineIM( im, T=None, kind='quintic', splineS=None):
    x = np.arange(im.shape[0])
    y = np.arange(im.shape[1])
    if T:
        grid = d2zip(np.mgrid[0:im.shape[0],0:im.shape[1]])
        Tgrid = T(grid)
        Tim = ndimage.interpolation.map_coordinates(im,Tgrid.T,order=1);

        #f = interpolate.interp2d(x, y, Tim.flatten(), kind=kind)
        tck = interp(x, y, Tim.flatten(), kind=kind, splineS=splineS)
    else:
        #f = interpolate.interp2d(x, y, im.flatten(), kind=kind)
        tck = interp(x, y, im.flatten(), kind=kind, splineS=splineS)

    return tck

def samplecross(sgridxy, im, dd=(0,0), hscaling=1. ):
    #return flt(ndimage.map_coordinates(im,sgrid,order=3,mode='nearest'))
    return (1./(hscaling**np.sum(dd)))*flt(interpolate.bisplev(sgridxy[0].flatten(),sgridxy[1].flatten(), im, dd[0], dd[1]) )

def sample(sgridxy, im, dd=(0,0), hscaling=1. ):
    xs = sgridxy[0].flatten()
    ys = sgridxy[1].flatten()
    assert(xs.size == ys.size)
    v = np.zeros(xs.size)
    for i in np.arange(v.size):
        v[i] = (1./(hscaling**np.sum(dd)))*flt(interpolate.bisplev(xs[i],ys[i], im, dd[0], dd[1]) )

    #return v.reshape(sgridxy[0].shape)
    return v

def smooth(im, smoothscale):
    return ndimage.gaussian_filter(im, sigma=(smoothscale, smoothscale), order=0) 

def apply_2d_slices(f, a):
    if np.size(np.shape(a)) > 1:
        return np.array([apply_2d_slices(f, a[i]) for i in range(a.shape[0])])
    else:
        return f(a)

def flt(a):
   return a.reshape(a.shape[0:-2]+(-1,))

def rse(a,N):
    return a.reshape(a.shape[0:-1]+(sqrt(N),sqrt(N)))

def d2zip(grid):
    return np.dstack(grid).reshape([-1,2])

def d2unzip(points,N=None):
    if not N:
        N = points.shape[0]
    return (points[:,0].reshape(sqrt(N),sqrt(N)),points[:,1].reshape(sqrt(N),sqrt(N)))

def plotJacobians(q,q_1):
    N = q.shape[0]
    DIM = 2

    scale = 10.

    e1 = scale*np.outer(np.ones(N),np.array([1,0]))
    e2 = scale*np.outer(np.ones(N),np.array([0,1]))
    ps = np.array( [
        q-np.einsum('ikl,il->ik',q_1,e1)+np.einsum('ikl,il->ik',q_1,e2),
        q+np.einsum('ikl,il->ik',q_1,e1)+np.einsum('ikl,il->ik',q_1,e2),
        q+np.einsum('ikl,il->ik',q_1,e1)-np.einsum('ikl,il->ik',q_1,e2),
        q-np.einsum('ikl,il->ik',q_1,e1)-np.einsum('ikl,il->ik',q_1,e2),
        q-np.einsum('ikl,il->ik',q_1,e1)+np.einsum('ikl,il->ik',q_1,e2),
        ] )
    for i in np.arange(N):
        plt.plot(ps[:,i,0],ps[:,i,1],'g')

def imsim( state, N=None, imshape=None, DIM=None, h=None, imms=None, Dimms=None, imf=None, imm=None, simfs=None, sDimfs=None, sgrid=None, visualize=False, state0=None, grid=None, order=None, imgrid=None, hscaling=None, SIGMA=None, imfs=None):
    q,q_1,q_2,p,mu_1,mu_2 = tj.state_to_weinstein_darboux( state,N,DIM )

    sampleq = partial(sample,d2unzip(q,N), hscaling=hscaling)
    simms = sampleq(imms)
    #sDimms = [apply_2d_slices(sampleq, Dimms[i]) for i in range(np.shape(Dimms)[0])]
    sDimms = [apply_2d_slices(partial(sampleq, imms), derivs[i]) for i in range(len(derivs))]

    d = DIM
    delta = np.identity(DIM)
    one = np.ones([DIM])
    one_minus_delta = np.ones([DIM,DIM])-np.eye(DIM) 

    # value
    v0 = simfs-simms
    m0 = (h**d)*np.einsum('i,i',v0,v0)
    if order >= 1:
        v1 = sDimfs[0]-np.einsum('bi,iba->ai',sDimms[0],q_1)
        m1 = (h**(d+2))/12*np.einsum('ai,ai',v1,v1)
    if order >= 2:
        G = sDimfs[1] \
                -np.einsum('dci,idb,ica->abi',sDimms[1],q_1,q_1) \
                -np.einsum('ci,icab->abi',sDimms[0],q_2)
        m2 = (h**(d+2))/12*np.einsum('i,aai->',v0,G) \
                + (h**(d+4))/(5*2**6)*np.einsum('aai,aai->',G,G) \
                + (h**(d+4))/(9*2**6)*np.einsum('ab,aai,bbi->',one_minus_delta,G,G) \
                + (h**(d+4))/(9*2**6)*np.einsum('ab,abi,abi->',one_minus_delta,G,G)

    # debug output
    if order >= 0:
        logging.info("m0: " + str(m0))
    if order >= 1:
        logging.info("m1: " + str(m1))
        #logging.info("sDimfs[0]: " + str(sDimfs[0]))
        #logging.info("moving: " + str(np.einsum('bi,iba->ai',sDimms[0],q_1)))
    if order >= 2:
        logging.info("m2: " + str(m2))
        #logging.info("G: " + str(G))
        #logging.info("sDimfs[1]: " + str(sDimfs[1]))
        #logging.info("moving: " + str(np.einsum('dci,idb,ica->abi',sDimms[1],q_1,q_1)+np.einsum('ci,icab->abi',sDimms[0],q_2)))
    
    # gradient
    # dq0
    g00 = -2*(h**d)*np.einsum('i,ai->ia',v0,sDimms[0])
    dq0 = g00
    if order >= 1:
        g01 = -(h**(d+2))/6*np.einsum('bai,ibe,ec,ci->ia',sDimms[1],q_1,delta,v1)
        dq0 = dq0+g01
    if order >= 2:
        g02 = -(h**(d+2))/12*np.einsum('ai,ddi->ia',sDimms[0],G)
        G1 = -np.einsum('bcai,ibe,icd->deai',sDimms[2],q_1,q_1) \
                -np.einsum('cai,icde->deai',sDimms[1],q_2)
        g03 = (h**(d+2))/12*np.einsum('i,ddai->ia',v0,G1)
        g04 = (h**(d+4))/(5*2**5)*np.einsum('ddi,ddai->ia',G,G1) \
                +(h**(d+4))/(9*2**5)*np.einsum('de,ddi,eeai->ia',one_minus_delta,G,G1) \
                +(h**(d+4))/(9*2**5)*np.einsum('de,dei,deai->ia',one_minus_delta,G,G1)
        dq0 = dq0+g02+g03+g04

    # rescale
    dq0 = hscaling*dq0

    # dq1
    if order >= 1:
        g10 = -(h**(d+2))/6*np.einsum('ai,bi->iab',sDimms[0],v1)
        dq1 = g10
        if order >= 2:
            G2 = -np.einsum('aci,ice,db->deabi',sDimms[1],q_1,delta) \
                    -np.einsum('aci,icd,eb->deabi',sDimms[1],q_1,delta)
            g11 = (h**(d+2))/12*np.einsum('i,ddabi->iab',v0,G2)
            g12 = (h**(d+4))/(5*2**5)*np.einsum('ddi,ddabi->iab',G,G2) \
                    +(h**(d+4))/(9*2**5)*np.einsum('de,ddi,eeabi->iab',one_minus_delta,G,G2) \
                    +(h**(d+4))/(9*2**5)*np.einsum('de,dei,deabi->iab',one_minus_delta,G,G2)
            dq1 = dq1+g11+g12

    # dq2
    if order >= 2:
        G3 = -np.einsum('bd,ce,ai->deabci',delta,delta,sDimms[0])
        g21 = (h**(d+2))/12*np.einsum('i,bc,bcabci->iabc',v0,delta,G3)
        g22 = (h**(d+4))/(5*2**5)*np.einsum('ddi,ddabci->iabc',G,G3) \
                +(h**(d+4))/(9*2**5)*np.einsum('de,ddi,eeabci->iabc',one_minus_delta,G,G3) \
                +(h**(d+4))/(9*2**5)*np.einsum('de,dei,deabci->iabc',one_minus_delta,G,G3)
        dq2 = g21+g22

    # visualization
    if visualize:
        logging.info("iteration visualization output")

        x = np.arange(imshape[0])
        y = np.arange(imshape[1])

        plt.figure(1)
        plt.clf()
        scimms = samplecross((x,y),imms).reshape(imshape)
        cmin = np.min([np.min(simms),np.min(simfs),np.min(scimms)])
        cmax = np.max([np.max(simms),np.max(simfs),np.max(scimms)])
        plt.imshow(scimms.T,vmin=cmin,vmax=cmax)
        plt.plot(d2zip(sgrid)[:,0],d2zip(sgrid)[:,1],'bo')
        plt.plot(q[:,0],q[:,1],'rx')
        plt.gray()
        plt.colorbar()
        plotJacobians(q,q_1)
        #plt.quiver(q[:,0],q[:,1],dq0[:,0],dq0[:,1],color='y')
        plt.xlim(0,imshape[0])
        plt.ylim(0,imshape[1])
        #plt.quiver(q[:,0],q[:,1],g00[:,0],g00[:,1])

        plt.figure(2)
        plt.clf()
        plt.imshow(rse(simfs,N).T,vmin=cmin,vmax=cmax)
        plt.colorbar()
        plt.figure(3)
        plt.clf()
        plt.imshow(rse(simms,N).T,vmin=cmin,vmax=cmax)
        plt.colorbar()

        plt.figure(4)
        plt.clf()
        plt.imshow(rse(v0,N).T)
        plt.colorbar()

        plt.figure(5)
        plt.clf()
        plt.imshow(rse(sDimms[0][0,:],N).T)
        plt.colorbar()

        # grid plot
        if sgrid != None:
            qf = d2zip(sgrid)

            plt.figure(6)
            plt.clf()
            plt.plot(qf[:,0],qf[:,1],'bo')
            plt.plot(q[:,0],q[:,1],'rx')

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
            circle_verts[0,0:N_vert] = SIGMA*np.cos(theta)
            circle_verts[1,0:N_vert] = SIGMA*np.sin(theta)
            verts = np.zeros([2, N_vert + 1])
            units = np.ones( N_vert + 1)

            for i in range(0,len(q)):
                plt.arrow(q[i,0], q[i,1], 0.2*p[i,0], 0.2*p[i,1],\
                        head_width=0.2, head_length=0.2,\
                        fc='b', ec='b')
                if (q_1 != None):
                    verts = np.dot(q_1[i,:,:], circle_verts ) \
                            + np.outer(q[i,:],units)
                    plt.plot(verts[0],verts[1],'r-')

            border = 0.4
            plt.xlim(min(np.vstack((qf,q))[:,0])-border,max(np.vstack((qf,q))[:,0])+border)
            plt.ylim(min(np.vstack((qf,q))[:,1])-border,max(np.vstack((qf,q))[:,1])+border)
            plt.axis('equal')

        # warped images
        if state0 != None and imgrid != None and imf != None and imm != None:
            # fixed image, interpolated
            plt.figure(20)
            plt.clf()
            simf = sample(d2unzip(imgrid),imf,hscaling=hscaling);
            plt.imshow(simf.reshape(sqrt(simf.shape[0]),sqrt(simf.shape[0])).T)
            plt.colorbar()
            # fixed image, interpolated
            plt.figure(21)
            plt.clf()
            simf = sample(d2unzip(imgrid),imfs,hscaling=hscaling);
            plt.imshow(simf.reshape(sqrt(simf.shape[0]),sqrt(simf.shape[0])).T)
            plt.colorbar()

            # moving image, interpolated without transformation
            plt.figure(22)
            plt.clf()
            simf = sample(d2unzip(imgrid),imm,hscaling=hscaling);
            plt.imshow(simf.reshape(sqrt(simf.shape[0]),sqrt(simf.shape[0])).T)
            plt.colorbar()
            # moving image, interpolated without transformation
            plt.figure(23)
            plt.clf()
            simf = sample(d2unzip(imgrid),imms,hscaling=hscaling);
            plt.imshow(simf.reshape(sqrt(simf.shape[0]),sqrt(simf.shape[0])).T)
            plt.colorbar()

            # moving image, interpolated
            plt.figure(24)
            plt.clf()
            (_,_,mimgridts) = tj.integrate(state0,pts=imgrid)
            mimgridT = mimgridts[-1:].reshape(-1,DIM)
            simm = sample(d2unzip(mimgridT),imm,hscaling=hscaling);
            plt.imshow(simm.reshape(sqrt(simm.shape[0]),sqrt(simm.shape[0])).T)
            plt.colorbar()
            # moving image, interpolated
            plt.figure(25)
            plt.clf()
            simm = sample(d2unzip(mimgridT),imms,hscaling=hscaling);
            plt.imshow(simm.reshape(sqrt(simm.shape[0]),sqrt(simm.shape[0])).T)
            plt.colorbar()

        plt.draw()
        #plt.show(block=False)

        # save figures
        for i in plt.get_fignums():
            plt.figure(i)
            try:
                os.mkdir('output/%s' % os.getpid() )
            except:
                None
            plt.savefig('output/%s/figure%d.eps' % (os.getpid(),i) )

    if order == 0:
        return (m0, (dq0, ))
    elif order == 1:
        return (m0+m1, (dq0,dq1))
    else:
        return (m0+m1+m2, (dq0,dq1,dq2))

def get(pointsPerAxis, immname, imfname, immT=None, visualize=False, border=0, normalize=False, order=2, smoothscaleFactor=0.5, SIGMAF=2., h=None, splineS=None, visRes=25j):
    """
    get image similarity measure
    """

    logging.info("Image sim parameters: visualize %s, border %d, normalize %s, order %d, smoothscaleFactor %g, sigmaF %g, splineS %s, visRes %s",visualize,border,normalize,order,smoothscaleFactor,SIGMAF,splineS,visRes)

    # load
    imm = np.double(np.load(immname))
    imf = np.double(np.load(imfname))
    imshape = imf.shape

    Nx = np.shape(imm)[0]-2*border
    Ny = np.shape(imm)[1]-2*border
    assert Nx == Ny
    assert Nx > 0
    N = Nx*Ny
    
    Ns = pointsPerAxis**2 # sample points
    relsmoothscale = smoothscaleFactor/sqrt(Ns)
    if not h:
        h = sqrt(N)/sqrt(Ns) # length scale h, dist h between particles
    logging.info("h: " + str(h))
    smoothscale = relsmoothscale*sqrt(N)
    logging.info("smoothscale: " + str(smoothscale))
    SIGMA = SIGMAF*sqrt(N)/sqrt(Ns)
    logging.info("SIGMA: " + str(SIGMA))
    
    # smooth
    imms = smooth(imm, smoothscale)
    imfs = smooth(imf, smoothscale)

    # normalize
    if normalize:
        imms = imms-np.min(imms)
        if np.max(imms) > 1e-5:
            imms = imms/np.max(imms)
        imfs = imfs-np.min(imfs)
        if np.max(imfs) > 1e-5:
            imfs = imfs/np.max(imfs)
    # grid
    indent = Nx/(2*sqrt(Ns))
    sgrid = np.mgrid[border+indent:border+Nx-indent:complex(0,sqrt(Ns)),border+indent:border+Ny-indent:complex(0,sqrt(Ns))] 
    sgridxy = (sgrid[0,:,0], sgrid[0,0,:])

    # plot
    if visualize:
        # plotting setup
        mpl.rcParams['image.interpolation'] = 'nearest'
        mpl.rcParams['image.origin'] = 'lower'

        # splot befine spline interpolation
        plt.figure(0)
        plt.imshow(imfs.T)
        plt.figure(1)
        plt.imshow(imms.T)
        #plt.show(block=True)

    # spline
    imfs = splineIM(imfs, splineS=splineS)
    imms = splineIM(imms, T=immT, splineS=splineS)
    # original images for visualization
    imf = splineIM(imf, kind='cubic', splineS=splineS)
    imm = splineIM(imm, T=immT, kind='cubic', splineS=splineS)

    # downsample
    samplesgrid = partial(sample,sgrid, hscaling=h*sqrt(Ns)/sqrt(N))
    simfs = samplesgrid(imfs)       
    #sDimfs = [apply_2d_slices(samplesgrid, Dimfs[i]) for i in range(np.shape(Dimfs)[0])]
    sDimfs = [apply_2d_slices(partial(samplesgrid, imfs), derivs[i]) for i in range(len(derivs))]

    f = partial(imsim, N=Ns, imshape=imshape, DIM=2, h=h, imms=imms, simfs=simfs, sDimfs=sDimfs, hscaling=h*sqrt(Ns)/sqrt(N), SIGMA=SIGMA, order=order)

    ## visualization
    reggrid = None
    if visualize:
        plt.figure(1)
        reggrid = pg.getGrid(border,-border+Nx,border,-border+Ny,xpts=40,ypts=40)
        pg.plotGrid(*reggrid)

        # attach grids
        f = partial(f, sgrid=sgrid, grid=reggrid)

        # image grid for warping
        pointsPerAxis = complex(0,visRes)
        imgrid = d2zip(np.mgrid[border:border+Nx:pointsPerAxis,border:border+Ny:pointsPerAxis])
        f = partial(f, imgrid=imgrid, imf=imf, imm=imm, imfs=imfs)

        plt.figure(0)
        plt.clf()
        x = np.arange(imshape[0])
        y = np.arange(imshape[1])
        scimfs = samplecross((x,y),imfs).reshape(imshape)
        cmin = np.min([np.min(simfs),np.min(scimfs)])
        cmax = np.max([np.max(simfs),np.max(scimfs)])
        plt.imshow(scimfs.T,vmin=cmin,vmax=cmax)
        plt.gray()
        plt.colorbar()
        q = d2zip(sgrid)
        plt.plot(q[:,0],q[:,1],'bo')
        plt.xlim(0,imshape[0])
        plt.ylim(0,imshape[1])
        # Jacobians
        DIM = 2
        N = q.shape[0]
        q_1 = np.outer(np.ones(N),np.eye(DIM)).reshape([N,DIM,DIM])
        plotJacobians(q,q_1)

        plt.figure(11)
        plt.clf()
        x = np.arange(imshape[0])
        y = np.arange(imshape[1])
        scimms = samplecross((x,y),imms).reshape(imshape)
        plt.imshow(scimms.T)
        plt.gray()
        plt.colorbar()
        plt.plot(d2zip(sgrid)[:,0],d2zip(sgrid)[:,1],'bo')
        plt.xlim(0,imshape[0])
        plt.ylim(0,imshape[1])

        plt.figure(12)
        plt.clf()
        scDimms = samplecross((x,y),imms,(1,0),hscaling=h*sqrt(Ns)/sqrt(N)).reshape(imshape)
        plt.imshow(scDimms.T)
        plt.gray()
        plt.colorbar()

        plt.figure(13)
        plt.clf()
        scDimms = samplecross((x,y),imms,(2,0),hscaling=h*sqrt(Ns)/sqrt(N)).reshape(imshape)
        plt.imshow(scDimms.T)
        plt.gray()
        plt.colorbar()

    sim = {'f': f, 'N': Ns, 'SIGMA': SIGMA, 'DIM': 2, 'order': order, 'initial': (d2zip(sgrid),)}

    return sim
    
