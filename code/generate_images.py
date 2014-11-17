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

import matplotlib.pyplot as plt
import two_jets as tj
import numpy as np
import kernels.pyGaussian as gaussian

R = np.load('setup.npy')
tj.N = int(R[0])
tj.DIM = int(R[1])
tj.SIGMA = R[2]

tj.gaussian.N = tj.N
tj.gaussian.DIM = tj.DIM
tj.gaussian.SIGMA = tj.SIGMA

DIM = 2

def display_velocity_field( q , p , mu_1 , mu_2 , q1=None ):
	W = 5*tj.SIGMA
	res = 30
	N_nodes = res**DIM
	store = np.outer( np.linspace(-W,W , res), np.ones(res) )
	nodes = np.zeros( [N_nodes , tj.DIM] )
	nodes[:,0] = np.reshape( store , N_nodes )
	nodes[:,1] = np.reshape( store.T , N_nodes )
	K,DK,D2K,D3K,D4K,D5K,D6K = tj.derivatives_of_kernel( nodes , q )
	vel_field = np.einsum('ijab,jb->ia',K,p)\
	    - np.einsum('ijabc,jbc->ia',DK,mu_1)\
	    + np.einsum('ijabcd,jbcd->ia',D2K,mu_2)
	U = vel_field[:,0]
	V = vel_field[:,1]
	f = plt.figure(1)
	plt.quiver( nodes[:,0] , nodes[:,1] , U , V , color='0.50' )
	plt.plot(q[:,0],q[:,1],'ro')

        # generate vertices of a circle
        N_vert = 20
        circle_verts = np.zeros( [ 2 , N_vert + 1 ] )
        theta = np.linspace(0,2*np.pi, N_vert )
        circle_verts[0,0:N_vert] = 0.2*np.cos(theta)
        circle_verts[1,0:N_vert] = 0.2*np.sin(theta)
        verts = np.zeros([2, N_vert + 1])
        units = np.ones( N_vert + 1)

	for i in range(0,len(q)):
		plt.arrow(q[i,0], q[i,1], 0.2*p[i,0], 0.2*p[i,1],\
				  head_width=0.2, head_length=0.2,\
				  fc='b', ec='b')
                if (q1 != None):
                        verts = np.dot(q1[i,:,:], circle_verts ) \
                                + np.outer(q[i,:],units)
                        print np.shape( verts )
                        print np.shape( q1 )
                        plt.plot(verts[0],verts[1],'b-')

        plt.axis([- W, W,- W, W ])
	return f

y_data = np.load('output/state_data.npy')
time_data = np.load('output/time_data.npy')

#print 'shape of y_data is ' + str( y_data.shape )
N_timestep = y_data.shape[0]
print 'generating png files'
for k in range(0,N_timestep):
	q,q_1,q_2,p,mu_1,mu_2 = tj.state_to_weinstein_darboux( y_data[k] )
	f = display_velocity_field(q,p,mu_1,mu_2,q_1)
	time_s = str(time_data[k])
	plt.suptitle('t = '+ time_s[0:4] , fontsize=16 , x = 0.75 , y = 0.25 )
	fname = './movie_frames/frame_'+str(k)+'.png'
	f.savefig( fname )
	plt.close(f)
print 'done'
