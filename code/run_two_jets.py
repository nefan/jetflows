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


import two_jets as tj
import numpy as np

DIM = tj.DIM = 2
N = tj.N = 4
SIGMA = tj.SIGMA = 1.0

def d2zip(grid):
    return np.dstack(grid).reshape([-1,2])


q = SIGMA*2*np.random.randn(N,DIM)
#q = SIGMA*2*np.mgrid[-1.5:1.5:np.complex(0,np.sqrt(N)),-1.5:1.5:np.complex(0,np.sqrt(N))] # particles in regular grid
#q = d2zip(q)
q_1 = np.outer(np.ones(N),np.eye(DIM)).reshape([N,DIM,DIM])
q_2 = np.zeros([N,DIM,DIM,DIM])
p = SIGMA*np.random.randn(N,DIM)
mu_1 = SIGMA*np.random.randn(N,DIM,DIM)
mu_2 = np.zeros([N,DIM,DIM,DIM])
for i in range(0,N):
    for d in range(0,DIM):
        store = (SIGMA**2)*np.random.randn(DIM,DIM)
        mu_2[i,d] = 0.5*(store + store.T)

#q = np.array([[-1.0 , 0.0],[1.0,0.0]])
#p = np.zeros([N,DIM])
#mu_1 = np.zeros([N,DIM,DIM])
#mu_2 = np.zeros([N,DIM,DIM,DIM])

#tj.test_functions(0)
(t_span, y_span) = tj.integrate(tj.weinstein_darboux_to_state(q, q_1, q_2, p, mu_1, mu_2 ), T=1.)

print 'initial energy was \n' + str(tj.energy(y_span[0]))
print 'final energy is \n'    + str(tj.energy(y_span[-1]))

# save result
np.save('output/state_data',y_span)
np.save('output/time_data',t_span)
np.save('output/setup',[N,DIM,SIGMA])
