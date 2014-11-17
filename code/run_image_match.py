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
Perform a matching of two images using two jets..
"""

import match as match
import matching.imagesim as imsim
import numpy as np

# TODO:
# add  in canonical example
# see examples/*.py for inspiration

print "Weights: " + str(weights)
sim = imsim.get(pointsPerAxis, immname=moving, imfname=fixed, immT=immT, border=border, normalize=True, visualize=True, order=order, smoothscaleFactor=smoothing, SIGMAF=SIGMAF, h=h)
(
print "initial point configuration: " + str(sim['initial'])

(fstate,res) = match.match(sim,sim['SIGMA'],weights,initial=sim['initial'],gradTol=5e-3,order=order)

print res

if True: # res.success:
    print("generating state data for optimization result")
    import two_jets as tj
    tj.DIM = DIM = sim['DIM']
    tj.N = N = sim['N']
    tj.SIGMA = SIGMA = sim['SIGMA']
    
    if order < 1:
        fstate = np.append(fstate, np.outer(np.ones(N),np.eye(DIM)).flatten()) # append mu_1
    if order < 2:
        fstate = np.append(fstate, np.zeros(N*DIM**3)) # append mu_2
    (t_span, y_span) = tj.integrate( fstate )
    
    # save result
    np.save('output/state_data',y_span)
    np.save('output/time_data',t_span)
    np.save('output/setup',[N,DIM,SIGMA])
