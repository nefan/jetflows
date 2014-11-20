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
Perform matching using supplied similarity measure using the two jet particle 
method in two_jets.py. Please look in matching/ for relevant similarity measures, 
e.g. image matching and point matching.
"""

import numpy as np
import two_jets as tj
from scipy.optimize import minimize,fmin_bfgs,fmin_cg,fmin_l_bfgs_b
from scipy.optimize import check_grad
from scipy.optimize import approx_fprime
# from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
import itertools
import logging

def F(sim, nonmoving, x, weights=None, adjint=True, order=2, scalegrad=None, simGradCheck=False, energyGradCheck=False, visualize=False):
    """
    Function that scipy's optimize function will call for returning the value 
    and gradient for a given x. The forward and adjoint integration is called 
    from this function using the values supplied by the similarity measure.
    """

    N = sim['N']
    DIM = sim['DIM']

    i = 0
    q = np.reshape( nonmoving[i:(i+N*DIM)] , [N,DIM] )
    tj.gaussian.N = N
    tj.gaussian.DIM = DIM
    tj.gaussian.SIGMA = tj.SIGMA
    K,DK,D2K,D3K,D4K,D5K,D6K = tj.derivatives_of_kernel(q,q)

    # input
    state0 = np.append(nonmoving, x)
    if order < 1:
        state0 = np.append(state0, np.zeros(N*DIM**2)) # append mu_1
    if order < 2:
        state0 = np.append(state0, np.zeros(N*DIM*tj.triuDim())) # append mu_2
    # shift from triangular to symmetric
    state0 = tj.triangular_to_state(state0)
    triunonmoving = nonmoving
    triux = x
    nonmoving = state0[0:state0.size/2]
    x = state0[state0.size/2:]

    # rescale
    if scalegrad:
        #logging.debug("rescaling, SIGMA " + str(tj.SIGMA))
        q0,q0_1,q0_2,p0,mu0_1,mu0_2 = tj.state_to_weinstein_darboux( state0 )
        if order >= 1:
            mu0_1 = tj.SIGMA*mu0_1
        if order == 2:
            mu0_2 = tj.SIGMA*mu0_2
        state0 = tj.weinstein_darboux_to_state(q0,q0_1,q0_2,p0,mu0_1,mu0_2)

    q0,q0_1,q0_2,p0,mu0_1,mu0_2 = tj.state_to_weinstein_darboux( state0 )

    # flow
    (t_span, y_span) = tj.integrate(state0)
    stateT = y_span[-1]
    
    # debug
    qT,qT_1,qT_2,pT,muT_1,muT_2 = tj.state_to_weinstein_darboux( stateT )
    #logging.info("q0: " + str(q0))
    #logging.info("p0_2: " + str(p0))
    #logging.info("qT: " + str(qT))
    logging.info("||p0||: " + str(np.linalg.norm(p0)))
    logging.info("||mu0_1||: " + str(np.linalg.norm(mu0_1)))
    logging.info("||mu0_2||: " + str(np.linalg.norm(mu0_2)))
    #if order >= 1:
        #logging.info("q0_1: " + str(q0_1))
        #logging.info("qT_1: " + str(qT_1))
        #logging.info("mu0_1: " + str(mu0_1))
    #if order >= 2:
        #logging.info("q0_2: " + str(q0_2))
        #logging.info("qT_2: " + str(qT_2))
        #logging.info("mu0_2: " + str(mu0_2))
    #logging.info("qT-q0: " + str(qT-q0))
    #logging.info("qT_1-q0_1: " + str(qT_1-q0_1))
    #logging.info("qT_2-q0_2: " + str(qT_2-q0_2))

    simT = sim['f'](stateT, state0=state0, visualize=visualize)

    # debug
    #logging.info('match term (before flow/after flow/diff): ' + str(sim['f'](state0)[0]) + '/' + str(simT[0]) + '/' + str(sim['f'](state0)[0]-simT[0]))
    logging.info('match term after flow: ' + str(simT[0]))
    
    Ediff = tj.Hamiltonian(q0,p0,mu0_1,mu0_2) # path energy from Hamiltonian
    logging.info('Hamiltonian: ' + str(Ediff))

    if not adjint:
        return weights[1]*simT[0]+weights[0]*Ediff

    dq = simT[1][0]
    if order >= 1:
        dq_1 = simT[1][1]
    else:
        dq_1 = np.zeros(q0_1.shape)
    if order >= 2:
        dq_2 = simT[1][2]
    else:
        dq_2 = np.zeros(q0_2.shape)

    logging.info("||dq||: " + str(np.linalg.norm(dq)))
    logging.info("||dq_1||: " + str(np.linalg.norm(dq_1)))
    logging.info("||dq_2||: " + str(np.linalg.norm(dq_2)))
    ds1 = tj.weinstein_darboux_to_state(dq,dq_1,dq_2,np.zeros(dq.shape),np.zeros(dq_1.shape),np.zeros(dq_2.shape),N,DIM)

    if simGradCheck:
        logging.info("computing finite difference approximation of sim gradient")
        fsim = lambda x: sim['f'](np.hstack( (x,stateT[x.size:],) ), state0=state0)[0]
        findiffgrad = approx_fprime(stateT[0:N*DIM+N*DIM**2+N*DIM**3],fsim,1e-5)
        compgrad = ds1[0:N*DIM+N*DIM**2+N*DIM**3]
        graderr = np.max(abs(findiffgrad-compgrad))
        logging.debug("sim gradient numerical check error: %e",graderr)
        logging.debug("finite diff gradient: " + str(findiffgrad))
        logging.debug("computed gradient: " + str(compgrad))
        logging.debug("difference: " + str(findiffgrad-compgrad))
    if energyGradCheck:
        logging.info("computing finite difference approximation of energy gradient")
        fsim = lambda x: tj.Hamiltonian(q0,np.reshape(x[0:N*DIM],[N,DIM]),np.reshape(x[N*DIM:N*DIM+N*DIM**2],[N,DIM,DIM]),np.reshape(x[N*DIM+N*DIM**2:N*DIM+N*DIM**2+N*DIM**3],[N,DIM,DIM,DIM]))
        findiffgrad = approx_fprime(np.hstack((p0.flatten(),mu0_1.flatten(),mu0_2.flatten(),)),fsim,1e-7)
        compgrad = tj.grad_Hamiltonian(q0,p0,mu0_1,mu0_2)
        graderr = np.max(abs(findiffgrad-compgrad))
        logging.debug("energy gradient numerical check error: %e",graderr)
        logging.debug("finite diff gradient: " + str(findiffgrad))
        logging.debug("computed gradient: " + str(compgrad))
        logging.debug("difference: " + str(findiffgrad-compgrad))
    
    (t_span, y_span) = tj.adj_integrate(stateT,ds1)
    adjstate0 = y_span[-1]

    assert(nonmoving.size+x.size<=adjstate0.size/2)
    gradE = tj.grad_Hamiltonian(q0,p0,mu0_1,mu0_2)
    assert(adjstate0.size/2-nonmoving.size == gradE.size) # gradE doesn't include point variations currently
    gradE = gradE[0:x.size]
    grad0 = weights[1]*adjstate0[adjstate0.size/2+nonmoving.size:adjstate0.size/2+nonmoving.size+x.size] + weights[0]*gradE # transported gradient + grad of energy

    adjstate0[adjstate0.size/2+nonmoving.size:adjstate0.size/2+nonmoving.size+grad0.size] = grad0
    grad0 = tj.state_to_triangular(adjstate0[adjstate0.size/2:adjstate0.size])[triunonmoving.size:triunonmoving.size+triux.size]

    grad0 = np.ndarray.flatten(grad0)

    # rescale
    if scalegrad:
        if order >= 1:
            grad0[N*DIM:N*DIM+N*DIM**2] = tj.SIGMA*grad0[N*DIM:N*DIM+N*DIM**2]
        if order == 2:
            grad0[N*DIM+N*DIM**2:N*DIM+N*DIM**2+N*DIM**3] = tj.SIGMA*grad0[N*DIM+N*DIM**2:N*DIM+N*DIM**2+N*DIM**3]

    # visualization
    dq0,dq0_1,dq0_2,dp0,dmu0_1,dmu0_2 = tj.state_to_weinstein_darboux( adjstate0[adjstate0.size/2:adjstate0.size],N,DIM )
    #logging.info("dp0: " + str(dp0))
    logging.info("||dp0|| final: " + str(np.linalg.norm(dp0)))
    #logging.info("dmu0_1: " + str(dmu0_1))
    logging.info("||dmu0_1|| final: " + str(np.linalg.norm(dmu0_1)))
    #logging.info("dmu0_2: " + str(dmu0_2))
    logging.info("||dmu0_2|| final: " + str(np.linalg.norm(dmu0_2)))
    #logging.info("adjstate0: " + str(adjstate0))
    #logging.info("grad0: " + str(grad0))
    #plt.figure(0)
    #plt.quiver(q0[:,0],q0[:,1],dp0[:,0],dp0[:,1])

    ## pause
    #raw_input("F: Press ENTER to continue")

    return (weights[1]*simT[0]+weights[0]*Ediff, grad0)

def Fdisp(sim, nonmoving, x, adjint=True, weights=None, order=2, scalegrad=None):
    """
    As F above but using linear displacement instead of integrating the flows. For testing.
    """

    # input
    N = sim['N'] # debug
    DIM = sim['DIM'] # debug
    tj.gaussian.N = N
    tj.gaussian.DIM = DIM
    tj.gaussian.SIGMA = tj.SIGMA
    #x = np.append(x, np.zeros(N*DIM**3)) # debug
    state0 = np.append(nonmoving, x)
    if order < 1:
        state0 = np.append(state0, np.zeros(N*DIM**2)) # append mu_1
    if order < 2:
        state0 = np.append(state0, np.zeros(N*DIM**3)) # append mu_2

    # shift from triangular to symmetric
    #assert(np.allclose(state0,tj.state_to_triangular(tj.triangular_to_state(state0))))
    state0 = tj.triangular_to_state(state0)
    triunonmoving = nonmoving
    triux = x
    nonmoving = state0[0:state0.size/2]
    x = state0[state0.size/2:]

    # rescale
    if scalegrad:
        #logging.debug("rescaling, SIGMA " + str(tj.SIGMA))
        q0,q0_1,q0_2,p0,mu0_1,mu0_2 = tj.state_to_weinstein_darboux( state0 )
        if order >= 1:
            mu0_1 = tj.SIGMA*mu0_1
        if order == 2:
            mu0_2 = tj.SIGMA*mu0_2
        state0 = tj.weinstein_darboux_to_state(q0,q0_1,q0_2,p0,mu0_1,mu0_2)

    q0,q0_1,q0_2,p0,mu0_1,mu0_2 = tj.state_to_weinstein_darboux( state0 )

    # displacement
    qT = q0+p0
    qT_1 = q0_1+mu0_1
    qT_2 = q0_2+mu0_2
    stateT = tj.weinstein_darboux_to_state(qT,qT_1,qT_2,p0,mu0_1,mu0_2)
    
    # debug
    #qT,qT_1,qT_2,pT,muT_1,muT_2 = tj.state_to_weinstein_darboux( stateT )
    #logging.info("q0_1: " + str(q0_1))
    #logging.info("q0_2: " + str(q0_2))
    #logging.info("p0: " + str(p0))
    #logging.info("mu0_1: " + str(mu0_1))
    #logging.info("mu0_2: " + str(mu0_2))
    #logging.info("qT: " + str(qT))
    #logging.info("qT_1: " + str(qT_1))
    #logging.info("qT_2: " + str(qT_2))
    #logging.info("qT-q0: " + str(qT-q0))
    #logging.info("qT_1-q0_1: " + str(qT_1-q0_1))
    #logging.info("qT_2-q0_2: " + str(qT_2-q0_2))
    logging.info("||p0||: " + str(np.linalg.norm(p0)))
    logging.info("||mu0_1||: " + str(np.linalg.norm(mu0_1)))
    logging.info("||mu0_2||: " + str(np.linalg.norm(mu0_2)))

    simT = sim['f'](stateT)

    # debug
    logging.info('match term (after flow): ' + str(simT[0]))
    
    Ediff = tj.Hamiltonian(q0,p0,mu0_1,mu0_2) # path energy from Hamiltonian
    logging.info('Hamiltonian: ' + str(Ediff))

    if not adjint:
        return weights[1]*simT[0]+weights[0]*Ediff

    dq = simT[1][0]
    if order >= 1:
        dq_1 = simT[1][1]
    else:
        dq_1 = np.zeros([N,DIM,DIM])
    if order >= 2:
        dq_2 = simT[1][2]
    else:
        dq_2 = np.zeros([N,DIM,DIM,DIM])
    ds1 = tj.weinstein_darboux_to_state(np.zeros(dq.shape),np.zeros(dq_1.shape),np.zeros(dq_2.shape),dq,dq_1,dq_2,sim['N'],sim['DIM'])
    
    adjstate0 = np.append(np.zeros(ds1.size), ds1)
    
    assert(nonmoving.size+x.size==adjstate0.size/2)
    gradE = tj.grad_Hamiltonian(q0,p0,mu0_1,mu0_2)
    assert(adjstate0.size/2-nonmoving.size == gradE.size) # gradE doesn't include point variations currently
    grad0 = np.array(weights[1]*adjstate0[adjstate0.size/2+nonmoving.size:] + weights[0]*gradE) # transported gradient + grad of energy

    # get from symmetric to triangular form
    #assert(np.allclose(adjstate0[adjstate0.size/2:adjstate0.size],tj.triangular_to_state(tj.state_to_triangular(adjstate0[adjstate0.size/2:adjstate0.size]))))
    grad0 = tj.state_to_triangular(adjstate0[adjstate0.size/2:adjstate0.size])[triunonmoving.size:triunonmoving.size+triux.size]

    grad0 = np.ndarray.flatten(grad0)

    # rescale
    if scalegrad:
        if order >= 1:
            grad0[N*DIM:N*DIM+N*DIM**2] = tj.SIGMA*grad0[N*DIM:N*DIM+N*DIM**2]
        if order == 2:
            grad0[N*DIM+N*DIM**2:N*DIM+N*DIM**2+N*DIM**3] = tj.SIGMA*grad0[N*DIM+N*DIM**2:N*DIM+N*DIM**2+N*DIM**3]

    # debug
    #grad0 = grad0[0:N*DIM+N*DIM**2]
    logging.info("||dq||: " + str(np.linalg.norm(dq)))
    logging.info("||dq_1||: " + str(np.linalg.norm(dq_1)))
    logging.info("||dq_2||: " + str(np.linalg.norm(dq_2)))
    #logging.info("grad0: " + str(grad0))
    
    ## pause
    #raw_input("Fdisp: Press ENTER to continue")
    
    return (weights[1]*simT[0]+weights[0]*Ediff, grad0)

def match(sim,SIGMA,weights,initial=None,initialMomentum=None,gradTol=None,order=2,scalegrad=True, maxIter=150, visualize=False, visualizeIterations=False):
    """
    Perform matching using the supplied similarity measure, the two_jet flow 
    and scipy's optimizer. A gradient check can be performed for the initial 
    value.

    The initial value is either zero of supplied as a parameter.

    Order specifies the order of the particle jets

    Weights determines the split between energy (weights[0]) and match term (weights[1])
    """

    # set flow parameters
    DIM = tj.DIM = sim['DIM']
    N = tj.N = sim['N']
    tj.SIGMA = SIGMA

    logging.info("Flow parameters: weights %s, order %d, SIGMA %g, scalegrad %s, gradTol %s, maxIter %d, visualize %s, visualizeIterations %s",weights,order,SIGMA,scalegrad,gradTol,maxIter,visualize,visualizeIterations)

    # initial guess (x0moving)
    if not initialMomentum:
        p = np.zeros([N,DIM])
        mu_1 = np.zeros([N,DIM,DIM])
        mu_2 = np.zeros([N,DIM,DIM,DIM])
    else:
        p = initialMomentum[0]
        mu_1 = initialMomentum[1]
        mu_2 = initialMomentum[2]
    if not initial:
        (q,q_1,q_2) = (np.zeros([N,DIM]), np.outer(np.ones(N),np.eye(DIM)).reshape([N,DIM,DIM]), np.zeros([N,DIM,DIM,DIM]))
    elif len(initial) == 1:
        (q,q_1,q_2) = (initial[0], np.outer(np.ones(N),np.eye(DIM)).reshape([N,DIM,DIM]), np.zeros([N,DIM,DIM,DIM]))
    elif len(initial) == 2:
        (q,q_1,q_2) = (initial[0], initial[1], np.zeros([N,DIM,DIM,DIM]))
    elif len(initial) == 3:
        (q,q_1,q_2) = initial
    x0 = tj.weinstein_darboux_to_state(q, q_1, q_2, p, mu_1, mu_2)# initial guess + zeros
    assert(np.allclose(x0,tj.triangular_to_state(tj.state_to_triangular(x0))))
    x0 = tj.state_to_triangular(x0) # we do not need to optimize over symmetric indices

    sizeqs = x0.size/2
    x0nonmoving = x0[0:sizeqs] # for now, the point positions are fixed
    if order == 0:
        x0moving = x0[sizeqs:sizeqs+p.size] # only p is optimized for
    elif order == 1:
        x0moving = x0[sizeqs:sizeqs+p.size+mu_1.size] # only p,mu_1 is optimized for
    else:
        x0moving = x0[sizeqs:x0.size] # p,mu_1,mu_2 is optimized for

    # optimization functions
    fsim = lambda x: F(sim, x0nonmoving, x, weights=weights, order=order, scalegrad=scalegrad, visualize=visualizeIterations)
    f = lambda x: F(sim, x0nonmoving, x, weights=weights, adjint=False, order=order, scalegrad=scalegrad, visualize=visualizeIterations)
    fgrad = lambda x: F(sim, x0nonmoving, x, weights=weights, order=order, scalegrad=scalegrad, visualize=visualizeIterations)[1]

    ## debug
    #fsim = lambda x: Fdisp(sim, x0nonmoving, x, weights=weights, order=order, scalegrad=scalegrad)
    #f = lambda x: Fdisp(sim, x0nonmoving, x, weights=weights, adjint=False, order=order, scalegrad=scalegrad)
    #fgrad = lambda x: Fdisp(sim, x0nonmoving, x, weights=weights, order=order, scalegrad=scalegrad)[1]

    if gradTol != None:
        # non-zero starting point
        res = minimize(fsim, x0moving, method='BFGS', jac=True, options={'disp': True, 'maxiter': 50})
        x0moving = res.x

        # grad check for similarity
        F(sim, x0nonmoving, x0moving, weights=weights, adjint=True, order=order, scalegrad=scalegrad,simGradCheck=True,energyGradCheck=True)

        # grad check full system
        #f = lambda x: F(sim, x0nonmoving, x, weights=[0,1], adjint=False, order=order, scalegrad=False, visualize=visualizeIterations)
        #fgrad = lambda x: F(sim, x0nonmoving, x, weights=[0,1], order=order, scalegrad=False, visualize=visualizeIterations)[1]
        logging.info("computing finite difference approximation of gradient")
        findiffgrad = approx_fprime(x0moving,f,1e-7)
        compgrad = fgrad(x0moving)
        graderr = np.max(abs(findiffgrad-compgrad))
        logging.info("gradient numerical check error: %e",graderr)
        if abs(graderr) > gradTol:
            logging.warning("finite diff gradient: " + str(findiffgrad))
            logging.warning("computed gradient: " + str(compgrad))
            logging.warning("difference: " + str(findiffgrad-compgrad))
            #return None # gradient check not passed

    # change optimization method to e.g. BFGS after debugging
    res = minimize(fsim, x0moving, method='BFGS', jac=True, options={'disp': True, 'maxiter': maxIter})
    #res = fmin_bfgs(f, x0moving, fprime=fgrad, disp=True, full_output=True)
    #res = fmin_cg(f, x0moving, fprime=fgrad, disp=True, full_output=True)
    #res = fmin_l_bfgs_b(f, x0moving, fprime=fgrad, disp=0)

    # visualize result
    if visualize:
        F(sim, x0nonmoving, res.x, weights=weights, adjint=False, order=order, scalegrad=scalegrad, visualize=True)

    return (np.append(x0nonmoving,res.x),res)


def genStateData(fstate, sim):
    logging.info("generating state data for optimization result")

    tj.DIM = DIM = sim['DIM']
    tj.N = N = sim['N']
    tj.SIGMA = SIGMA = sim['SIGMA']
    order = sim['order']
    
    if order < 1:
        fstate = np.append(fstate, np.zeros(N*DIM**2)) # append mu_1
    if order < 2:
        fstate = np.append(fstate, np.zeros(N*DIM*tj.triuDim())) # append mu_2
    fstate = tj.triangular_to_state(fstate)

    (t_span, y_span) = tj.integrate( fstate )
    
    # save result
    np.save('output/state_data',y_span)
    np.save('output/time_data',t_span)
    np.save('output/setup',[N,DIM,SIGMA])
