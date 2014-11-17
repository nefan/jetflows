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

#from scipy.spatial.distance import pdist , squareform
import numpy as np
from scipy.integrate import odeint
import kernels.pyGaussian as gaussian
import multiprocessing as mp
from functools import partial
import ctypes
import time
import itertools
import logging

import __builtin__
try:
    __debug = __builtin__.__debug
except AttributeError:
    __debug = False

N = None
DIM = None
SIGMA = None

parallel = True
pool = None
tic = None
nrProcesses = None

def getN():
    return N

def getDIM():
    return DIM

def get_dim_state():
    return 2*N*(DIM + DIM**2 + DIM**3)

def get_dim_adj_state():
    return 2*dim_state

def derivatives_of_kernel( nodes , q, ompParallel=False ):
    #given x_i and x_j the K = Kernel( x_ij) and derivatives with x_ij = x_i - x_j.
    #The code is written such that we evaluate at the nodes, and entry (i,j) is the contribution at node i due to particle j.
    delta = np.identity( DIM )
    G,DG,D2G,D3G,D4G,D5G,D6G = gaussian.derivatives_of_Gaussians( nodes , q, ompParallel )
    K = np.einsum('ij,ab->ijab',G,delta)
    DK = np.einsum('ijc,ab->ijabc',DG,delta)
    D2K = np.einsum('ijcd,ab->ijabcd',D2G,delta)
    D3K = np.einsum('ijcde,ab->ijabcde',D3G,delta)
    D4K = np.einsum('ijcdef,ab->ijabcdef',D4G,delta)
    D5K = np.einsum('ijcdefg,ab->ijabcdefg',D5G,delta)
    D6K = np.einsum('ijcdefgh,ab->ijabcdefgh',D6G,delta)
    #EXAMPLE OF INDEX CONVENTION 'ijabc' refers to the c^th derivative of the ab^th entry of K(q_i - q_j)
    return K, DK, D2K, D3K , D4K , D5K , D6K

def Hamiltonian( q , p , mu_1 , mu_2 ):
    #returns the Hamiltonian.  Serves as a safety to check our equations of motion are correct.
    K,DK,D2K,D3K,D4K,D5K,D6G = derivatives_of_kernel(q,q)
    term_00 = 0.5*np.einsum('ia,ijab,jb',p,K,p)
    term_01 = - np.einsum('ia,ijabc,jbc',p,DK,mu_1)
    term_11 = -0.5*np.einsum('iad,ijabcd,jbc',mu_1,D2K,mu_1)
    term_02 = np.einsum('ia,jbcd,ijabcd',p,mu_2,D2K)
    term_12 = np.einsum('iae,jbcd,ijabecd',mu_1,mu_2,D3K)
    term_22 = 0.5*np.einsum('iaef,jbcd,ijabcdef',mu_2,mu_2,D4K)
    return term_00 + term_01 + term_11 + term_02 + term_12 + term_22
    
def grad_Hamiltonian( q , p , mu_1 , mu_2 ):
    # the (p,mu_1,mu_2) gradient of the Hamiltonian
    K,DK,D2K,D3K,D4K,D5K,D6G = derivatives_of_kernel(q,q)
    g_p = np.einsum('ijab,jb->ia',K,p) # term_00
    g_p = g_p - np.einsum('ijabc,jbc->ia',DK,mu_1) # term_01
    g_mu_1 = - np.einsum('ia,ijabc->jbc',p,DK) # term_01
    g_mu_1 = g_mu_1 - np.einsum('ijabcd,jbc->iad',D2K,mu_1) # term_11
    g_p = g_p + np.einsum('jbcd,ijabcd->ia',mu_2,D2K) # term_02
    g_mu_2 = np.einsum('ia,ijabcd->jbcd',p,D2K) # term_02
    g_mu_1 = g_mu_1 + np.einsum('jbcd,ijabecd->iae',mu_2,D3K) # term_12
    g_mu_2 = g_mu_2 + np.einsum('iae,ijabecd->jbcd',mu_1,D3K) # term_12
    g_mu_2 = g_mu_2 + np.einsum('jbcd,ijabcdef->iaef',mu_2,D4K) # term_22
    return np.hstack((g_p.flatten(),g_mu_1.flatten(),g_mu_2.flatten()))

def ode_function_single( x , t ):
    state = x[0:get_dim_state()]
    pts = x[get_dim_state():].reshape(-1,DIM)

    q , q_1 , q_2,  p , mu_1 , mu_2 = state_to_weinstein_darboux( state )
    K,DK,D2K,D3K,D4K,D5K,D6K = derivatives_of_kernel( q , q )

    dq = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu_1) + np.einsum('jbcd,ijabcd->ia',mu_2,D2K)
    T00 = -np.einsum('ic,jb,ijcba->ia',p,p,DK)
    T01 = np.einsum('id,jbc,ijdbac->ia',p,mu_1,D2K) - np.einsum('jd,ibc,ijdbac->ia',p,mu_1,D2K)
    T02 = -np.einsum('ie,jbcd,ijebacd->ia',p,mu_2,D3K)-np.einsum('je,ibcd,ijebacd->ia',p,mu_2,D3K)
    T12 = -np.einsum('ife,jbcd,ijfbacde->ia',mu_1,mu_2,D4K)+np.einsum('jfe,ibcd,ijfbacde->ia',mu_1,mu_2,D4K)
    T11 = np.einsum('ied,jbc,ijebacd->ia',mu_1,mu_1,D3K)
    T22 = -np.einsum('izef,jbcd,ijzbafcde->ia',mu_2,mu_2,D5K)
    xi_1 = np.einsum('ijacb,jc->iab',DK,p) \
        - np.einsum('ijadbc,jdc->iab',D2K,mu_1) \
        + np.einsum('jecd,ijaebcd->iab',mu_2,D3K)
    xi_2 = np.einsum('ijadbc,jd->iabc',D2K,p) \
        - np.einsum('ijadebc,jde->iabc',D3K,mu_1) \
        + np.einsum('jefd,ijaebcfd->iabc',mu_2,D4K)
    dq_1 = np.einsum('iac,icb->iab',xi_1,q_1)
    dq_2 = np.einsum('iade,idb,iec->iabc',xi_2,q_1,q_1) + np.einsum('iad,idbc->iabc',xi_1,q_2)
    dp = T00 + T01 + T02 + T12 + T11 + T22
    dmu_1 = np.einsum('iac,ibc->iab',mu_1,xi_1)\
        - np.einsum('icb,ica->iab',mu_1,xi_1)\
        + np.einsum('iadc,ibdc->iab',mu_2,xi_2)\
        - np.einsum('idbc,idac->iab',mu_2,xi_2)\
        - np.einsum('idcb,idca->iab',mu_2,xi_2)
    dmu_2 = np.einsum('iadc,ibd->iabc',mu_2,xi_1)\
        + np.einsum('iabd,icd->iabc',mu_2,xi_1)\
        - np.einsum('idbc,ida->iabc',mu_2,xi_1)
    dstate = weinstein_darboux_to_state( dq , dq_1, dq_2, dp , dmu_1 , dmu_2 )

    # points carried along the flow
    q , q_1 , q_2,  p , mu_1 , mu_2 = state_to_weinstein_darboux( state )
    K,DK,D2K,D3K,D4K,D5K,D6K = derivatives_of_kernel( pts , q )
    dpts = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu_1) + np.einsum('jbcd,ijabcd->ia',mu_2,D2K)

    return np.hstack((dstate,dpts.flatten()))

#tmpq = None

def ode_function_par(state, s):
    #q = np.ctypeslib.as_array(tmpq)

    q , q_1 , q_2,  p , mu_1 , mu_2 = state_to_weinstein_darboux( state )
    K,DK,D2K,D3K,D4K,D5K,D6K = derivatives_of_kernel( q[s,] , q )

    T00 = -np.einsum('ic,jb,ijcba->ia',p[s,],p,DK)
    T01 = np.einsum('id,jbc,ijdbac->ia',p[s,],mu_1,D2K) - np.einsum('jd,ibc,ijdbac->ia',p,mu_1[s,],D2K)
    T02 = -np.einsum('ie,jbcd,ijebacd->ia',p[s,],mu_2,D3K)-np.einsum('je,ibcd,ijebacd->ia',p,mu_2[s,],D3K)
    T12 = -np.einsum('ife,jbcd,ijfbacde->ia',mu_1[s,],mu_2,D4K)+np.einsum('jfe,ibcd,ijfbacde->ia',mu_1,mu_2[s,],D4K)
    T11 = np.einsum('ied,jbc,ijebacd->ia',mu_1[s,],mu_1,D3K)
    T22 = -np.einsum('izef,jbcd,ijzbafcde->ia',mu_2[s,],mu_2,D5K)
    xi_1 = np.einsum('ijacb,jc->iab',DK,p) \
        - np.einsum('ijadbc,jdc->iab',D2K,mu_1) \
        + np.einsum('jecd,ijaebcd->iab',mu_2,D3K)
    xi_2 = np.einsum('ijadbc,jd->iabc',D2K,p) \
        - np.einsum('ijadebc,jde->iabc',D3K,mu_1) \
        + np.einsum('jefd,ijaebcfd->iabc',mu_2,D4K)

    dq = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu_1) + np.einsum('jbcd,ijabcd->ia',mu_2,D2K)
    dq_1 = np.einsum('iac,icb->iab',xi_1,q_1[s,])
    dq_2 = np.einsum('iade,idb,iec->iabc',xi_2,q_1[s,],q_1[s,]) + np.einsum('iad,idbc->iabc',xi_1,q_2[s,])
    dp = T00 + T01 + T02 + T12 + T11 + T22
    dmu_1 = np.einsum('iac,ibc->iab',mu_1[s,],xi_1)\
        - np.einsum('icb,ica->iab',mu_1[s,],xi_1)\
        + np.einsum('iadc,ibdc->iab',mu_2[s,],xi_2)\
        - np.einsum('idbc,idac->iab',mu_2[s,],xi_2)\
        - np.einsum('idcb,idca->iab',mu_2[s,],xi_2)
    dmu_2 = np.einsum('iadc,ibd->iabc',mu_2[s,],xi_1)\
        + np.einsum('iabd,icd->iabc',mu_2[s,],xi_1)\
        - np.einsum('idbc,ida->iabc',mu_2[s,],xi_1)

    return (s, dq, dq_1, dq_2, dp, dmu_1, dmu_2)

def flow_points_par(state, s, pts):
    # points carried along the flow
    q , q_1 , q_2,  p , mu_1 , mu_2 = state_to_weinstein_darboux( state )
    K,DK,D2K,D3K,D4K,D5K,D6K = derivatives_of_kernel( pts , q )
    dpts = np.einsum('ijab,jb->ia',K,p) - np.einsum('ijabc,jbc->ia',DK,mu_1) + np.einsum('jbcd,ijabcd->ia',mu_2,D2K)

    return (s, dpts)

def start_pool():
    assert(parallel)
    global pool
    if not pool:
        global nrProcesses
        nrProcesses = min(mp.cpu_count()/2,N)
        pool = mp.Pool(nrProcesses)
        print "starting pool, " + str(nrProcesses) + " processes (cpu count " + str(mp.cpu_count()) + ", N " + str(N) + ")"

def ode_function( x , t ):
    #global tic
    #if not tic:
    #    print "ode_function, t=" + str(t) + ", time " + str(time.time())
    #else:
    #    print "ode_function, t=" + str(t) + ", time since last " + str(time.time()-tic)
    #tic = time.time()

    state = x[0:get_dim_state()]
    pts = x[get_dim_state():].reshape(-1,DIM)

    # parallel stuff
    start_pool()
    # slices
    slices = np.array_split(np.arange(N),nrProcesses)
    # parallel compute
    results = pool.map(partial(ode_function_par, state), slices)
    # get results
    dq = np.empty([N,DIM])
    dq_1 = np.empty([N,DIM,DIM])
    dq_2 = np.empty([N,DIM,DIM,DIM])
    dp = np.empty([N,DIM])
    dmu_1 = np.empty([N,DIM,DIM])
    dmu_2 = np.empty([N,DIM,DIM,DIM])
    for r in results:
        #res = r.get()
        #print "result: " + str(r[0])
        dq[r[0],] = r[1]
        dq_1[r[0],] = r[2]
        dq_2[r[0],] = r[3]
        dp[r[0],] = r[4]
        dmu_1[r[0],] = r[5]
        dmu_2[r[0],] = r[6]

    dstate = weinstein_darboux_to_state( dq , dq_1, dq_2, dp , dmu_1 , dmu_2 )

    # points carried along the flow
    # slices
    Npts = pts.shape[0]
    dpts = np.empty([Npts,DIM])
    if Npts > 0:
        slices = np.array_split(np.arange(Npts),nrProcesses)
        # parallel compute
        results = [pool.apply_async(flow_points_par, (state, s, pts[s,])) for s in slices]
        # get results
        for r in results:
            r = r.get()
            #print "result: " + str(r[0])
            dpts[r[0],] = r[1]

    # result
    res = np.hstack((dstate,dpts.flatten()))

    # debug
    if __debug:
        res_single = ode_function_single(x,t)
        assert(np.allclose(res,res_single))
        #logging.debug("ode_fun parallel error: " + str(np.linalg.norm(res-res_single)))

    return res

def state_to_weinstein_darboux( state, N=None, DIM=None ):
    if N == None: N = getN()
    if DIM == None: DIM = getDIM()

    i = 0
    q = np.reshape( state[i:(i+N*DIM)] , [N,DIM] )
    i = i + N*DIM
    q_1 = np.reshape( state[i:(i+N*DIM*DIM)] , [N,DIM,DIM] )
    i = i + N*DIM*DIM
    q_2 = np.reshape( state[i:(i+N*DIM*DIM*DIM)] , [N,DIM,DIM,DIM] )
    i = i + N*DIM*DIM*DIM
    p = np.reshape( state[i:(i+N*DIM)] , [N,DIM] )
    i = i + N*DIM
    mu_1 = np.reshape( state[i:(i + N*DIM*DIM)] , [N,DIM,DIM] )
    i = i + N*DIM*DIM
    mu_2 = np.reshape( state[i:(i + N*DIM*DIM*DIM)] ,[N,DIM,DIM,DIM] ) 
    return q , q_1 , q_2 , p , mu_1 , mu_2

def weinstein_darboux_to_state( q , q_1, q_2, p , mu_1, mu_2 , N=None , DIM=None ):
    if N == None: N = getN()
    if DIM == None: DIM = getDIM()

    state = np.zeros( 2* (N*DIM + N*DIM*DIM + N*DIM*DIM*DIM) )
    i = 0
    state[i:(i+N*DIM)] = np.reshape( q , N*DIM )
    i = i + N*DIM 
    state[i:(i+N*DIM*DIM)] = np.reshape( q_1 , N*DIM*DIM )
    i = i + N*DIM*DIM 
    state[i:(i+N*DIM*DIM*DIM)] = np.reshape( q_2 , N*DIM*DIM*DIM )
    i = i + N*DIM*DIM*DIM
    state[i:(i + N*DIM)] = np.reshape( p , N*DIM )
    i = i + N*DIM
    state[i:(i+N*DIM*DIM)] = np.reshape( mu_1 , N*DIM*DIM)
    i = i + N*DIM*DIM
    state[i:(i+N*DIM*DIM*DIM)] = np.reshape( mu_2 , N*DIM*DIM*DIM ) 
    return state

def triuDim(DIM=None):
    if DIM == None: DIM = getDIM()

    return np.triu_indices(DIM)[0].size

def state_to_triangular(state, N=None, DIM=None):
    # remove superfluous entries arising from symmetri in 2nd order indices
    if N == None: N = getN()
    if DIM == None: DIM = getDIM()

    triuind = np.triu_indices(DIM)

    Mq = state[N*DIM+N*DIM**2:N*DIM+N*DIM**2+N*DIM**3].copy().reshape([N,DIM,DIM,DIM])
    Mmu = state[2*N*DIM+2*N*DIM**2+N*DIM**3:2*N*DIM+2*N*DIM**2+2*N*DIM**3].copy().reshape([N,DIM,DIM,DIM])
    for i,a in itertools.product(range(N),range(DIM)):
        Mq[i,a,:,:] = .5*(Mq[i,a,:,:]+Mq[i,a,:,:].T)
        Mmu[i,a,:,:] = .5*(Mmu[i,a,:,:]+Mmu[i,a,:,:].T)

    triuMq = Mq[:,:,triuind[0],triuind[1]]
    triuMmu = Mmu[:,:,triuind[0],triuind[1]]

    statetriu = np.hstack( (state[0:N*DIM+N*DIM**2],triuMq.flatten(),state[N*DIM+N*DIM**2+N*DIM**3:2*N*DIM+2*N*DIM**2+N*DIM**3],triuMmu.flatten(),) )
    assert(statetriu.size == 2*N*DIM+2*N*DIM**2+2*N*DIM*triuDim())

    return statetriu

def triangular_to_state(statetriu, N=None, DIM=None):
    # restore superfluous entries arising from symmetri in 2nd order indices
    if N == None: N = getN()
    if DIM == None: DIM = getDIM()

    triuind = np.triu_indices(DIM)

    triuMq = statetriu[N*DIM+N*DIM**2:N*DIM+N*DIM**2+N*DIM*triuDim()].reshape([N,DIM,triuDim()])

    triuMmu = statetriu[2*N*DIM+2*N*DIM**2+N*DIM*triuDim():2*N*DIM+2*N*DIM**2+2*N*DIM*triuDim()].reshape([N,DIM,triuDim()])

    Mq = np.zeros([N,DIM,DIM,DIM])
    Mmu = np.zeros([N,DIM,DIM,DIM])
    for i,a in itertools.product(range(N),range(DIM)):
        Miaq = np.zeros([DIM,DIM])
        Miaq[triuind[0],triuind[1]] = triuMq[i,a,:]
        Mq[i,a,:,:] = .5*(Miaq+Miaq.T)

        Miamu = np.zeros([DIM,DIM])
        Miamu[triuind[0],triuind[1]] = triuMmu[i,a,:]
        Mmu[i,a,:,:] = .5*(Miamu+Miamu.T)

    state = np.hstack( (statetriu[0:N*DIM+N*DIM**2],Mq.flatten(),statetriu[N*DIM+N*DIM**2+N*DIM*triuDim():2*N*DIM+2*N*DIM**2+N*DIM*triuDim()],Mmu.flatten(),) )
    assert(state.size == 2*N*DIM+2*N*DIM**2+2*N*DIM**3)

    return state


def adj_ode_single( adj_state, t ):
    #computes the forward time adjoint equations d\lambda/dt = -lambda^T M
    #adj_state is an array of dimension 2*dim(state) = 2*( 2*N*(DIM + DIM**2 + DIM**3) )
    dim_state = get_dim_state()
    q,q_1,q_2,p,mu_1,mu_2 = state_to_weinstein_darboux( adj_state[0:dim_state])
    Lq , Lq_1, Lq_2, Lp, Lmu_1, Lmu_2 = state_to_weinstein_darboux( adj_state[dim_state:(2*dim_state)])
    d_adj_state = np.zeros( adj_state.shape )
    K,DK,D2K,D3K,D4K,D5K,D6K = derivatives_of_kernel(q,q)

    delta = np.eye(DIM)
    delta_ij = np.eye(N)
    #Ratios of delta_xi_1
    xi_1_over_q = np.einsum('kd,jkadbc,ij->ijabc',p,D2K,delta_ij)\
        - np.einsum('ked,jkaebcd,ij->ijabc',mu_1,D3K,delta_ij)\
        + np.einsum('kfed,jkafbcde,ij->ijabc',mu_2,D4K,delta_ij)\
        - np.einsum('jd,ijadbc->ijabc',p,D2K)\
        + np.einsum('jed,ijaebcd->ijabc',mu_1,D3K)\
        - np.einsum('jfed,ijafbcde->ijabc',mu_2,D4K)
    xi_1_over_p = np.einsum('ijcab->ijabc',DK)
    xi_1_over_mu_1 = np.einsum('ijcadb->ijabcd',-D2K)
    xi_1_over_mu_2 = np.einsum('ijcaedb->ijabcde',D3K)

    #Ratios of delta_xi_2
    xi_2_over_q = np.einsum('ke,jkaebcd,ij->ijabcd',p,D3K,delta_ij)\
        - np.einsum('kfe,jkafbcde,ij->ijabcd',mu_1,D4K,delta_ij)\
        + np.einsum('klef,jkalbcdef,ij->ijabcd',mu_2,D5K,delta_ij)\
        - np.einsum('je,ijaebcd->ijabcd',p,D3K)\
        + np.einsum('jfe,ijafbcde->ijabcd',mu_1,D4K)\
        - np.einsum('jlef,ijalbcdef->ijabcd',mu_2,D5K)
    xi_2_over_p = np.einsum('ijadbc->ijabcd',D2K)
    xi_2_over_mu_1 = np.einsum('ijadbce->ijabcde',-D3K)
    xi_2_over_mu_2 = np.einsum('ijadbcef->ijabcdef',D4K)

    #T over q formulas
    T00_over_q = - np.einsum('jc,kd,jkcdab,ij->ijab',p,p,D2K,delta_ij)\
        + np.einsum('ic,jd,ijcdab->ijab',p,p,D2K)
    T01_over_q = np.einsum('ij,jd,kec,jkdeabc->ijab',delta_ij,p,mu_1,D3K)\
        - np.einsum('ij,kd,jec,jkdeabc->ijab',delta_ij,p,mu_1,D3K)\
        - np.einsum('id,jec,ijdebca->ijab',p,mu_1,D3K)\
        + np.einsum('jd,iec,ijdebca->ijab',p,mu_1,D3K)
    T02_over_q = np.einsum('ie,jfcd,ijefcdba->ijab',p,mu_2,D4K)\
        + np.einsum('je,ifcd,ijefcdba->ijab',p,mu_2,D4K)\
        - np.einsum('ij,je,kfcd,jkefcdba->ijab',delta_ij,p,mu_2,D4K)\
        - np.einsum('ij,ke,jfcd,jkefcdba->ijab',delta_ij,p,mu_2,D4K)
    T12_over_q = np.einsum('ij,kfe,jlcd,jkflbecda->ijab',delta_ij,mu_1,mu_2,D5K)\
        - np.einsum('ij,jfe,klcd,jkflbecda->ijab',delta_ij,mu_1,mu_2,D5K)\
        - np.einsum('jfe,ilcd,ijflbecda->ijab',mu_1,mu_2,D5K)\
        + np.einsum('ife,jlcd,ijflbecda->ijab',mu_1,mu_2,D5K)
    T11_over_q = np.einsum('ij,jed,kfc,jkefbcda->ijab',delta_ij,mu_1,mu_1,D4K)\
        - np.einsum('ied,jfc,ijefbcda->ijab',mu_1,mu_1,D4K)
    T22_over_q = -np.einsum('ij,jlef,kzcd,jklzbedcfa->ijab',delta_ij,mu_2,mu_2,D6K)\
        + np.einsum('ilef,jzcd,ijlzbedcfa->ijab',mu_2,mu_2,D6K)
    
    #delta T over p formulas
    T00_over_p = - np.einsum('ij,kc,jkbca->ijab',delta_ij,p,DK)\
        - np.einsum('ic,ijcba->ijab',p,DK)
    T01_over_p = - np.einsum('idc,ijbdca->ijab',mu_1,D2K)\
        + np.einsum('ij,kdc,jkbdca->ijab',delta_ij,mu_1,D2K)
    T02_over_p = - np.einsum('ij,kecd,jkbecda->ijab',delta_ij,mu_2,D3K)\
        - np.einsum('iecd,ijbecda->ijab',mu_2,D3K)
    
    # delta T over mu_1 formulas
    T01_over_mu_1 = - np.einsum('ij,kd,jkdbca->ijabc',delta_ij,p,D2K)\
        + np.einsum('id,ijdbca->ijabc',p,D2K)
    T11_over_mu_1 = np.einsum('ij,ked,jkbecda->ijabc',delta_ij,mu_1,D3K)\
        + np.einsum('ied,ijebdca->ijabc',mu_1,D3K)
    T12_over_mu_1 = - np.einsum('ij,kfed,jkbfceda->ijabc',delta_ij,mu_2,D4K)\
        + np.einsum('ifed,ijbfceda->ijabc',mu_2,D4K)

    # delta T over mu_2 formulas
    T02_over_mu_2 = - np.einsum('ij,kecd,jkbecda->ijabcd',delta_ij,mu_2,D3K)\
        - np.einsum('iecd,ijbecda->ijabcd',mu_2,D3K)
    T12_over_mu_2 = np.einsum('ij,kfe,jkfbecda->ijabcd',delta_ij,mu_1,D4K)\
        - np.einsum('ife,ijfbecda->ijabcd',mu_1,D4K)
    T22_over_mu_2 = - np.einsum('ij,kzef,jkbzcfeda->ijabcd',delta_ij,mu_2,D5K)\
        - np.einsum('izef,ijzbedcfa->ijabcd',mu_2,D5K)

    #delta p over [blank] formulas
    p_over_q = T00_over_q + T01_over_q + T11_over_q + T12_over_q + T02_over_q + T22_over_q
    p_over_p = T00_over_p + T01_over_p + T02_over_p
    p_over_mu_1 = T11_over_mu_1 + T01_over_mu_1 + T12_over_mu_1
    p_over_mu_2 = T22_over_mu_2 + T02_over_mu_2 + T12_over_mu_2

    #delta q over blank formulas
    q_over_q = np.einsum('kc,jkacb,ij->ijab',p,DK,delta_ij) \
        - np.einsum('kdc,jkadcb,ij->ijab',mu_1,D2K,delta_ij) \
        + np.einsum('kecd,jkaecdb,ij->ijab',mu_2,D3K,delta_ij)\
        - np.einsum('jc,ijacb->ijab',p,DK) \
        + np.einsum('jdc,ijadcb->ijab',mu_1,D2K) \
        - np.einsum('jecd,ijaecdb->ijab',mu_2,D3K)
    q_over_p = K
    q_over_mu_1 = -DK
    q_over_mu_2 = D2K
    
    #delta q_1 over blank formulas
    xi_1 = np.einsum('ijacb,jc->iab',DK,p) \
        - np.einsum('ijadbc,jdc->iab',D2K,mu_1) \
        + np.einsum('jecd,ijaebcd->iab',mu_2,D3K)
    xi_2 = np.einsum('ijadbc,jd->iabc',D2K,p) \
        - np.einsum('ijadebc,jde->iabc',D3K,mu_1) \
        + np.einsum('jefd,ijeabcfd->iabc',mu_2,D4K)
    q_1_over_q = np.einsum('ijadc,idb->ijabc',xi_1_over_q,q_1)
    q_1_over_q_1 = np.einsum('iac,db,ij->ijabcd',xi_1,delta,delta_ij)
    q_1_over_p = np.einsum('ijadc,idb->ijabc',xi_1_over_p,q_1)
    q_1_over_mu_1 = np.einsum('ijaecd,ieb->ijabcd',xi_1_over_mu_1,q_1)
    q_1_over_mu_2 = np.einsum('ijafcde,ifb->ijabcde',xi_1_over_mu_2,q_1)

    #delta q_2 over blank formulas
    q_2_over_q = np.einsum('ijafed,ifb,iec->ijabcd',xi_2_over_q,q_1,q_1) \
        + np.einsum('ijaed,iebc->ijabcd',xi_1_over_q,q_2)
    q_2_over_q_1 = np.einsum('iadf,ifc,eb,ij->ijabcde',xi_2,q_1,delta,delta_ij) \
        + np.einsum('iadf,ifb,ec,ij->ijabcde',xi_2,q_1,delta,delta_ij)
    q_2_over_q_2 = np.einsum('iad,eb,fc,ij->ijabcdef',xi_1,delta,delta,delta_ij)
    q_2_over_p = np.einsum('ijafed,ifb,iec->ijabcd',xi_2_over_p,q_1,q_1) \
        + np.einsum('ijaed,iebc->ijabcd',xi_1_over_p,q_2)
    q_2_over_mu_1 = np.einsum('ijafzde,ifb,izc->ijabcde',xi_2_over_mu_1,q_1,q_1) \
        + np.einsum('ijafde,ifbc->ijabcde',xi_1_over_mu_1,q_2)
    q_2_over_mu_2 = np.einsum('ijazldef,izb,ilc->ijabcdef',xi_2_over_mu_2,q_1,q_1) \
        + np.einsum('ijazdef,izbc->ijabcdef',xi_1_over_mu_2,q_2)

    #delta mu_1 over blank formulas
    mu_1_over_q = np.einsum('iad,ijbdc->ijabc',mu_1,xi_1_over_q) \
        - np.einsum('idb,ijdac->ijabc',mu_1,xi_1_over_q) \
        + np.einsum('iade,ijbdec->ijabc',mu_2,xi_2_over_q) \
        - np.einsum('idbe,ijdaec->ijabc',mu_2,xi_2_over_q) \
        - np.einsum('ideb,ijdaec->ijabc',mu_2,xi_2_over_q)
    mu_1_over_p = np.einsum('iad,ijbdc->ijabc',mu_1,xi_1_over_p) \
        - np.einsum('idb,ijdac->ijabc',mu_1,xi_1_over_p) \
        + np.einsum('iade,ijbdec->ijabc',mu_2,xi_2_over_p) \
        - np.einsum('idbe,ijdaec->ijabc',mu_2,xi_2_over_p) \
        - np.einsum('ideb,ijdaec->ijabc',mu_2,xi_2_over_p)
    mu_1_over_mu_1 = np.einsum('ij,ac,ibd->ijabcd',delta_ij,delta,xi_1)\
        + np.einsum('iae,ijbecd->ijabcd',mu_1,xi_1_over_mu_1) \
        - np.einsum('ij,db,ica->ijabcd',delta_ij,delta,xi_1) \
        - np.einsum('ieb,ijeacd->ijabcd',mu_1,xi_1_over_mu_1) \
        + np.einsum('iaef,ijbefcd->ijabcd',mu_2,xi_2_over_mu_1) \
        - np.einsum('ifbe,ijfaecd->ijabcd',mu_2,xi_2_over_mu_1) \
        - np.einsum('ifeb,ijfaecd->ijabcd',mu_2,xi_2_over_mu_1)
    mu_1_over_mu_2 = np.einsum('iaf,ijbfcde->ijabcde',mu_1,xi_1_over_mu_2) \
        - np.einsum('ifb,ijfacde->ijabcde',mu_1,xi_1_over_mu_2) \
        + np.einsum('ij,ac,ibde->ijabcde',delta_ij,delta,xi_2) \
        + np.einsum('iafl,ijbflcde->ijabcde',mu_2,xi_2_over_mu_2) \
        - np.einsum('ij,db,icae->ijabcde',delta_ij,delta,xi_2) \
        - np.einsum('ifbl,ijfalcde->ijabcde',mu_2,xi_2_over_mu_2) \
        - np.einsum('ij,eb,icad->ijabcde',delta_ij,delta,xi_2) \
        - np.einsum('iflb,ijfalcde->ijabcde',mu_2,xi_2_over_mu_2)

    mu_2_over_q = np.einsum('iaec,ijbed->ijabcd',mu_2,xi_1_over_q) \
        + np.einsum('iabe,ijced->ijabcd',mu_2,xi_1_over_q) \
        - np.einsum('iebc,ijead->ijabcd',mu_2,xi_1_over_q)
    mu_2_over_p = np.einsum('iaec,ijbed->ijabcd',mu_2,xi_1_over_p) \
        + np.einsum('iabe,ijced->ijabcd',mu_2,xi_1_over_p) \
        - np.einsum('iebc,ijead->ijabcd',mu_2,xi_1_over_p)
    mu_2_over_mu_1 = np.einsum('iafc,ijbfde->ijabcde',mu_2,xi_1_over_mu_1)\
        + np.einsum('iabf,ijcfde->ijabcde',mu_2,xi_1_over_mu_1)\
        - np.einsum('ifbc,ijfade->ijabcde',mu_2,xi_1_over_mu_1)
    mu_2_over_mu_2 = np.einsum('ij,cf,ad,ibe->ijabcdef',delta_ij,delta,delta,xi_1)\
        + np.einsum('ialc,ijbldef->ijabcdef',mu_2,xi_1_over_mu_2)\
        + np.einsum('ij,da,eb,icf->ijabcdef',delta_ij,delta,delta,xi_1)\
        + np.einsum('iabl,ijcldef->ijabcdef',mu_2,xi_1_over_mu_2)\
        - np.einsum('ij,eb,fc,ida->ijabcdef',delta_ij,delta,delta,xi_1)\
        - np.einsum('ilbc,ijladef->ijabcdef',mu_2,xi_1_over_mu_2)

    dLq = - np.einsum('jb,jiba->ia',Lq,q_over_q)\
        - np.einsum('jbc,jibca->ia',Lq_1,q_1_over_q)\
        - np.einsum('jbcd,jibcda->ia',Lq_2,q_2_over_q)\
        - np.einsum('jb,jiba->ia',Lp,p_over_q)\
        - np.einsum('jbc,jibca->ia',Lmu_1,mu_1_over_q)\
        - np.einsum('jbcd,jibcda->ia',Lmu_2,mu_2_over_q)
    dLq_1 = - np.einsum('jcd,jicdab->iab',Lq_1,q_1_over_q_1)\
        - np.einsum('jcde,jicdeab->iab',Lq_2,q_2_over_q_1)\
#        - np.einsum('jcde,jicdeab->iab',Lmu_2,mu_2_over_q_1)
#        - np.einsum('jcd,jicdab->iab',Lmu_1,mu_1_over_q_1)\
#        - np.einsum('jc,jicab->iab',Lp,p_over_q_1)\
#        - np.einsum('jc,jicab->iab',Lq,q_over_q_1) == 0
    dLq_2 = - np.einsum('jdef,jidefabc->iabc',Lq_2,q_2_over_q_2)\
#        - np.einsum('jdef,jidefabc->iabc',Lmu_2,mu_2_over_q_2)
#        - np.einsum('jde,jideabc->iabc',Lmu_1,mu_1_over_q_2)\
#        - np.einsum('jd,jidabc->iabc',Lp,p_over_q_2)\
#        - np.einsum('jde,jideabc->iabc',Lq_1,q_1_over_q_2)\
#        - np.einsum('jd,jidabc->iabc',Lq,q_over_q_2)\
    dLp = - np.einsum('jb,jiba->ia',Lq,q_over_p)\
        - np.einsum('jbc,jibca->ia',Lq_1,q_1_over_p)\
        - np.einsum('jbcd,jibcda->ia',Lq_2,q_2_over_p)\
        - np.einsum('jb,jiba->ia',Lp,p_over_p)\
        - np.einsum('jbc,jibca->ia',Lmu_1,mu_1_over_p)\
        - np.einsum('jbcd,jibcda->ia',Lmu_2,mu_2_over_p)
    dLmu_1 = - np.einsum('jc,jicab->iab',Lq,q_over_mu_1)\
        - np.einsum('jcd,jicdab->iab',Lq_1,q_1_over_mu_1)\
        - np.einsum('jcde,jicdeab->iab',Lq_2,q_2_over_mu_1)\
        - np.einsum('jc,jicab->iab',Lp,p_over_mu_1)\
        - np.einsum('jcd,jicdab->iab',Lmu_1,mu_1_over_mu_1)\
        - np.einsum('jcde,jicdeab->iab',Lmu_2,mu_2_over_mu_1)
    dLmu_2 = - np.einsum('jd,jidabc->iabc',Lq,q_over_mu_2)\
        - np.einsum('jde,jideabc->iabc',Lq_1,q_1_over_mu_2)\
        - np.einsum('jdef,jidefabc->iabc',Lq_2,q_2_over_mu_2)\
        - np.einsum('jd,jidabc->iabc',Lp,p_over_mu_2)\
        - np.einsum('jde,jideabc->iabc',Lmu_1,mu_1_over_mu_2)\
        - np.einsum('jdef,jidefabc->iabc',Lmu_2,mu_2_over_mu_2)

    d_adj_state[0:dim_state] = ode_function_single( adj_state[0:dim_state] , 0 )
    d_adj_state[dim_state:2*dim_state] = weinstein_darboux_to_state( dLq, dLq_1, dLq_2, dLp, dLmu_1, dLmu_2 )

    return -d_adj_state

def adj_ode_par(adj_state, s):
    #computes the forward time adjoint equations d\lambda/dt = -lambda^T M
    #adj_state is an array of dimension 2*dim(state) = 2*( 2*N*(DIM + DIM**2 + DIM**3) )
    dim_state = get_dim_state()
    q,q_1,q_2,p,mu_1,mu_2 = state_to_weinstein_darboux( adj_state[0:dim_state])
    Lq , Lq_1, Lq_2, Lp, Lmu_1, Lmu_2 = state_to_weinstein_darboux( adj_state[dim_state:(2*dim_state)])
    K,DK,D2K,D3K,D4K,D5K,D6K = derivatives_of_kernel(q,q[s,])

    delta = np.eye(DIM)
    delta_ij = np.eye(N)[:,s]
    #Ratios of delta_xi_1
    xi_1_over_q = np.einsum('kd,kjadbc,ij->ijabc',p,D2K,delta_ij)\
        - np.einsum('ked,kjaebcd,ij->ijabc',mu_1,-D3K,delta_ij)\
        + np.einsum('kfed,kjafbcde,ij->ijabc',mu_2,D4K,delta_ij)\
        - np.einsum('jd,ijadbc->ijabc',p[s,],D2K)\
        + np.einsum('jed,ijaebcd->ijabc',mu_1[s,],D3K)\
        - np.einsum('jfed,ijafbcde->ijabc',mu_2[s,],D4K)
    xi_1_over_p = np.einsum('ijcab->ijabc',DK)
    xi_1_over_mu_1 = np.einsum('ijcadb->ijabcd',-D2K)
    xi_1_over_mu_2 = np.einsum('ijcaedb->ijabcde',D3K)

    #Ratios of delta_xi_2
    xi_2_over_q = np.einsum('ke,kjaebcd,ij->ijabcd',p,-D3K,delta_ij)\
        - np.einsum('kfe,kjafbcde,ij->ijabcd',mu_1,D4K,delta_ij)\
        + np.einsum('klef,kjalbcdef,ij->ijabcd',mu_2,-D5K,delta_ij)\
        - np.einsum('je,ijaebcd->ijabcd',p[s,],D3K)\
        + np.einsum('jfe,ijafbcde->ijabcd',mu_1[s,],D4K)\
        - np.einsum('jlef,ijalbcdef->ijabcd',mu_2[s,],D5K)
    xi_2_over_p = np.einsum('ijadbc->ijabcd',D2K)
    xi_2_over_mu_1 = np.einsum('ijadbce->ijabcde',-D3K)
    xi_2_over_mu_2 = np.einsum('ijadbcef->ijabcdef',D4K)

    #T over q formulas
    # comment "x", indices of D?K have been switched
    T00_over_q = ( - np.einsum('jc,kd,kjcdab,ij->ijab',p[s,],p,D2K,delta_ij) # x
        + np.einsum('ic,jd,ijcdab->ijab',p,p[s,],D2K) )
    T01_over_q = (np.einsum('ij,jd,kec,kjdeabc->ijab',delta_ij,p[s,],mu_1,-D3K) # x
        - np.einsum('ij,kd,jec,kjdeabc->ijab',delta_ij,p,mu_1[s,],-D3K)\
        - np.einsum('id,jec,ijdebca->ijab',p,mu_1[s,],D3K)\
        + np.einsum('jd,iec,ijdebca->ijab',p[s,],mu_1,D3K) )
    T02_over_q = (np.einsum('ie,jfcd,ijefcdba->ijab',p,mu_2[s,],D4K)\
        + np.einsum('je,ifcd,ijefcdba->ijab',p[s,],mu_2,D4K)\
        - np.einsum('ij,je,kfcd,kjefcdba->ijab',delta_ij,p[s,],mu_2,D4K) # x
        - np.einsum('ij,ke,jfcd,kjefcdba->ijab',delta_ij,p,mu_2[s,],D4K) ) # x
    T12_over_q = (np.einsum('ij,kfe,jlcd,kjflbecda->ijab',delta_ij,mu_1,mu_2[s,],-D5K)  # x
        - np.einsum('ij,jfe,klcd,kjflbecda->ijab',delta_ij,mu_1[s,],mu_2,-D5K)  # x
        - np.einsum('jfe,ilcd,ijflbecda->ijab',mu_1[s,],mu_2,D5K)\
        + np.einsum('ife,jlcd,ijflbecda->ijab',mu_1,mu_2[s,],D5K) )
    T11_over_q = (np.einsum('ij,jed,kfc,kjefbcda->ijab',delta_ij,mu_1[s,],mu_1,D4K)  # x
        - np.einsum('ied,jfc,ijefbcda->ijab',mu_1,mu_1[s,],D4K) )
    T22_over_q = (-np.einsum('ij,jlef,kzcd,kjlzbedcfa->ijab',delta_ij,mu_2[s,],mu_2,D6K) # x
        + np.einsum('ilef,jzcd,ijlzbedcfa->ijab',mu_2,mu_2[s,],D6K) )

    
    #delta T over p formulas
    T00_over_p = ( - np.einsum('ij,kc,kjbca->ijab',delta_ij,p,-DK) # x
        - np.einsum('ic,ijcba->ijab',p,DK) )
    T01_over_p = - np.einsum('idc,ijbdca->ijab',mu_1,D2K)\
        + np.einsum('ij,kdc,kjbdca->ijab',delta_ij,mu_1,D2K) # x
    T02_over_p = (- np.einsum('ij,kecd,kjbecda->ijab',delta_ij,mu_2,-D3K)  # x
        - np.einsum('iecd,ijbecda->ijab',mu_2,D3K) )
    
    # delta T over mu_1 formulas
    T01_over_mu_1 = ( - np.einsum('ij,kd,kjdbca->ijabc',delta_ij,p,D2K) # x
        + np.einsum('id,ijdbca->ijabc',p,D2K) )
    T11_over_mu_1 = ( np.einsum('ij,ked,kjbecda->ijabc',delta_ij,mu_1,-D3K)  # x
        + np.einsum('ied,ijebdca->ijabc',mu_1,D3K) )
    T12_over_mu_1 = ( - np.einsum('ij,kfed,kjbfceda->ijabc',delta_ij,mu_2,D4K) # x
        + np.einsum('ifed,ijbfceda->ijabc',mu_2,D4K) )

    # delta T over mu_2 formulas
    T02_over_mu_2 = ( - np.einsum('ij,kecd,kjbecda->ijabcd',delta_ij,mu_2,-D3K) # x
        - np.einsum('iecd,ijbecda->ijabcd',mu_2,D3K) )
    T12_over_mu_2 = ( np.einsum('ij,kfe,kjfbecda->ijabcd',delta_ij,mu_1,D4K) # x
        - np.einsum('ife,ijfbecda->ijabcd',mu_1,D4K) )
    T22_over_mu_2 = ( - np.einsum('ij,kzef,kjbzcfeda->ijabcd',delta_ij,mu_2,-D5K) # x
        - np.einsum('izef,ijzbedcfa->ijabcd',mu_2,D5K) )

    #delta p over [blank] formulas
    p_over_q = T00_over_q + T01_over_q + T11_over_q + T12_over_q + T02_over_q + T22_over_q
    p_over_p = T00_over_p + T01_over_p + T02_over_p
    p_over_mu_1 = T11_over_mu_1 + T01_over_mu_1 + T12_over_mu_1
    p_over_mu_2 = T22_over_mu_2 + T02_over_mu_2 + T12_over_mu_2

    #delta q over blank formulas
    q_over_q = np.einsum('kc,kjacb,ij->ijab',p,-DK,delta_ij) \
        - np.einsum('kdc,kjadcb,ij->ijab',mu_1,D2K,delta_ij) \
        + np.einsum('kecd,kjaecdb,ij->ijab',mu_2,-D3K,delta_ij)\
        - np.einsum('jc,ijacb->ijab',p[s,],DK) \
        + np.einsum('jdc,ijadcb->ijab',mu_1[s,],D2K) \
        - np.einsum('jecd,ijaecdb->ijab',mu_2[s,],D3K)
    q_over_p = K
    q_over_mu_1 = -DK
    q_over_mu_2 = D2K
    
    #delta q_1 over blank formulas
    xi_1 = np.zeros([N,DIM,DIM])
    xi_1[s,:,:] = np.einsum('jiacb,jc->iab',-DK,p) \
        - np.einsum('jiadbc,jdc->iab',D2K,mu_1) \
        + np.einsum('jecd,jiaebcd->iab',mu_2,-D3K)
    xi_2 = np.zeros([N,DIM,DIM,DIM])
    xi_2[s,:,:,:] = np.einsum('jiadbc,jd->iabc',D2K,p) \
        - np.einsum('jiadebc,jde->iabc',-D3K,mu_1) \
        + np.einsum('jefd,jieabcfd->iabc',mu_2,D4K)
    q_1_over_q = np.einsum('ijadc,idb->ijabc',xi_1_over_q,q_1)
    q_1_over_q_1 = np.einsum('iac,db,ij->ijabcd',xi_1,delta,delta_ij)
    q_1_over_p = np.einsum('ijadc,idb->ijabc',xi_1_over_p,q_1)
    q_1_over_mu_1 = np.einsum('ijaecd,ieb->ijabcd',xi_1_over_mu_1,q_1)
    q_1_over_mu_2 = np.einsum('ijafcde,ifb->ijabcde',xi_1_over_mu_2,q_1)

    #delta q_2 over blank formulas
    q_2_over_q = np.einsum('ijafed,ifb,iec->ijabcd',xi_2_over_q,q_1,q_1) \
        + np.einsum('ijaed,iebc->ijabcd',xi_1_over_q,q_2)
    q_2_over_q_1 = np.einsum('iadf,ifc,eb,ij->ijabcde',xi_2,q_1,delta,delta_ij) \
        + np.einsum('iadf,ifb,ec,ij->ijabcde',xi_2,q_1,delta,delta_ij)
    q_2_over_q_2 = np.einsum('iad,eb,fc,ij->ijabcdef',xi_1,delta,delta,delta_ij)
    q_2_over_p = np.einsum('ijafed,ifb,iec->ijabcd',xi_2_over_p,q_1,q_1) \
        + np.einsum('ijaed,iebc->ijabcd',xi_1_over_p,q_2)
    q_2_over_mu_1 = np.einsum('ijafzde,ifb,izc->ijabcde',xi_2_over_mu_1,q_1,q_1) \
        + np.einsum('ijafde,ifbc->ijabcde',xi_1_over_mu_1,q_2)
    q_2_over_mu_2 = np.einsum('ijazldef,izb,ilc->ijabcdef',xi_2_over_mu_2,q_1,q_1) \
        + np.einsum('ijazdef,izbc->ijabcdef',xi_1_over_mu_2,q_2)

    #delta mu_1 over blank formulas
    mu_1_over_q = np.einsum('iad,ijbdc->ijabc',mu_1,xi_1_over_q) \
        - np.einsum('idb,ijdac->ijabc',mu_1,xi_1_over_q) \
        + np.einsum('iade,ijbdec->ijabc',mu_2,xi_2_over_q) \
        - np.einsum('idbe,ijdaec->ijabc',mu_2,xi_2_over_q) \
        - np.einsum('ideb,ijdaec->ijabc',mu_2,xi_2_over_q)
    mu_1_over_p = np.einsum('iad,ijbdc->ijabc',mu_1,xi_1_over_p) \
        - np.einsum('idb,ijdac->ijabc',mu_1,xi_1_over_p) \
        + np.einsum('iade,ijbdec->ijabc',mu_2,xi_2_over_p) \
        - np.einsum('idbe,ijdaec->ijabc',mu_2,xi_2_over_p) \
        - np.einsum('ideb,ijdaec->ijabc',mu_2,xi_2_over_p)
    mu_1_over_mu_1 = np.einsum('ij,ac,ibd->ijabcd',delta_ij,delta,xi_1)\
        + np.einsum('iae,ijbecd->ijabcd',mu_1,xi_1_over_mu_1) \
        - np.einsum('ij,db,ica->ijabcd',delta_ij,delta,xi_1) \
        - np.einsum('ieb,ijeacd->ijabcd',mu_1,xi_1_over_mu_1) \
        + np.einsum('iaef,ijbefcd->ijabcd',mu_2,xi_2_over_mu_1) \
        - np.einsum('ifbe,ijfaecd->ijabcd',mu_2,xi_2_over_mu_1) \
        - np.einsum('ifeb,ijfaecd->ijabcd',mu_2,xi_2_over_mu_1)
    mu_1_over_mu_2 = np.einsum('iaf,ijbfcde->ijabcde',mu_1,xi_1_over_mu_2) \
        - np.einsum('ifb,ijfacde->ijabcde',mu_1,xi_1_over_mu_2) \
        + np.einsum('ij,ac,ibde->ijabcde',delta_ij,delta,xi_2) \
        + np.einsum('iafl,ijbflcde->ijabcde',mu_2,xi_2_over_mu_2) \
        - np.einsum('ij,db,icae->ijabcde',delta_ij,delta,xi_2) \
        - np.einsum('ifbl,ijfalcde->ijabcde',mu_2,xi_2_over_mu_2) \
        - np.einsum('ij,eb,icad->ijabcde',delta_ij,delta,xi_2) \
        - np.einsum('iflb,ijfalcde->ijabcde',mu_2,xi_2_over_mu_2)

    mu_2_over_q = np.einsum('iaec,ijbed->ijabcd',mu_2,xi_1_over_q) \
        + np.einsum('iabe,ijced->ijabcd',mu_2,xi_1_over_q) \
        - np.einsum('iebc,ijead->ijabcd',mu_2,xi_1_over_q)
    mu_2_over_p = np.einsum('iaec,ijbed->ijabcd',mu_2,xi_1_over_p) \
        + np.einsum('iabe,ijced->ijabcd',mu_2,xi_1_over_p) \
        - np.einsum('iebc,ijead->ijabcd',mu_2,xi_1_over_p)
    mu_2_over_mu_1 = np.einsum('iafc,ijbfde->ijabcde',mu_2,xi_1_over_mu_1)\
        + np.einsum('iabf,ijcfde->ijabcde',mu_2,xi_1_over_mu_1)\
        - np.einsum('ifbc,ijfade->ijabcde',mu_2,xi_1_over_mu_1)
    mu_2_over_mu_2 = np.einsum('ij,cf,ad,ibe->ijabcdef',delta_ij,delta,delta,xi_1)\
        + np.einsum('ialc,ijbldef->ijabcdef',mu_2,xi_1_over_mu_2)\
        + np.einsum('ij,da,eb,icf->ijabcdef',delta_ij,delta,delta,xi_1)\
        + np.einsum('iabl,ijcldef->ijabcdef',mu_2,xi_1_over_mu_2)\
        - np.einsum('ij,eb,fc,ida->ijabcdef',delta_ij,delta,delta,xi_1)\
        - np.einsum('ilbc,ijladef->ijabcdef',mu_2,xi_1_over_mu_2)

    dLq = - np.einsum('jb,jiba->ia',Lq,q_over_q)\
        - np.einsum('jbc,jibca->ia',Lq_1,q_1_over_q)\
        - np.einsum('jbcd,jibcda->ia',Lq_2,q_2_over_q)\
        - np.einsum('jb,jiba->ia',Lp,p_over_q)\
        - np.einsum('jbc,jibca->ia',Lmu_1,mu_1_over_q)\
        - np.einsum('jbcd,jibcda->ia',Lmu_2,mu_2_over_q)
    dLq_1 = - np.einsum('jcd,jicdab->iab',Lq_1,q_1_over_q_1)\
        - np.einsum('jcde,jicdeab->iab',Lq_2,q_2_over_q_1)\
#        - np.einsum('jcde,jicdeab->iab',Lmu_2,mu_2_over_q_1)
#        - np.einsum('jcd,jicdab->iab',Lmu_1,mu_1_over_q_1)\
#        - np.einsum('jc,jicab->iab',Lp,p_over_q_1)\
#        - np.einsum('jc,jicab->iab',Lq,q_over_q_1) == 0
    dLq_2 = - np.einsum('jdef,jidefabc->iabc',Lq_2,q_2_over_q_2)\
#        - np.einsum('jdef,jidefabc->iabc',Lmu_2,mu_2_over_q_2)
#        - np.einsum('jde,jideabc->iabc',Lmu_1,mu_1_over_q_2)\
#        - np.einsum('jd,jidabc->iabc',Lp,p_over_q_2)\
#        - np.einsum('jde,jideabc->iabc',Lq_1,q_1_over_q_2)\
#        - np.einsum('jd,jidabc->iabc',Lq,q_over_q_2)\
    dLp = - np.einsum('jb,jiba->ia',Lq,q_over_p)\
        - np.einsum('jbc,jibca->ia',Lq_1,q_1_over_p)\
        - np.einsum('jbcd,jibcda->ia',Lq_2,q_2_over_p)\
        - np.einsum('jb,jiba->ia',Lp,p_over_p)\
        - np.einsum('jbc,jibca->ia',Lmu_1,mu_1_over_p)\
        - np.einsum('jbcd,jibcda->ia',Lmu_2,mu_2_over_p)
    dLmu_1 = - np.einsum('jc,jicab->iab',Lq,q_over_mu_1)\
        - np.einsum('jcd,jicdab->iab',Lq_1,q_1_over_mu_1)\
        - np.einsum('jcde,jicdeab->iab',Lq_2,q_2_over_mu_1)\
        - np.einsum('jc,jicab->iab',Lp,p_over_mu_1)\
        - np.einsum('jcd,jicdab->iab',Lmu_1,mu_1_over_mu_1)\
        - np.einsum('jcde,jicdeab->iab',Lmu_2,mu_2_over_mu_1)
    dLmu_2 = - np.einsum('jd,jidabc->iabc',Lq,q_over_mu_2)\
        - np.einsum('jde,jideabc->iabc',Lq_1,q_1_over_mu_2)\
        - np.einsum('jdef,jidefabc->iabc',Lq_2,q_2_over_mu_2)\
        - np.einsum('jd,jidabc->iabc',Lp,p_over_mu_2)\
        - np.einsum('jde,jideabc->iabc',Lmu_1,mu_1_over_mu_2)\
        - np.einsum('jdef,jidefabc->iabc',Lmu_2,mu_2_over_mu_2)

    return (s, dLq, dLq_1, dLq_2, dLp, dLmu_1, dLmu_2)

def adj_ode( adj_state, t ):
    #global tic
    #if not tic:
    #    print "adj_ode, t=" + str(t) + ", time " + str(time.time())
    #else:
    #    print "adj_ode, t=" + str(t) + ", time since last " + str(time.time()-tic)
    #tic = time.time()

    # parallel stuff
    start_pool()

    # slices
    slices = np.array_split(np.arange(N),nrProcesses)

    # parallel compute
    results = pool.map(partial(adj_ode_par, adj_state), slices)

    # get results
    dLq = np.empty([N,DIM])
    dLq_1 = np.empty([N,DIM,DIM])
    dLq_2 = np.empty([N,DIM,DIM,DIM])
    dLp = np.empty([N,DIM])
    dLmu_1 = np.empty([N,DIM,DIM])
    dLmu_2 = np.empty([N,DIM,DIM,DIM])
    for r in results:
        dLq[r[0],] = r[1]
        dLq_1[r[0],] = r[2]
        dLq_2[r[0],] = r[3]
        dLp[r[0],] = r[4]
        dLmu_1[r[0],] = r[5]
        dLmu_2[r[0],] = r[6]

    dim_state = get_dim_state()
    d_adj_state = np.empty( adj_state.shape )
    d_adj_state[0:dim_state] = ode_function( adj_state[0:dim_state] , 0 )
    d_adj_state[dim_state:2*dim_state] = weinstein_darboux_to_state( dLq, dLq_1, dLq_2, dLp, dLmu_1, dLmu_2 )

    # result
    res = -d_adj_state

    # debug
    if __debug:
        res_single = adj_ode_single(adj_state,t)
        assert(np.allclose(res,res_single))
        #logging.debug("adj_ode parallel error: " + str(np.linalg.norm(res-res_single)))

    return res

def symmetrize_mu_2( state ):
    q,q_1,q_2,p,mu_1,mu_2 = state_to_weinstein_darboux( state )
    for i in range(0,N):
        for d in range(0,DIM):
            store = mu_2[i,d]
            mu_2[i,d] = 0.5*(store + store.T)

def test_Gaussians( q ):
    h = 1e-9
    G,DG,D2G,D3G,D4G,D5G,D6G = gaussian.derivatives_of_Gaussians(q,q)
    q_a = np.copy(q)
    q_b = np.copy(q)
    q_c = np.copy(q)
    q_d = np.copy(q)
    q_e = np.copy(q)
    q_f = np.copy(q)
    for i in range(0,N):
        for a in range(0,DIM):
            error_max = 0.
            q_a[i,a] = q[i,a]+h
            G_a , DG_a , D2G_a , D3G_a, D4G_a , D5G_a ,D6G_a = gaussian.derivatives_of_Gaussians(q_a, q) 
            for j in range(0,N):
                error = (G_a[i,j] - G[i,j])/h - DG[i,j,a]
                error_max = np.maximum( np.absolute( error ) , error_max )
            print 'max error for DG was ' + str( error_max )
            error_max = 0.
            for b in range(0,DIM):
                q_b[i,b] = q[i,b] + h
                G_b , DG_b , D2G_b , D3G_b , D4G_b , D5G_b , D6G_b = gaussian.derivatives_of_Gaussians( q_b , q ) 
                for j in range(0,N):
                    error = (DG_b[i,j,a] - DG[i,j,a])/h - D2G[i,j,a,b]
                    error_max = np.maximum( np.absolute( error ) , error_max )
                print 'max error for D2G was ' + str( error_max )
                error_max = 0.
                for c in range(0,DIM):
                    q_c[i,c] = q_c[i,c] + h
                    G_c , DG_c , D2G_c , D3G_c , D4G_c , D5G_c , D6G_c = gaussian.derivatives_of_Gaussians( q_c , q ) 
                    for j in range(0,N):
                        error = (D2G_c[i,j,a,b] - D2G[i,j,a,b])/h - D3G[i,j,a,b,c]
                        error_max = np.maximum( np.absolute(error) , error_max )
                    print 'max error for D3G was ' + str( error_max )
                    error_max = 0.
                    for d in range(0,DIM):
                        q_d[i,d] = q[i,d] + h
                        G_d, DG_d , D2G_d , D3G_d, D4G_d , D5G_d , D6G_d = gaussian.derivatives_of_Gaussians( q_d , q )
                        for j in range(0,N):
                            error = (D3G_d[i,j,a,b,c] - D3G[i,j,a,b,c])/h - D4G[i,j,a,b,c,d]
                            error_max = np.maximum( np.absolute(error) , error_max )
                        print 'max error for D4G was '+ str(error_max)
                        error_max = 0.
                        for e in range(0,DIM):
                            q_e[i,e] = q[i,e] + h
                            G_e, DG_e , D2G_e , D3G_e, D4G_e, D5G_e , D6G_e = gaussian.derivatives_of_Gaussians( q_e , q )
                            for j in range(0,N):
                                error = (D4G_e[i,j,a,b,c,d] - D4G[i,j,a,b,c,d])/h - D5G[i,j,a,b,c,d,e]
                                error_max = np.maximum( np.absolute(error) , error_max )
                            print 'max error for D5G was '+ str(error_max)
                            for f in range(0,DIM):
                                q_f[i,f] = q[i,f] + h
                                G_f, DG_f , D2G_f , D3G_f, D4G_f, D5G_f,D6G_f = gaussian.derivatives_of_Gaussians( q_f , q )
                                for j in range(0,N):
                                    error = (D5G_f[i,j,a,b,c,d,e] - D5G[i,j,a,b,c,d,e])/h - D6G[i,j,a,b,c,d,e,f]
                                    error_max = np.maximum( np.absolute(error) , error_max )
                                    print 'max error for D6G was '+ str(error_max)
                                    error_max = 0.
                                q_f[i,f] = q_f[i,f] - h
                            q_e[i,e] = q_e[i,e] - h
                        q_d[i,d] = q_d[i,d] - h
                    q_c[i,c] = q_c[i,c] - h
                q_b[i,b] = q_b[i,b] - h
            q_a[i,a] = q_a[i,a] - h
    return 1

def test_kernel_functions( q ):
    h = 1e-8
#    G,DG,D2G,D3G,D4G,D5G = gaussian.derivatives_of_Gaussians(q,q)
    q_a = np.copy(q)
    q_b = np.copy(q)
    q_c = np.copy(q)
    q_d = np.copy(q)
    q_e = np.copy(q)
    K,DK,D2K,D3K,D4K,D5K,D6K = derivatives_of_kernel(q,q)
    delta = np.identity(DIM)
    error_max = 0.
    for i in range(0,N):
        for j in range(0,N):
            x = q[i,:] - q[j,:]
            r_sq = np.inner( x , x )
            for a in range(0,DIM):
                for b in range(0,DIM):
                    G = np.exp( -r_sq / (2.*SIGMA**2) )
                    K_ij_ab = G*delta[a,b]
                    error = K_ij_ab - K[i,j,a,b]
                    error_max = np.maximum( np.absolute(error) , error_max )

    print 'error_max for K = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF K APPEARS TO BE INACCURATE'

    error_max = 0.
    for i in range(0,N):
        for a in range(0,DIM):
            q_a[i,a] = q[i,a] + h
            K_a,DK_a,D2K_a,D3K_a,D4K_a,D5K_a,D6K_a = derivatives_of_kernel(q_a,q)
            for j in range(0,N):
                der = ( K_a[i,j,:,:] - K[i,j,:,:] ) / h
                error = np.linalg.norm(  der - DK[i,j,:,:,a] )
                error_max = np.maximum(error, error_max)
            q_a[i,a] = q[i,a]
    print 'error_max for DK = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF DK APPEARS TO BE INACCURATE'

    error_max = 0.
    q_b = np.copy(q)
    for i in range(0,N):
        for a in range(0,DIM):
            for b in range(0,DIM):
                q_b[i,b] = q[i,b] + h
                K_b,DK_b,D2K_b,D3K_b,D4K_b,D5K_b,D6K_b = derivatives_of_kernel(q_b,q)
                for j in range(0,N):
                    der = (DK_b[i,j,:,:,a] - DK[i,j,:,:,a] )/h
                    error = np.linalg.norm( der - D2K[i,j,:,:,a,b] )
                    error_max = np.maximum( error, error_max )
                q_b[i,b] = q[i,b]

    print 'error_max for D2K = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF D2K APPEARS TO BE INACCURATE'

    error_max = 0.
    q_c = np.copy(q)
    for i in range(0,N):
        for a in range(0,DIM):
            for b in range(0,DIM):
                for c in range(0,DIM):
                    q_c[i,c] = q[i,c] + h
                    K_c,DK_c,D2K_c,D3K_c,D4K_c,D5K_c,D6K_c = derivatives_of_kernel(q_c,q)
                    for j in range(0,N):
                        der = (D2K_c[i,j,:,:,a,b] - D2K[i,j,:,:,a,b] )/h
                        error = np.linalg.norm( der - D3K[i,j,:,:,a,b,c] )
                        error_max = np.maximum( error, error_max )
                    q_c[i,c] = q[i,c]

    print 'error_max for D3K = ' + str( error_max )

    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF D3K APPEARS TO BE INACCURATE'

    print 'TESTING SYMMETRIES'
    print 'Is K symmetric with respect to ij?'
    error_max = 0
    for i in range(0,N):
        for j in range(0,N):
            error = np.linalg.norm( K[i,j,:,:] - K[j,i,:,:] )
            error_max = np.maximum( error, error_max )
    print 'max for K_ij - K_ji = ' + str( error_max )

    print 'Is DK anti-symmetric with respect to ij?'
    error_max = 0
    for i in range(0,N):
        for j in range(0,N):
            for a in range(0,DIM):
                error = np.linalg.norm( DK[i,j,:,:,a] + DK[j,i,:,:,a] )
                error_max = np.maximum( error, error_max )
    print 'max for DK_ij + DK_ji = ' + str( error_max )
    return 1


def test_functions( trials ):
    #checks that each function does what it is supposed to
    h = 10e-7
    q = SIGMA*np.random.randn(N,DIM)
    q1 = SIGMA*np.random.randn(N,DIM,DIM)
    q2 = SIGMA*np.random.randn(N,DIM,DIM,DIM)
    p = SIGMA*np.random.randn(N,DIM)
    mu_1 = np.random.randn(N,DIM,DIM)
#    mu_1 = np.zeros([N,DIM,DIM])
#    mu_2 = np.zeros([N,DIM,DIM,DIM])
    mu_2 = np.random.randn(N,DIM,DIM,DIM)
    
    test_Gaussians( q )
    test_kernel_functions( q )

    s = weinstein_darboux_to_state( q , q1 , q2 ,  p , mu_1 , mu_2 )
    ds = ode_function( s , 0 )
    dq,dq1,dq2,dp_coded,dmu_1,dmu_2 = state_to_weinstein_darboux( ds ) 

    print 'a test of the ode:'
    print 'dp_coded =' + str(dp_coded)
    Q = np.copy(q)
    dp_estim = np.zeros([N,DIM])
    for i in range(0,N):
        for a in range(0,DIM):
            Q[i,a] = q[i,a] + h
            dp_estim[i,a] = - ( Hamiltonian(Q,p,mu_1,mu_2) - Hamiltonian(q,p,mu_1,mu_2) ) / h 
            Q[i,a] = Q[i,a] - h
    print 'dp_estim =' + str(dp_estim)
    print 'dp_error =' + str(dp_estim - dp_coded)
    return 1


def integrate(state, N_t=20, T=1., pts=None):
    """
    flow forward integration

    Points pts are carried along the flow without affecting it
    """
    assert(N)
    assert(DIM)
    assert(SIGMA)
    gaussian.N = N
    gaussian.DIM = DIM
    gaussian.SIGMA = SIGMA

    t_span = np.linspace(0. ,T , N_t )
    #print 'forward integration: SIGMA = ' + str(SIGMA) + ', N = ' + str(N) + ', DIM = ' + str(DIM) + ', N_t = ' + str(N_t) + ', T = ' + str(T)

    if parallel:
        odef = ode_function
    else:
        odef = ode_function_single
    
    if pts == None:
        y_span = odeint( odef , state , t_span)
        return (t_span, y_span)
    else:
        y_span = odeint( odef , np.hstack((state,pts.flatten())) , t_span)
        return (t_span, y_span[:,0:get_dim_state()], y_span[:,get_dim_state():])


def adj_integrate(state, dstate, N_t=20, T=1.):
    """
    adjoint backwards transport
    """
    assert(N)
    assert(DIM)
    assert(SIGMA)
    gaussian.N = N
    gaussian.DIM = DIM
    gaussian.SIGMA = SIGMA

    t_span = np.linspace(0. ,T , N_t )
    #print 'adjoint integration: SIGMA = ' + str(SIGMA) + ', N = ' + str(N) + ', DIM = ' + str(DIM) + ', N_t = ' + str(N_t) + ', T = ' + str(T)
    
    if parallel:
        odef = adj_ode
    else:
        odef = adj_ode_single
    
    adjstate = np.append(state,dstate)
    y_span = odeint( odef , adjstate , t_span )

    return (t_span, y_span)


def energy(state):
    q,q_1,q_2,p,mu_1,mu_2 = state_to_weinstein_darboux( state )

    return Hamiltonian(q,p,mu_1,mu_2)

