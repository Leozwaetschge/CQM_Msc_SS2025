#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from scipy import linalg
import scipy.sparse
from scipy.sparse import linalg

from numpy import transpose as tr, conjugate as co
from scipy.linalg import expm, svd
from scipy.sparse.linalg import eigsh, LinearOperator
import math

"""
Helper functions
===========================
"""

def dot(A,B):
    """ Does the dot product like np.dot, but preserves the shapes also for singleton dimensions """
    s1 = A.shape
    s2 = B.shape
    return np.dot(A,B).reshape((s1[0],s2[1]))

def inverse(S,d):
    """
    Helper function.
    Returns inverse of non-zero part of a diagonal matrix
    
    Parameters
    ----------
    S: array [d2xd2]
       S=np.diag([lambda_1, ...0,..lambda_d,0..]) diagonal, with dimension d2>=d
    d: int
       number of non-zero diagonal elements of S
       
    Returns
    -------
    array [dxd], Sinv=np.diag([1/lambda_1,...1/lambda_d]) with dimension d
    """
    d2=np.shape(S)[0]
    Sinv=np.zeros((d,d))
    for i in range(d2):
        if (S[i]>1e-3):
            Sinv[i,i]=1.0/S[i]
    return Sinv


def random_mps(L, chi):
    # define random MPS with given maximal bond dimension
    s = 2  # local Hilbert space dimension
    shapes = [(s, min(chi, s**i, s**(L-i)), min(chi, s**(i+1), s**(L-1-i))) for i in range(L)]
    M = [np.random.standard_normal(shape)for shape in shapes]
    return M

"""
Canonization
===========================
""" 

def right_canonize(M1,M2,return_S = False):
    """ Right normalizes M2 into B matrix, M1 loses its canonization """
    s, da, db = M2.shape
    U, S, Vh = svd(M2.transpose((1,0,2)).reshape((da,s*db)))
    #this reshapes M2 and finds its svd
    B2     = Vh.reshape((Vh.shape[0],s,db)).transpose((1,0,2))[:,:da,:]
    M1     = np.tensordot(M1,dot(U[:,:min(da,np.shape(S)[0])],np.diag(S[:min(da,np.shape(S)[0])])),axes=((2),(0)))
    if return_S:
        return M1, B2, S
    else:
        return M1, B2


def right_canonize_complete(M):
    """ performs right-canonization and returns right-normalized MPS [B_1, ...B_N]
    
    Parameters
    ----------
    M: list of tensors [M_1, ...M_N]
       where M_i is an array of shape (s,dleft,dright)-> s=2 (physical index), dleft and dright are the bond dimensions
       Matrix Product representation of a given state with N spins
   
    Returns
    -------
    B: list of tensors [B_1, ...B_N]
       right-normalized MPS representation of given state
    """
    N=len(M)
    B=[]
    Mitilde,Bi,Si=right_canonize(M[N-2],M[N-1],True)
    Bi.reshape(np.shape(Bi)[0],np.shape(Bi)[1],1)
    B.insert(0,Bi)
    for i in range(N-2):
        Mitilde,Bi,Si=right_canonize(M[N-3-i],Mitilde,True)
        B.insert(0,Bi)
    _,Bi,Si=right_canonize(np.zeros((np.shape(M[0])[0],1,1)),Mitilde,True)
    B.insert(0,Bi)
    
    return B

    
def left_canonize(M1,M2,return_S = False):
    """ Left normalizes M1 into A matrix, M2 loses its canonization"""
    s, da, db = M1.shape
    U, S, Vh = svd(M1.reshape((s*da,db)))
    A1      = U.reshape((s,da,U.shape[1]))[:,:,:min(db,np.shape(S)[0])]
    M2     = np.tensordot(dot(np.diag(S[:min(db,np.shape(S)[0])]),Vh[:min(db,np.shape(S)[0]),:]),M2,axes=((1),(1)))
    if return_S:
        return A1, M2, S
    else:
        return A1, M2


def canonize_start(B1,B2):
    """
    performs svd on first site
    
    Parameters
    ----------
    B1: array, shape (s,da,db) with da=1
        first tensor from the left of the right-canonized MPS
    B2: array, shape (s,db,dright)
        second tensor from the left of the right-canonized MPS
        
    Returns
    -------
    Gamma_1: array, shape (s,da,db)
         Vidal form tensor on first site
    Btilde_2: array, shape (s,db,dright)
        Tensor to be used as input for the next canonization step
    S: array, shape (min(da,db)) (singular values)
        np.diag(S) corresponds to Lambda_1, Vidal form tensor between site 1 and 2
    """
    s,da,db=np.shape(B1)
    reshapedB=np.reshape(B1,(s*da,db))
    U,S,Vdag=np.linalg.svd(reshapedB,full_matrices=0)
    A2=np.reshape(U,(s,da,U.shape[1]))
    Gamma_1=np.zeros((s,da,db))
    Gamma_1[:,:,:U.shape[1]]=A2
    Btilde_2=np.tensordot(np.dot(np.diag(S),Vdag),B2,axes=(1,1))
    Btilde_2=np.transpose(Btilde_2,(1,0,2)) 
    return Gamma_1,Btilde_2,S

def canonize_step(Btilde_i,S1,B_ip1):
    """
    performs svd on site i
    
    Parameters
    ----------
    Btilde_i: array, shape (s,da,db) 
        tensor on site i obtained from svd of the last step
    S1: array, 
        singular values obtained in previous step
    B_ip1: array, shape (s,db,dright)
        tensor on site i+1 of the right-canonized MPS
        
    Returns
    -------
    Gamma_i: array, shape (s,da,db)
        Gamma_i: Vidal form tensor on site i
    Btilde_ip1: array, shape (s,db,dright)
        Tensor to be used as input for the next canonization step
    S2: array, shape (min(da,db)) (singular values)
        np.diag(S) corresponds to Lambda_i, Vidal form tensor between site i and i+1
        to be used as input for next step
    """
    s,da,db=np.shape(Btilde_i)
    reshapedB=np.reshape(Btilde_i,(s*da,db))
    U,S2,Vdag=np.linalg.svd(reshapedB,full_matrices=0)
    Gamma_i=np.reshape(U,(s,da,U.shape[1]))[:,:,:db]
    Gamma_i=np.tensordot(inverse(S1,da),Gamma_i,axes=(1,1))
    Gamma_i=np.transpose(Gamma_i,(1,0,2))
    Btilde_ip1=np.tensordot(np.dot(np.diag(S2),Vdag),B_ip1,axes=(1,1))
    Btilde_ip1=np.transpose(Btilde_ip1,(1,0,2))  
    return Gamma_i,Btilde_ip1,S2

def canonize_end(Btilde_N,S1):
    """
    performs svd on last site
    input: M1 obtained from svd of the last step, S1 singular values of svd of last step
    output: Gamma matrix of last site
    wave-function is normalized by setting lambda of last site to 1
    
    performs svd on last site
    
    Parameters
    ----------
    Btilde_N: array, shape (s,da,db) 
        tensor on site i obtained from svd of the last step
    S1: array, 
        singular values obtained in previous step

    Returns
    -------
    Gamma_N: array, shape (s,da,db) with db=1
        Gamma_N: Vidal form tensor on site N
   
    The wave-function is normalized by setting lambda of last site to 1
    """
    s,da,db=np.shape(Btilde_N)
    reshapedB=np.reshape(Btilde_N,(s*da,db))
    U,S2,Vdag=np.linalg.svd(reshapedB,full_matrices=0)
    Gamma_N=np.reshape(U,(s,da,U.shape[1]))[:,:,:1]
    Gamma_N=np.tensordot(inverse(S1,da),Gamma_N,axes=(1,1))
    Gamma_N=np.transpose(Gamma_N,(1,0,2))
    return Gamma_N



def canonize(M):
    """
    Given an MPS, this function computes the Vidal form 
    by first right-normalizing and then performing a sweep from the left.
    
    Parameters
    ----------
    M: list of tensors [M_1, ...M_N]
       where M_i is an array of shape (s,dleft,dright)-> s=2 (physical index), dleft and dright are the bond dimensions
       Matrix Product representation of a given state with N spins
    
    Returns
    -------
    Gammas: list
       [Gamma1,...GammaN]
    Lambdas: list
       [Lambda1,....LambdaN-1]
           
    s.t. the MPS in Vidalform is: [Gamma1,Lambda1,Gamma2,....LambdaN-1,GammaN]
    """
    N=len(M)
    M=right_canonize_complete(M)
    Gammas=[]
    # add dummy lambda at "0th" bond
    Lambdas=[np.ones(1)]
    Gamma1,M_i,S_i=canonize_start(M[0],M[1])
    Gammas.append(Gamma1)
    Lambdas.append(S_i)
    for i in range(N-2):
        Gammai,M_i,S_i=canonize_step(M_i,S_i,M[i+2])
        Gammas.append(Gammai)
        Lambdas.append(S_i)
    Gamma_N=canonize_end(M_i,S_i)
    Gammas.append(Gamma_N)
    # add dummy lambda at "Lth" bond
    Lambdas.append(np.ones(1))
    return Gammas,Lambdas

"""
Expectation value
===========================
"""
def begin_exp(G1, Lam1,H):
    """
    Performs first step of computing the expectation value <Psi|H|Psi>
    Parameters
    ----------
    G1: array, shape (2,1,db_psi)
        Gamma_1 of Psi (Vidal tensor at site 1)
        
    Lam1: array, shape (db_psi) (careful with svd rank, if truncated)
        Lambda_1 of Psi (Vidal tensor between sites 1 and 2) 
        
    H: array, shape (s,s,1,db_H) 
        MPO at site 1
        
    Returns
    -------
    L: array, shape (db_psi, db_H, db_psi)
       contraction at site 1, including the Lambda matrices between site 1 and 2
       to be used as input for next step
    """
    
    A1=np.tensordot(G1[:,:,:np.shape(Lam1)[0]],np.diag(Lam1),axes=(2,0))
    
    Adag1=np.conj(A1)
    
    L=np.tensordot(A1,H,axes=(0,0))
    L=np.tensordot(L,Adag1,axes=(2,0))
    L=np.reshape(L,(np.shape(A1)[2],np.shape(H)[3],np.shape(Adag1)[2]))
    return L

def step_exp(L,G1,Lam1,H):
    """
    Performs i'th step of computing the expectation value <Psi|H|Psi>
    Parameters
    ----------
    L: array, shape (da_psi, da_H, da_psi)
        contraction obtained in previous step
        
    G1: array, shape (2,da_psi,db_psi)
        Gamma_i of Psi (Vidal tensor at site i)
        
    Lam1: array, shape (db_psi) (careful with svd rank, if truncated)
        Lambda_i of Psi (Vidal tensor between sites i and i+1) 
        
    H: array, shape (s,s,da_H,db_H) 
        MPO at site i
        
    Returns
    -------
    L: array, shape (db_psi, db_H, db_psi)
       contraction up to site i, including the Lambda matrices between site i and i+1
       to be used as input for next step
    """
    A1=np.tensordot(G1[:,:,:np.shape(Lam1)[0]],np.diag(Lam1),axes=(2,0))
    Adag1=np.conj(A1)
 
    L=np.tensordot(L,A1,axes=(0,1))
    L=np.tensordot(L,H,axes=([0,2],[2,0]))
    L=np.tensordot(L,Adag1,axes=([0,2],[1,0]))
    return L

def end_exp(L,G1,H):
    """
    G1: Gamma matrices of Psi at site N
    H: MPO at site N
    L: contraction up to site N-1
    returns complete contraction
    
    Performs the las step of computing the expectation value <Psi|H|Psi>
    
    Parameters
    ----------
    L: array, shape (da_psi, da_H, da_psi)
       contraction obtained in previous step
    
    G1: array, shape (2,da_psi,1)
       Gamma_N of Psi (Vidal tensor at site N)
        
    H: array, shape (s,s,da_H,1)
       MPO at site N

    Returns
    -------
    exp_value: real or complex
       contraction up to site N, corresponding to the expectation value <Psi|H|Psi>
    """
    A1=G1
    Adag1=np.conj(A1)
    L=np.tensordot(L,A1,axes=(0,1))
    L=np.tensordot(L,H,axes=([0,2],[2,0]))
    L=np.tensordot(L,Adag1,axes=([0,2],[1,0]))
    exp_value=L[0,0,0]
    return exp_value

def calculate_exp(Gammas1,Lambdas1,Hamiltonian):
    """returns the expectation value <Psi|H|Psi>
    with Gammas1,Lambdas1 the canonized MPS representation of Psi
    and Hamiltonian an MPO representation of H
    """
    # drop the dummy Lambdas
    Lambdas1 = Lambdas1[1:-1]
    L=begin_exp(Gammas1[0],Lambdas1[0],Hamiltonian[0])
    for i in range(len(Gammas1)-2):
        L=step_exp(L,Gammas1[i+1],Lambdas1[i+1],Hamiltonian[i+1])
    L=end_exp(L,Gammas1[-1],Hamiltonian[-1])
    return L


"""
DMRG
==================================================
"""

# ---- R^{i + 1}     -- B^{\dag} -- R^i
#      R^{i + 1}          |         R^i
# ---- R^{i + 1} === ---- W ------- R^i
#      R^{i + 1}          |         R^i
# ---- R^{i + 1}     ---- B ------- R^i

def add_site_to_R_env(R_env, B, W):
    """
    R_env: right environment from previous step; shape (da_psi, da_H, da_psi) ### CHECK THIS ###
    B: right-normalized; shape (s, da_psi, db_psi)
    W: MPO; shape (s, s, da_H, db_H)
    
    Returns
    R_env: updated right environment; shape (db_psi, db_H, db_psi)
    """

    R_env2 = R_env

    R_env = np.tensordot(B.conj(), R_env, axes=(2, 0))
    R_env = np.tensordot(W, R_env, axes=([0,3],[0,2]))
    R_env = np.tensordot(B, R_env, axes=([0,2],[0,3]))
    R_env = np.transpose(R_env, (2, 1, 0))
    
    # Method 2: using np.einsum
    # contract a_i' (index r in einsum) 
    R_env2 = np.einsum('slr,rab->slab', B.conj(), R_env2) 
    # contract b_i (index d in einsum) and sigma'_i (index a in einsum)
    R_env2 = np.einsum('abcd,axdy->bcxy', W, R_env2)
    # contract sigma_i and a_i
    R_env2 = np.einsum('abc,axyc->bxy', B, R_env2)
    R_env2 = np.transpose(R_env2, (2, 1, 0))

    # R_env and R_env2 should be the same
    if  np.allclose(R_env, R_env2, atol=1e-10) is False:
        print("Inconsistency! ", R_env, R_env2)

    return R_env    

#  L^i -- A^{\dag} ---     L^{i + 1} ---
#  L^i       |             L^{i + 1}
#  L^i ------W ------- === L^{i + 1} ---
#  L^i       |             L^{i + 1}
#  L^i ------A -------     L^{i + 1} ---

def add_site_to_L_env(L_env, A, W):
    """
    L_env: left environment from previous step; shape (da_psi, da_H, da_psi) 
    A: left-normalized; shape (s, da_psi, db_psi)
    W: MPO; shape (s, s, da_H, db_H)
    
    Returns
    L_env: updated left environment; shape (db_psi, db_H, db_psi)
    """
        
    L_env2 = L_env 
    
    # Methods 1 and 2 are equivalent
    # Method 1: using np.tensordot
    L_env2 = np.tensordot(A.conj(), L_env2, axes=(1, 0))
    L_env2 = np.tensordot(W, L_env2, axes=([0,2],[0,2]))
    L_env2 = np.tensordot(A, L_env2, axes=([0,1],[0,3]))
    L_env2 = np.transpose(L_env2, (2, 1, 0))
    
    # Method 2: using np.einsum
    L_env = np.einsum('slr,lxy->srxy', A.conj(), L_env)
    L_env = np.einsum('axby,awbz->xywz', W, L_env)
    L_env = np.einsum('slr,sxyl->rxy', A, L_env)
    L_env = np.transpose(L_env, (2, 1, 0))

    # L_env and L_env2 should be the same
    if  np.allclose(L_env, L_env2, atol=1e-10) is False:
        print("Inconsistency! ", L_env, L_env2)

    return L_env2


# L -- M^{\dag} -- R
# L    |           R
# L -- W --------- R
# L    |           R
# L -- M --------- R

def H_local(L_env, W, R_env, M):
    """
    L_env: left environment up to site l-l
    W: MPO at site l
    R_env: right environment from site l+1
    M: MPS matrix at site l
    """
    s = 2
    if len(M.shape) == 1:  # in case of the boundary
        flatten = True
        M = np.reshape(M, (s, L_env.shape[0], R_env.shape[0]))
    elif len(M.shape) == 3:  # in case of the bulk index
        flatten = False
    else:
        raise ValueError('Unknown format for M')
    
    # again, you can use either np.tensordot or np.einsum 
    # contract a_{l - 1}
    hpsi = np.einsum('abc,scd->absd', L_env, M)
    # contract b_{l} and \sigma_l
    hpsi = np.einsum('abcd,xcby->adxy', W, hpsi)
    # contract b_{l - 1} and a_{l}
    hpsi = np.einsum('abcd,xbd->acx', hpsi, R_env)
    if flatten:
        return hpsi.flatten()
    return hpsi


def run_DMRG(W, chi=60, nmax=1000, verbose=False, atol=1e-12):
    """
    W: Hamiltonian as MPO [W1, W2, ..., WL]
    chi: maximum bond dimension
    nmax: maximum number of left and right sweeps
    verbose: if True, print intermediate information
    atol: tolerance for convergence
    
    returns: 
    
    
    
    """
    # number of sites
    L = len(W)
    
    # local Hilbert space dimension (=2 for spins)
    s = W[0].shape[0]

    # define random MPS with given maximal bond dimension
    shapes = [(s, min(chi, s ** i, s ** (L - i)), min(chi, s ** (i + 1), s ** (L - 1 - i))) for i in range(L)]
    M = [np.random.standard_normal(shape)for shape in shapes]
    
    # right-normalize the MPS
    for i in range(L - 1, 0, -1):
        M[i - 1], M[i] = right_canonize(M[i - 1], M[i])
    
    # to run on the left boundary, introduce fake matrix that we discard later
    _, M[0] = right_canonize(np.ones((1, 1, 1)), M[0])
    
    ## compute right environments
    R_environments = [np.ones((1, 1, 1))]    # again start with a fake matrix on the right boundary
    for i in range(1, L):
        R_environments.append(add_site_to_R_env(R_environments[i - 1], M[-i], W[-i]))
    
    # algorithm has converged if delta = abs(energy - energy_previous) < atol
    energy_previous = np.inf
    delta = np.inf
    # we store the energies after every optimization step
    energies = []
    for n in range(nmax):
        ## right sweep
        L_environments = [np.ones((1, 1, 1))]  # again start with a fake matrix on the left boundary
    
        #Lambdas = []
        for i in range(L - 1):
            # precompute the local H dimension
            local_H_dim = s * L_environments[i].shape[0] * R_environments[L - 1 - i].shape[0]
            
            # construct the LinearOperator 
            Hop = scipy.sparse.linalg.LinearOperator((local_H_dim, local_H_dim), \
                matvec = lambda M: H_local(L_environments[i], W[i], R_environments[L - 1 - i], M))
            
            # obtain best local MPS (linearized) and local energy using the Lanczos algorithm
            energy, V = scipy.sparse.linalg.eigsh(Hop, k = 1, v0 = M[i].flatten(), \
                                                  tol = 1e-2 if n < 2 else 0, which = 'SA')
            energy = energy[0]
            energies.append(energy)
            if verbose:
                print("E = ", energy)
            delta = energy - energy_previous
            energy_previous = energy

            # reshape the obtained result to the shape of the MPS matrix
            M[i] = V.reshape((s, L_environments[i].shape[0], R_environments[L - 1 - i].shape[0]))
            
            # left-normalize the result (it will affect M[i + 1], but we will optimize it the next step)
            M[i], M[i + 1], Lambda = left_canonize(M[i], M[i + 1], return_S = True)

            L_environments.append(add_site_to_L_env(L_environments[i], M[i], W[i]))

        ## repeat the same for left sweep
        R_environments = [np.ones((1, 1, 1))]
        Lambdas = []
        for i in range(L - 1, 0, -1):
            local_H_dim = s * L_environments[i].shape[0] * R_environments[L - 1 - i].shape[0]

            Hop = scipy.sparse.linalg.LinearOperator((local_H_dim, local_H_dim), \
                                 matvec = lambda M: H_local(L_environments[i], W[i], \
                                                            R_environments[L - 1 - i], M))

            energy, V = scipy.sparse.linalg.eigsh(Hop, k = 1, v0 = M[i].flatten(), \
                                                  tol = 1e-2 if n < 2 else 0, which = 'SA')
            energy = energy[0]
            energies.append(energy)
            if verbose:
                print("E = ", energy)
            delta = energy - energy_previous
            energy_previous = energy
            # print(energy, i, 'left')
            M[i] = V.reshape((s, L_environments[i].shape[0], R_environments[L - 1 - i].shape[0]))
            M[i - 1], M[i], Lambda = right_canonize(M[i - 1], M[i], return_S = True)
            Lambdas.append(Lambda)
            R_environments.append(add_site_to_R_env(R_environments[L - 1 - i], M[i], W[i]))
        
        # === check convergence ===
        if verbose:
            print(f"step {n+1}: E = {energy}, dE = {abs(delta)}")
        if abs(delta) < atol:
            if verbose:
                print(f"Converged after {n+1} sweeps!")
                print(f'Ground-state energy: {energy}')
            break
        # === check convergence ===

    return energies, Lambdas, M
