"""
Contains iteratiove solvers like GMRES and BiCGSTAB

@author: ion
"""

import torch as tn
import datetime
import numpy as np
import scipy.linalg as slinalg

def BiCGSTAB(Op, rhs, x0, eps=1e-6, nmax = 40):
    pass

def BiCGSTAB_reset(Op,rhs,x0,eps=1e-6,nmax=40):
    """
    BiCGSTAB solver.
    """ 
    # initial residual
    r = rhs - Op.matvec(x0)
    
    # choose rop
    r0p = tn.rand(r.shape,dtype = x0.dtype)
    while tn.dot(r.squeeze(),r0p.squeeze()) == 0:
        r0p = tn.rand(r.shape,dtype = x0.dtype)
        
    p = r
    x = x0
    
    norm_rhs = tn.linalg.norm(rhs)
    r_nn = tn.linalg.norm(r)
    nit = 0 
    for k in range(nmax):
        nit += 1
        Ap = Op.matvec(p)
        alpha = tn.dot(r.squeeze(),r0p.squeeze()) / tn.dot(Ap.squeeze(),r0p.squeeze())
        s = r - alpha * Ap
        if tn.linalg.norm(s)<eps:
            x_n = x+alpha*p
            break
        
        As = Op.matvec(s)
        omega = tn.dot(As.squeeze(),s.squeeze()) / tn.dot(As.squeeze(),As.squeeze())
        
        x_n = x + alpha*p + omega*s
        r_n = s - omega*As
        r_nn = tn.linalg.norm(r_n)
        # print('\t\t\t',r_nn)
        # print(r_nn,eps,norm_rhs)
        if r_nn < eps * norm_rhs:
        # if tf.linalg.norm(r_n)<eps:
            #print(r_n)
            break
        
        beta = (alpha/omega)*tn.dot(r_n.squeeze(),r0p.squeeze())/tn.dot(r.squeeze(),r0p.squeeze())
        p = r_n+beta*(p-omega*Ap)
        
        if abs(tn.dot(r_n.squeeze(),r0p.squeeze())) < 1e-6:
            r0p = r_n
            p_n = r_n
        # updates
        r = r_n
        x = x_n
        
    flag = False if k==nmax else True
    
    relres = r_nn/norm_rhs 
    
    return x_n,flag,nit,relres

    

def gmres_restart(LinOp, b, x0 , N, max_iterations, threshold, resets = 4):
    
    iters = 0
    converged = False
    for r in range(resets):
        x0, flag, it = gmres(LinOp,b,x0, N, max_iterations,threshold)
        iters += it
        if flag:
            converged = True
            break
    return x0, converged, iters
                 

def gmres(LinOp, b, x0, N, max_iterations, threshold):

    converged = False
    
    r = (b - LinOp.matvec(x0)).squeeze()
    
    b_norm = tn.linalg.norm(b)
    error = tn.linalg.norm(r) / b_norm

    e1 = tn.zeros((max_iterations + 1), dtype=b.dtype, device=b.device)
    e1[0] = 1
    
    sn = []
    cs = []

    err = [error]
    
    r_norm = tn.linalg.norm(r)
    if not r_norm > 0:
        return x0, True, 0

    Q = tn.empty((N, max_iterations+1), dtype=b.dtype, device=b.device) 
    Q[:,0] = r / r_norm
    H = tn.zeros((max_iterations + 1, max_iterations), dtype=b.dtype, device=b.device)
    
    beta = (r_norm * e1).cpu().numpy()
  
    for k in range(max_iterations):
        
        q = LinOp.matvec(Q[:,k]).squeeze()

        for _ in range(2):
            QCq = Q[:, :k + 1].T @ q
            H[:k + 1, k] += QCq
            q = q - Q[:, :k + 1] @ QCq
        h = tn.linalg.norm(q)
        
        q = q / h
        H[k + 1, k] = h
        Q[:, k + 1] = q
        
        column, c, s = apply_givens_rotation(H[:k + 2, k], cs, sn, k + 1)
        H[:k + 2, k] = tn.from_numpy(column).to(h.device)
        cs.append(c)
        sn.append(s)

        xrot = slinalg.get_blas_funcs('rot', (beta,))
        xrot(beta[k : k + 1], beta[k + 1: k + 2], cs[k], sn[k], n=1, overwrite_x=True, overwrite_y=True)

        error = np.abs(beta[k + 1])
        err.append(error)
        
        if error <= threshold * b_norm:
            converged = True
            break
            
    y = tn.linalg.solve_triangular(H[:k + 1,:k + 1], tn.from_numpy(beta[:k + 1]).to(H.device).reshape([-1, 1]), upper=True)
    x = x0 + Q[:, :k + 1] @ y 
    return x, converged, k
    

  

def apply_givens_rotation(h, cs, sn, k):
    column = h.cpu().numpy()
    xrot = slinalg.get_blas_funcs('rot', (column,))
    for i in range(k - 1):
        xrot(column[i : i + 1], column[i + 1: i + 2], cs[i], sn[i], n=1, overwrite_x=True, overwrite_y=True)
    xrotg = slinalg.get_blas_funcs('rotg', (column,))
    cs_k, sn_k = xrotg(column[k - 1], column[k])
    xrot(column[k - 1 : k], column[k: k + 1], cs_k, sn_k, n=1, overwrite_x=True, overwrite_y=True)
    #h = tn.from_numpy(column).to(h.device)
 
    return column, cs_k, sn_k


# class Lop():
    # def __init__(self):
        # n = 30
        # self.n =  n # mode size
        # self.A = -2*tn.eye(n, dtype = tn.float64)+tn.diag(tn.ones(n-1,dtype = tn.float64),-1)+tn.diag(tn.ones(n-1,dtype = tn.float64),1)
        # self.A[0,1] = 0
        # self.A[-1,-2] = 0
        # self.b =  tn.ones((n,1),dtype=tn.float64)
        # self.b[0,0] = 0
        # self.b[-1,0] = 0
    # def matvec(self, x):
        # return tn.reshape(self.A@x,[-1,1])
    
# lop  = Lop()

# x,flag,nit = gmres(lop,lop.b,lop.b,lop.n,40,1e-7)
# x_n,flag,nit,relres = BiCGSTAB_reset(lop,lop.b,lop.b)