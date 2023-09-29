"""
Test the multilinear solvers.
"""
import torchtt 
import torch as tn
import numpy as np
import pytest

err_rel = lambda t, ref :  tn.linalg.norm(t-ref).numpy() / tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf

def create_banded(a, m, A, band=1):
    A_band = tn.zeros(a, m, m, A, dtype=tn.float64)
    for k in range(-band, band+1):
        diag = tn.diagonal(A_band, k, 1, 2)
        diag += 100 * tn.rand(a, A, diag.shape[-1])
    return A_band

basic_dtype = tn.complex128

@pytest.mark.skipif(not torchtt.solvers.cpp_enabled(), reason="C++ extension must be present.")
def test_amen_solve():
    A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
    x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
    b = A @ x 
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner=None, use_cpp=True) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed."

@pytest.mark.skipif(not torchtt.solvers.cpp_enabled(), reason="C++ extension must be present.")
def test_amen_solve_cprec():
    A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
    x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
    b = (A @ x).round(1e-16) 
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='c', use_cpp=True) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed (c preconditioner)."

@pytest.mark.skipif(not torchtt.solvers.cpp_enabled(), reason="C++ extension must be present.")
def test_amen_solve_rprec():
    A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
    x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
    b = A @ x 
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='r', use_cpp=True) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed (right preconditioner)."

def test_amen_solve_cprec_nocpp():
    A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
    x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
    b = A @ x 
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='c', use_cpp=False) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed (c preconditioner, without C++)."

def test_amen_solve_rprec_nocpp():
    A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
    x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
    b = A @ x 
    xx = torchtt.solvers.amen_solve(A, b, verbose = False, eps=1e-10, nswp = 40, preconditioner='r', use_cpp=False) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed (right preconditioner)."

def test_amen_solve_nocpp():
    A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
    x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
    b = A @ x 
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner=None, use_cpp = False) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed."
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='c', use_cpp = False) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed (c preconditioner)."
    
    bands_A = [1, 1, 1]
    A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
    x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
    for i in range(len(A.cores)):
        A_core = A.cores[i]
        A.cores[i] = create_banded(A_core.shape[0], A_core.shape[1], A_core.shape[-1], bands_A[i])
    b = A @ x 
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner=None, use_cpp = False, bandsA=bands_A) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed."
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='c', use_cpp = False, bandsA=bands_A) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed (c preconditioner)."
    
