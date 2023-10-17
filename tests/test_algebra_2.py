"""
Test the advanced multilinear algebra operations between torchtt.TT objects.
Some operations (matvec for large ranks and elemntwise division) can be only computed using optimization (AMEN and DMRG).
"""
import pytest
import torchtt as tntt
import torch as tn
import numpy as np


def err_rel(t, ref): return tn.linalg.norm(t-ref).numpy() / \
    tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf
    
def create_banded(a, m, A, band=1):
    A_band = tn.zeros(a, m, m, A, dtype=tn.float64)
    for k in range(-band, band+1):
        diag = tn.diagonal(A_band, k, 1, 2)
        diag += 100 * tn.rand(a, A, diag.shape[-1])
    return A_band


@pytest.mark.parametrize("dtype", [tn.float64, tn.complex128])
def test_dmrg_hadamard(dtype):
    """
    Test hadamard product using DMRG.
    """
    n = 32
    z = tntt.random([n]*8,[1]+7*[3]+[1], dtype=dtype)
    zm = z + z

    x = tntt.random([n]*8,[1]+7*[5]+[1], dtype=dtype)
    xm = x + x
    xm = xm + xm

    # conventional method 
    y = 8 * (z * x).round(1e-12)

    yf = tntt.dmrg_hadamard(zm, xm, eps=1e-12, verb=False)

    rel_error = (y-yf).norm().numpy()/y.norm().numpy()

    assert rel_error < 1e-12


@pytest.mark.parametrize("dtype", [tn.complex128])
def test_dmrg_matvec(dtype):
    """
    Test the fast matrix vector product using DMRG iterations.
    """
    n = 32
    A = tntt.random([(n, n)]*8, [1]+7*[3]+[1], dtype=dtype)
    Am = A + A

    x = tntt.random([n]*8, [1]+7*[5]+[1], dtype=dtype)
    xm = x + x
    xm = xm + xm

    # conventional method
    y = 8 * (A @ x).round(1e-12)

    # dmrg matvec
    yf = Am.fast_matvec(xm)

    rel_error = (y-yf).norm().numpy()/y.norm().numpy()

    assert rel_error < 1e-12


@pytest.mark.parametrize("dtype", [tn.complex128])
def test_dmrg_matvec_non_square(dtype):
    """
    Test the fast matrix vector product using DMRG iterations for non-square matrices.
    """
    n = 32
    A = tntt.random([(n+2,n)]*8,[1]+7*[3]+[1], dtype=dtype)
    Am = A + A 

    x = tntt.random([n]*8,[1]+7*[5]+[1], dtype=dtype)
    xm = x + x
    xm = xm + xm

    # conventional method 
    y = 8 * (A @ x).round(1e-12)

    # dmrg matvec
    yf = Am.fast_matvec(xm)

    rel_error = (y-yf).norm().numpy()/y.norm().numpy()

    assert rel_error < 1e-12
      

@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_division(dtype):
    """
    Test the division between tensors performed with AMEN optimization.
    """
    N = [7, 8, 9, 10]
    xs = tntt.meshgrid(
        [tn.linspace(0, 1, n, dtype=dtype) for n in N])
    x = xs[0]+xs[1]+xs[2]+xs[3]+xs[1]*xs[2]+(1-xs[3])*xs[2]+1
    x = x.round(0)
    y = tntt.ones(x.N, dtype=dtype)

    a = y/x
    b = 1/x
    c = tn.tensor(1.0)/x

    assert err_rel(a.full(), y.full()/x.full()) < 1e-11
    assert err_rel(b.full(), 1/x.full()) < 1e-11
    assert err_rel(c.full(), 1/x.full()) < 1e-11

@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_division_preconditioned(dtype):
    """
    Test the elementwise division using AMEN (use preconditioner for the local subsystem).
    """
    N = [7, 8, 9, 10]
    xs = tntt.meshgrid(
        [tn.linspace(0, 1, n, dtype=dtype) for n in N])
    x = xs[0]+xs[1]+xs[2]+xs[3]+xs[1]*xs[2]+(1-xs[3])*xs[2]+1
    x = x.round(0)
    y = tntt.ones(x.N)

    a = tntt.elementwise_divide(y, x, preconditioner='c')

    assert err_rel(a.full(), y.full()/x.full()) < 1e-11

@pytest.mark.parametrize("dtype", [tn.float64])
@pytest.mark.parametrize("cpp", [False, True] if tntt.cpp_enabled() else [False])
def test_amen_mv(dtype, cpp):
    """
    Test the AMEn matvec.
    """

    A = tntt.randn([(3, 4), (5, 6), (7, 8), (2, 3)], [1, 2, 2, 3, 1], dtype=dtype)
    x = tntt.randn([4, 6, 8, 3], [1, 4, 3, 3, 1], dtype=dtype)

    Cr = 25 * A @ x

    A = A + A + A + A + A
    x = x + x + x + x + x

    C = tntt.amen_mv(A, x, use_cpp=cpp)

    assert ((C-Cr).norm()/Cr.norm()) < 1e-11

    bands_A = [1, 1, 1]
    A = tntt.randn([(128, 128), (32, 32), (32, 32)], [1, 20, 5, 1], dtype=tn.float64)
    x = tntt.randn([128, 32, 32], [1, 4, 13, 1], dtype=tn.float64)
    for i in range(len(A.cores)):
        A_core = A.cores[i]
        A.cores[i] = create_banded(A_core.shape[0], A_core.shape[1], A_core.shape[-1], bands_A[i])
    
    yr = 25 * A @ x
    
    A = A + A + A + A + A
    x = x + x + x + x + x
    
    y = tntt.amen_mv(A, x)
    assert ((y-yr).norm()/yr.norm()) < 1e-11
    y = tntt.amen_mv(A, x, bandsA=bands_A)
    assert ((y-yr).norm()/yr.norm()) < 1e-11
    
@pytest.mark.parametrize("dtype", [tn.float64])
@pytest.mark.parametrize("cpp", [False, True] if tntt.cpp_enabled() else [False])
def test_amen_mv_multiple(dtype, cpp):
    """
    Test the AMEn matvec with multiple vectors.
    """

    A = tntt.randn([(3, 4), (5, 6), (7, 8), (2, 3)], [1, 2, 2, 3, 1], dtype=dtype)
    xs = []
    Cr = 0
    for i in range(8):
        xs.append(tntt.randn([4, 6, 8, 3], [1, 3, 2, 2, 1], dtype=dtype))
        xs[-1] = xs[-1] - xs[-1] + xs[-1]
        Cr = Cr + A@xs[-1]
    C = tntt.amen_mv(A, xs, use_cpp=cpp)

    assert ((C-Cr).norm()/Cr.norm()) < 1e-11

@pytest.mark.parametrize("dtype", [tn.float64])
@pytest.mark.parametrize("cpp", [False, ] if tntt.cpp_enabled() else [False])
def test_amen_mvm(dtype, cpp):
    """
    Test the AMEn matvec with multiple vectors.
    """

    n = 3
    As = [tntt.randn([(3, 4), (5, 6), (7, 8), (2, 3)], [1, 2, 2, 3, 1], dtype=dtype) for i in range(n)]
    xs = [tntt.randn([4,6,8,3], [1, 2, 2, 1, 1], dtype=dtype) for i in range(n)]
    Bs = [tntt.randn([( 4,2), (6,3), (8,2), (3,1)], [1, 2, 2, 3, 1], dtype=dtype) for i in range(n)]

    Cr = 0
    for i in range(n):
        Cr += As[i] @ tntt.diag(xs[i]) @ Bs[i]
        Cr = Cr.round(1e-14)
    C = tntt.amen_mvm(As, xs, Bs, use_cpp=cpp)

    assert ((C-Cr).norm()/Cr.norm()) < 1e-11
    
@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_mm(dtype):
    """
    Test the AMEn matmat.
    """

    A = tntt.randn([(3, 4), (5, 6), (7, 8), (2, 3)], [1, 2, 2, 3, 1], dtype=dtype)
    B = tntt.randn([(4, 2), (6, 4), (8, 5), (3, 7)], [1, 4, 3, 3, 1], dtype=dtype)

    Cr = 25 * A @ B

    A = A + A + A + A + A
    B = B + B + B + B + B

    C = tntt.amen_mm(A, B)

    assert ((C-Cr).norm()/Cr.norm()) < 1e-11
    
    bands_A = [2, 1, 0]
    bands_B = [3, 2, 0]
    A = tntt.random([(128, 128), (32, 32), (16, 16)], [1, 2, 20, 1])
    B = tntt.random([(128, 128), (32, 32), (16, 16)], [1, 20, 3, 1])
    for i in range(len(A.cores)):
        A_core = A.cores[i]
        B_core = B.cores[i]
        A.cores[i] = create_banded(A_core.shape[0], A_core.shape[1], A_core.shape[-1], bands_A[i])
        B.cores[i] = create_banded(B_core.shape[0], B_core.shape[1], B_core.shape[-1], bands_B[i])
    Cr = 25 * A @ B
    
    A = A + A + A + A + A
    B = B + B + B + B + B
        
    C = tntt.amen_mm(A, B)
    assert ((C-Cr).norm()/Cr.norm()) < 1e-11
    C = tntt.amen_mm(A, B, bandsA=bands_A)
    assert ((C-Cr).norm()/Cr.norm()) < 1e-11
    C = tntt.amen_mm(A, B, bandsB=bands_B)
    assert ((C-Cr).norm()/Cr.norm()) < 1e-11
    C = tntt.amen_mm(A, B, bandsA=bands_A, bandsB=bands_B)
    assert ((C-Cr).norm()/Cr.norm()) < 1e-11


if __name__ == '__main__':
    pytest.main()

