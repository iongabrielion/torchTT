"""
System solvers in the TT format.

"""

import torch as tn
import numpy as np
import torchtt
import datetime
from torchtt._decomposition import QR, SVD, lr_orthogonal, rl_orthogonal, rank_chop
from torchtt._iterative_solvers import BiCGSTAB_reset, gmres_restart
import opt_einsum as oe
import torch.nn.functional as tnf
from .errors import *
from ._amen_approx import amen_approx, mv_local_op, mv_multiple_local_op, mvm_multiple_local_op, mm_multiple_local_op, hadamard_multiple_local_op

try:
    import torchttcpp
    _flag_use_cpp = True
except:
    import warnings
    warnings.warn(
        "\x1B[33m\nC++ implementation not available. Using pure Python.\n\033[0m")
    _flag_use_cpp = False


def cpp_enabled():
    """
    Is the C++ backend enabled?

    Returns:
        bool: the flag
    """
    return _flag_use_cpp

# def _local_AB(Phi_left, Phi_right, coreA, coreB, bandA=-1, bandB=-1):
#     """
#     Perfomrs the contraction for the right side of amen mm

#     Args:
#         Phi_left (torch.tensor): left phi
#         Phi_right (torch.tensor): right phi
#         coreA (torch.tensor): core of A
#         coreB (torch.tensor): core of B
#         bandA (int): if positive specifies number of diagonals in the square matrix core of A [i, :, :, j]. 0 means diagonal structure, 1 means                        tridiagonal, ...
#         bandB (int): if positive specifies number of diagonals in the square matrix core of B [i, :, :, j]. 0 means diagonal structure, 1 means                        tridiagonal, ...

#     Returns:
#         torch.tensor: _description_
#     """
#     w = tn.zeros(Phi_left.shape[0], coreA.shape[1], coreB.shape[2], Phi_right.shape[0], dtype=coreA.dtype, device=coreA.device)
#     if coreA.shape[1] != coreA.shape[2]:
#         bandA = -1
#     if coreB.shape[1] != coreB.shape[2]:
#         bandB = -1
#     if bandA < 0:
#         if bandB < 0:
#             w = oe.contract('rab,amkA,bknB,RAB->rmnR', Phi_left, coreA, coreB, Phi_right)
#         else:
#             for i in range(-bandB, bandB+1):
#                 diag_B = tn.diagonal(coreB, i, 1, 2)
#                 if i < 0:
#                     w[:, :, :i, :] += oe.contract('rab,amnA,bBn,RAB->rmnR', Phi_left, coreA[:, :, -i:, :], diag_B, Phi_right)
#                 elif i > 0:
#                     w[:, :, i:, :] += oe.contract('rab,amnA,bBn,RAB->rmnR', Phi_left, coreA[:, :, :-i, :], diag_B, Phi_right)
#                 else:
#                     w += oe.contract('rab,amnA,bBn,RAB->rmnR', Phi_left, coreA, diag_B, Phi_right)
#     else:
#         if bandB < 0:
#             for i in range(-bandA, bandA+1):
#                 diag_A = tn.diagonal(coreA, i, 1, 2)
#                 if i < 0:
#                     w[:, -i:, :, :] += oe.contract('rab,aAm,bmnB,RAB->rmnR', Phi_left, diag_A, coreB[:, :i, :, :], Phi_right)
#                 elif i > 0:
#                     w[:, :-i, :, :] += oe.contract('rab,aAm,bmnB,RAB->rmnR', Phi_left, diag_A, coreB[:, i:, :, :], Phi_right)
#                 else:
#                     w += oe.contract('rab,aAm,bmnB,RAB->rmnR', Phi_left, diag_A, coreB, Phi_right)
#         else:
#             band = min(bandA + bandB, coreA.shape[1] - 1)
#             band_min = min(bandA, bandB)
#             for k in range(-band, band+1):
#                 diag_w = tn.diagonal(w, k, 1, 2)
#                 if k > 0:
#                     b = bandA - k
#                     d = bandB - k
#                     for l in [*range(-min(d, bandA), 0),  *range(max(0, k-bandB), min(k, bandA)+1), *range(k+1, k+min(b, bandB)+1)]:
#                         diag_A = tn.diagonal(coreA, l, 1, 2)
#                         diag_B = tn.diagonal(coreB, k-l, 1, 2)
#                         if l < k:
#                             diag_A = diag_A[:, :, :max(l, 0)-k]
#                         diag_B = diag_B[:, :, min(k, max(l, 0)):]
#                         diag_w_slice = diag_w[:, :, max(-l, 0):]
#                         if l > k:
#                             diag_w_slice = diag_w_slice[:, :, :k-l]
#                         diag_w_slice += oe.contract('rab,aAm,bBm,RAB->rRm', Phi_left, diag_A, diag_B, Phi_right)
#                 elif k < 0:
#                     d = bandB + k
#                     b = bandA + k
#                     for l in [*range(min(k-d, -bandA), k), *range(max(k, -bandB), min(0, k+bandA)+1), *range(1, min(b, bandB)+1)]:
#                         diag_B = tn.diagonal(coreB, l, 1, 2)
#                         diag_A = tn.diagonal(coreA, k-l, 1, 2)
#                         diag_A = diag_A[:, :, max(-max(l, k), 0):]
#                         if l > k:
#                             diag_B = diag_B[:, :, :k-min(l, 0)]
#                         diag_w_slice = diag_w[:, :, max(0, l):]
#                         if l < k:
#                             diag_w_slice = diag_w_slice[:, :, :l-k]
#                         diag_w_slice += oe.contract('rab,aAm,bBm,RAB->rRm', Phi_left, diag_A, diag_B, Phi_right)
#                 else:
#                     for l in range(-band_min, band_min+1):
#                         diag_A = tn.diagonal(coreA, l, 1, 2)
#                         diag_B = tn.diagonal(coreB, -l, 1, 2)
#                         if l > 0:
#                             diag_w[:, :, :-l] += oe.contract('rab,aAm,bBm,RAB->rRm', Phi_left, diag_A, diag_B, Phi_right)
#                         else:
#                             diag_w[:, :, -l:] += oe.contract('rab,aAm,bBm,RAB->rRm', Phi_left, diag_A, diag_B, Phi_right)

#     return w


def _local_AB(Phi_left, Phi_right, coreA, coreB, bandA=-1, bandB=-1):
    """
    Perfomrs the contraction for the right side of amen mm

    Args:
        Phi_left (torch.tensor): left phi
        Phi_right (torch.tensor): right phi
        coreA (torch.tensor): core of A
        coreB (torch.tensor): core of B

    Returns:
        torch.tensor: _description_
    """
    w = tn.zeros(Phi_left.shape[0], coreA.shape[1], coreB.shape[2],
                 Phi_right.shape[0], dtype=coreA.dtype, device=coreA.device)
    if (bandA >= 0) and (bandB >= 0):
        if (bandA >= bandB):
            bandA = -1
        else:
            bandB = -1

    if bandA < 0:
        if bandB < 0:
            w = oe.contract('rab,amkA,bknB,RAB->rmnR',
                            Phi_left, coreA, coreB, Phi_right)
        else:
            diagonals = tn.stack([tnf.pad(tn.diagonal(coreB, i, 1, 2), (-i, 0)) for i in range(-bandB, 0)] +
                                 [tnf.pad(tn.diagonal(coreB, i, 1, 2), (0, i)) for i in range(0, bandB+1)])
            tmp = oe.contract('rab,amnA,lbBn,RAB->lrmnR',
                              Phi_left, coreA, diagonals, Phi_right)
            w = tn.sum(tn.stack([tnf.pad(tmp[i + bandB, :, :, -i:, :], (0, 0, 0, -i)) for i in range(-bandB, 1)] +
                                [tnf.pad(tmp[i + bandB, :, :, :-i, :], (0, 0, i, 0)) for i in range(1, bandB+1)]), axis=0)
    else:
        diagonals = tn.stack([tnf.pad(tn.diagonal(coreA, i, 1, 2), (0, -i)) for i in range(-bandA, 0)] +
                             [tnf.pad(tn.diagonal(coreA, i, 1, 2), (i, 0)) for i in range(0, bandA+1)])
        tmp = oe.contract('rab,laAm,bmnB,RAB->lrmnR',
                          Phi_left, diagonals, coreB, Phi_right)
        w = tn.sum(tn.stack([tnf.pad(tmp[i + bandA, :, i:, :, :], (0, 0, 0, 0, 0, i)) for i in range(0, bandA+1)] +
                            [tnf.pad(tmp[i + bandA, :, :i, :, :], (0, 0, 0, 0, -i, 0)) for i in range(-bandA, 0)]), axis=0)
    return w

def amen_hadamard(xs, eps = 1e-10, nswp=20, y0 = None, rmax=999999, kickrank=4, kick2=0, verbose=False, use_cpp=True):
    """
    Perform elementwise multiplication between TT tensors.
    input `xs` has to be a list of tuple of tensors. The tensors in the tuple are elementwise multiplied and the results of the tuples arre added.
    
    Examples:
    
    ```
    import torchtt as tntt
    import torch as tn

    xs = [tntt.randn([4, 6, 8, 3], [1, 2, 2, 1, 1]) for i in range(8)]

    y_ref = xs[0]*xs[1] + 1e-6*xs[2]*xs[3]*xs[4] + xs[5]*xs[6] + xs[7]
    y_ref = y_ref.round(1e-18)

    y = tntt.amen_hadamard([(xs[0], xs[1]), (1e-2*xs[2], 1e-2*xs[3], 1e-2*xs[4]), (xs[5], xs[6]), (xs[7],
                                                                                                )], eps=1e-8)

    print((y-y_ref).norm()/y_ref.norm())
    ```

    Args:
        xs (list[tuple[torchtt.TT]]): _description_
        eps (float, optional): relative tolerance. Defaults to 1e-10.
        nswp (int, optional): number of sweeps. Defaults to 20.
        y0 (torchtt.TT, optional): initial guess. Defaults to None.
        rmax (int, optional): mqaximum rank. Defaults to 999999.
        kickrank (int, optional): kickrank. Defaults to 4.
        kick2 (int, optional): kick2. Defaults to 0.
        verbose (bool, optional): display additional info. Defaults to False.
        use_cpp (bool, optional): use the C++ extensions if available. Defaults to True.

    Raises:
        InvalidArguments: xs must be nonempty list of tuples of torch tensors.
        NotImplementedError: C++ extensions not available

    Returns:
        _type_: torchtt.TT
    """
    if not isinstance(xs, list) or len(xs) == 0 or not all([isinstance(xs[i], tuple) for i in range(len(xs))]): 
        raise InvalidArguments("xs must be nonempty list of tuples of torch tensors.")
    
    
    is_ttm = xs[0][0].is_ttm
    use_cpp = False
    if use_cpp and _flag_use_cpp:
        raise NotImplementedError("C++ extensions not available")
        # cores = torchttcpp.amen_solve(A_cores, B_cores, x_cores, b.N, A.R, b.R, x_R, nswp, eps, rmax, max_full, kickrank, kick2, local_iterations, resets, verbose, prec)
        # return torchtt.TT(list(cores))
    else:
        op = hadamard_multiple_local_op(xs, len(xs[0][0].cores), kickrank+kick2 > 0)
        return amen_approx(op, eps, xs[0][0].M if is_ttm else None, xs[0][0].N, y0, None, nswp, kickrank, kick2, 2 if verbose else 0, True, False, xs[0][0].cores[0].device)

    
    
        

def amen_mvm(A, x, B, nswp=22, y0=None, eps=1e-10, rmax=999999, kickrank=4, kick2=0, verbose=False, use_cpp=True):
    """
    Approxmate `y = A[0] @ totchtt.diag(x[0]) @ B[0] + A[1] @ totchtt.diag(x[1]) @ B[1] + ... `

    Args:
        A (list[torchtt.TT]): the TT matrices.
        x (list[torchtt.TT]): the TT tensors.
        B (list[torchtt.TT]): the TT matrices.
        nswp (int, optional): number of sweeps. Defaults to 22.
        y0 (torchtt.TT, optional): initial guess. Defaults to None.
        eps (float, optional): relative tolerance. Defaults to 1e-10.
        rmax (int, optional): maximum rank. Defaults to 1024.
        kickrank (int, optional): rank enrichment. Defaults to 4.
        kick2 (int, optional): rank of random component. Defaults to 0.
        verbose (bool, optional): discplay extra info to console. Defaults to False.
        use_cpp (bool, optional): use the C++ version (not implemented yet). Defaults to True.

    Returns:
        torchtt.TT: the result.
    """

    if use_cpp and _flag_use_cpp:
        if y0 == None:
            x_cores = []
            x_R = [1]*(1+len(A.N))
        else:
            x_cores = y0.cores
            x_R = y0.R

        # cores = torchttcpp.amen_solve(A_cores, B_cores, x_cores, b.N, A.R, b.R, x_R, nswp, eps, rmax, max_full, kickrank, kick2, local_iterations, resets, verbose, prec)
        # return torchtt.TT(list(cores))
    else:
        op = mvm_multiple_local_op(A, x, B, A[0].M, B[0].N, kickrank+kick2 > 0)
        return amen_approx(op, eps, A[0].M, B[0].N, None, None, nswp, kickrank, kick2, 2 if verbose else 0, True, False, A[0].cores[0].device)


def amen_mv(A, b, nswp=22, x0=None, eps=1e-10, rmax=1024, kickrank=4, kick2=0, verbose=False, use_cpp=True, bandsA=None):
    """
    Compute the matrix vector product between a TTM and a TT.
    Suited when the output is expected to be low rank. 

    Args:
        A (torchtt.TT): the matrix in TT.
        b (torchtt.TT | list[torchtt.TT]): the tensor TT or a list of tensors.
        nswp (int, optional): number of sweeps. Defaults to 22.
        x0 (torchtt.TT, optional): initial guess. In None is provided the initial guess is a ones tensor. Defaults to None.
        eps (float, optional): relative residual. Defaults to 1e-10.
        rmax (int, optional): maximum rank. Defaults to 100.
        kickrank (int, optional): rank enrichment. Defaults to 4.
        kick2 (int, optional): [description]. Defaults to 0.
        verbose (bool, optional): choose whether to display or not additional information during the runtime. Defaults to True.
        use_cpp (bool, optional): use the C++ implementation of AMEn. Defaults to True.
        bandsA (list, optional): for each core if positive specifies number of diagonals in the square matrix core of A [i, :, :, j].

    Raises:
        InvalidArguments: A and b must be TT instances.
        InvalidArguments: Invalid preconditioner.
        IncompatibleTypes: A must be TT-matrix and b must be vector.
        ShapeMismatch: Dimension mismatch.

    Returns:
        torchtt.TT: the approximation of the solution in TT format.
    """
    # perform checks of the input data
    if not (isinstance(A, torchtt.TT) and (isinstance(b, torchtt.TT) or isinstance(b, list))):
        raise InvalidArguments('A and b must be TT instances.')
    if not A.is_ttm:
        raise IncompatibleTypes('A must be TT-matrix and b must be vector.')
    # if A.N != b.N:
    #    raise ShapeMismatch('Dimension mismatch.')

    if use_cpp and _flag_use_cpp:
        if x0 is None:
            x_cores = torchtt.randn(
                A.M, [1]+[2]*(len(A.M)-1)+[1], device=A.cores[0].device).cores
        else:
            x_cores = x0.cores

        if isinstance(b, torchtt.TT):
            cores = torchttcpp.amen_mv(A.cores, b.cores, A.M, A.N, x_cores,
                                       eps, nswp, kickrank, kick2, 2 if verbose else 0, True, False)
            return torchtt.TT(list(cores))
        else:
            cores = torchttcpp.amen_mv_multiple(A.cores, [
                                                tmp.cores for tmp in b], A.M, A.N, x_cores, eps, nswp, kickrank, kick2, 2 if verbose else 0, True, False)
            return torchtt.TT(list(cores))
    else:
        if isinstance(b, list):
            op = mv_multiple_local_op(A, b, A.M, A.N, kickrank+kick2 > 0)
        else:
            op = mv_local_op(A, b, A.M, A.N, kickrank+kick2 > 0)
        return amen_approx(op, eps, None, A.M, x0, None, nswp, kickrank, kick2, 2 if verbose else 0, True, False, A.cores[0].device)
        # return _amen_mm_python(A.cores, [c[:, :, None, :] for c in b.cores], A.M, [1]*len(A.M), A.N, False, nswp, x0.cores if x0 is not None else None, x0.R if x0 is not None else None, eps, rmax, kickrank, kick2, verbose, bandsA)


def amen_mm(A, B, nswp=22, X0=None, eps=1e-10, rmax=1024, kickrank=4, kick2=0, verbose=False, use_cpp=True, bandsA=None, bandsB=None):
    """
    Perform the TTM-TTM product using AMEn optimization.
    Suited when the operators have high ranks, but the result is expected to be low rank.

    Args:
        A (torchtt.TT): the first TTM.
        B (torchtt.TT): the second TTM.
        nswp (int, optional): number of sweeps. Defaults to 22.
        X0 (torchtt.TT, optional): initial guess (None means no initial guess). Defaults to None.
        eps (float, optional): realtive tolerance. Defaults to 1e-10.
        rmax (int, optional): maximum rank. Defaults to 1024.
        kickrank (int, optional): kickrank. Defaults to 4.
        kick2 (int, optional): kick2. Defaults to 0.
        verbose (bool, optional): show debug info. Defaults to False.
        bandsA (list, optional): for each core if positive specifies number of diagonals in the square matrix core of A [i, :, :, j].
        bandsB (list, optional): for each core if positive specifies number of diagonals in the square matrix core of B [i, :, :, j].

    Returns:
        torchtt.TT: the result.
    """
    use_cpp = False
    
    if bandsA is None:
        bandsA = [[-1]*len(A.M)] if isinstance(A, torchtt.TT) else [[-1]*len(a.M) for a in A]
    elif isinstance(A, torchtt.TT):
        bandsA = [bandsA]
    if bandsB is None:
        bandsB = [[-1]*len(B.M)] if isinstance(B, torchtt.TT) else [[-1]*len(b.M) for b in B]
    elif isinstance(B, torchtt.TT):
        bandsB = [bandsB]

    if use_cpp and _flag_use_cpp:
        if X0 is None:
            x_cores = []
        else:
            x_cores = X0.cores

        # cores = torchttcpp.amen_solve(A_cores, B_cores, x_cores, b.N, A.R, b.R, x_R, nswp, eps, rmax, max_full, kickrank, kick2, local_iterations, resets, verbose, prec)
        # return torchtt.TT(list(cores))
    else:
        op = mm_multiple_local_op([A] if isinstance(A, torchtt.TT) else A, [B] if isinstance(B, torchtt.TT) else B, A.M if isinstance(A, torchtt.TT) else A[0].M, B.N if isinstance(B, torchtt.TT) else B[0].N, kickrank+kick2 > 0, bandsA, bandsB)
        return amen_approx(op, eps, A.M if isinstance(A, torchtt.TT) else A[0].M, B.N if isinstance(B, torchtt.TT) else B[0].N, X0, None, nswp, kickrank, kick2, 2 if verbose else 0, True, False, A.cores[0].device if isinstance(A, torchtt.TT) else A[0].cores[0].device)

    # return _amen_mm_python(A.cores, B.cores, A.M, B.N, A.N, True, nswp, X0.cores if X0 is not None else None, X0.R if X0 is not None else None,   eps, rmax, kickrank, kick2, verbose, bandsA, bandsB)


def _amen_mm_python(A_cores, B_cores, M, N, K, to_ttm, nswp=22, X0_cores=None, rx=None, eps=1e-10, rmax=1024, kickrank=4, kick2=0, verbose=False, bandsA=None, bandsB=None):
    if bandsA == None:
        bandsA = [-1] * len(A_cores)
    if bandsB == None:
        bandsB = [-1] * len(B_cores)
    if verbose:
        time_total = datetime.datetime.now()

    dtype = A_cores[0].dtype
    device = A_cores[0].device
    d = len(N)

    if X0_cores is None:
        x_cores = [tn.zeros([1, m, n, 1], dtype=dtype, device=device)
                   for m, n in zip(M, N)]
        rx = [1]*(d+1)
    else:
        x_cores = [tn.reshape(c, [c.shape[0], m, n, c.shape[-1]])
                   for c, m, n in zip(X0_cores, M, N)]

    # check if rmax is a list
    if isinstance(rmax, int):
        rmax = [1] + (d-1) * [rmax] + [1]

    # z cores
    rz = [1]+(d-1)*[kickrank+kick2]+[1]
    z_tt = torchtt.random([(m, n)
                          for m, n in zip(M, N)], rz, dtype, device=device)
    z_cores = [tn.reshape(c, [c.shape[0], -1, c.shape[-1]])
               for c in z_tt.cores]
    z_cores, rz = rl_orthogonal(z_cores, rz, False)
    z_cores = [tn.reshape(c, [c.shape[0], m, -1, c.shape[-1]])
               for c, m in zip(z_cores, M)]

    norms = np.zeros(d)
    Phiz = [tn.ones((1, 1), dtype=dtype, device=device)] + [None] * (d-1) + \
        [tn.ones((1, 1), dtype=dtype, device=device)]  # size is rzk x rxk
    Phiz_rhs = [tn.ones((1, 1, 1), dtype=dtype, device=device)] + [None] * (d-1) + \
        [tn.ones((1, 1, 1), dtype=dtype, device=device)
         ]   # size is rzk x rAk x rBk

    Phis = [tn.ones((1, 1), dtype=dtype, device=device)] + [None] * (d-1) + \
        [tn.ones((1, 1), dtype=dtype, device=device)]  # size is rk x rk
    Phis_rhs = [tn.ones((1, 1, 1), dtype=dtype, device=device)] + [None] * (d-1) + \
        [tn.ones((1, 1, 1), dtype=dtype, device=device)
         ]  # size is rk x rAk x rBk

    last = False

    normA = np.ones((d-1))
    normb = np.ones((d-1))
    normx = np.ones((d-1))
    nrmsc = 1.0

    if verbose:
        print('Starting AMEn multiplication with:\n\tepsilon: %g\n\tsweeps: %d' % (
            eps, nswp))
        print()

    for swp in range(nswp):
        # right to left orthogonalization

        if verbose:
            print()
            print('Starting sweep %d %s...' %
                  (swp+1, "(last one) " if last else ""))
            tme_sweep = datetime.datetime.now()

        tme = datetime.datetime.now()
        for k in range(d-1, 0, -1):

            # update the z part (ALS) update
            if not last:
                if swp > 0:
                    # shape rzp x MN x rz
                    czx = tn.einsum('zr,rmnR,ZR->zmnZ',
                                    Phiz[k], x_cores[k], Phiz[k+1])
                    # shape is rzp x MN x rz
                    czAB = _local_AB(
                        Phiz_rhs[k], Phiz_rhs[k+1], A_cores[k], B_cores[k], bandsA[k], bandsB[k])

                    cz_new = czAB*nrmsc - czx
                    _, _, vz = SVD(tn.reshape(cz_new, [cz_new.shape[0], -1]))
                    # truncate to kickrank
                    cz_new = vz[:min(kickrank, vz.shape[0]), :].t()
                    if k < d-1:  # extend cz_new with random elements
                        cz_new = tn.cat(
                            (cz_new, tn.randn((cz_new.shape[0], kick2),  dtype=dtype, device=device)), 1)
                else:
                    cz_new = tn.reshape(z_cores[k], [rz[k], -1]).t()

                qz, _ = QR(cz_new)
                rz[k] = qz.shape[1]
                z_cores[k] = tn.reshape(qz.t(), [rz[k], M[k], N[k], rz[k+1]])

            # norm correction ?
            if swp > 0:
                nrmsc = nrmsc * normA[k-1] * normx[k-1] / normb[k-1]

            core = tn.reshape(x_cores[k], [rx[k], M[k]*N[k]*rx[k+1]]).t()
            Qmat, Rmat = QR(core)

            core_prev = tn.einsum('ijlk,km->ijlm', x_cores[k-1], Rmat.T)
            rx[k] = Qmat.shape[1]

            current_norm = tn.linalg.norm(core_prev)
            if current_norm > 0:
                core_prev = core_prev / current_norm
            else:
                current_norm = 1.0
            normx[k-1] = normx[k-1]*current_norm

            x_cores[k] = tn.reshape(Qmat.t(), [rx[k], M[k], N[k], rx[k+1]])
            x_cores[k-1] = core_prev[:]

            # update phis (einsum)
            Phis[k] = _compute_phi_bck_x(Phis[k+1], x_cores[k], x_cores[k])
            Phis_rhs[k] = _compute_phi_AB('bck',
                                          Phis_rhs[k+1], A_cores[k], B_cores[k], x_cores[k], bandsA[k], bandsB[k])

            # ... and norms
            # norm = tn.linalg.norm(Phis[k])
            # norm = norm if norm > 0 else 1.0
            # normA[k-1] = norm
            # Phis[k] = Phis[k] / norm
            norm = tn.linalg.norm(Phis_rhs[k])
            norm = norm if norm > 0 else 1.0
            normb[k-1] = norm
            Phis_rhs[k] = Phis_rhs[k]/norm

            # norm correction
            nrmsc = nrmsc * normb[k-1] / (normA[k-1] * normx[k-1])

            # compute phis_z
            if not last:
                Phiz[k] = _compute_phi_bck_x(
                    Phiz[k+1], z_cores[k], x_cores[k]) / normA[k-1]
                Phiz_rhs[k] = _compute_phi_AB('bck',
                                              Phiz_rhs[k+1], A_cores[k], B_cores[k], z_cores[k], bandsA[k], bandsB[k]) / normb[k-1]

        # start loop
        max_dx = 0

        for k in range(d):
            if verbose:
                print('\tCore', k)
            previous_solution = x_cores[k]  # tn.reshape(x_cores[k], [-1, 1])

            # compute new approximation
            solution_now = _local_AB(
                Phis_rhs[k], Phis_rhs[k+1], A_cores[k], B_cores[k], bandsA[k], bandsB[k]) * nrmsc
            norm_solution = tn.linalg.norm(solution_now)

            # compute residual and step size
            dx = tn.linalg.norm(solution_now-previous_solution) / \
                tn.linalg.norm(solution_now)
            if verbose:
                print('\t\tdx = %g' % (dx))

            max_dx = max(dx, max_dx)

            solution_now = tn.reshape(solution_now, [rx[k]*M[k]*N[k], rx[k+1]])
            # truncation
            if k < d-1:
                u, s, v = SVD(solution_now)

                r = rank_chop(s.cpu().numpy(), (norm_solution.cpu()
                              * eps / (d**(0.5 if last else 1.5))).numpy())
                r = min([r, tn.numel(s), rmax[k+1]])
            else:
                u, v = QR(solution_now)
                # v = v.t()
                r = u.shape[1]
                s = tn.ones(r,  dtype=dtype, device=device)

            u = u[:, :r]
            v = tn.diag(s[:r]) @ v[:r, :]
            v = v.t()

            if not last:
                # shape rzp x MN x rz
                czx = tn.einsum(
                    'zr,rmnR,ZR->zmnZ', Phiz[k], tn.reshape(u@v.t(), [rx[k], M[k], N[k], rx[k+1]]), Phiz[k+1])
                # shape is rzp x MN x rz
                czAB = _local_AB(
                    Phiz_rhs[k], Phiz_rhs[k+1], A_cores[k], B_cores[k], bandsA[k], bandsB[k])
                # czAB_not_band = _local_AB(
                #     Phiz_rhs[k], Phiz_rhs[k+1], A_cores[k], B_cores[k])

                cz_new = czAB*nrmsc - czx

                uz, _, _ = SVD(tn.reshape(cz_new, [rz[k]*M[k]*N[k], rz[k+1]]))
                # truncate to kickrank
                cz_new = uz[:, :min(kickrank, uz.shape[1])]
                if k < d-1:  # extend cz_new with random elements
                    cz_new = tn.cat(
                        (cz_new, tn.randn((cz_new.shape[0], kick2),  dtype=dtype, device=device)), 1)

                qz, _ = QR(cz_new)
                rz[k+1] = qz.shape[1]
                z_cores[k] = tn.reshape(qz, [rz[k], M[k], N[k], rz[k+1]])

            if k < d-1:
                if not last:
                    # shape rzp x MN x rz
                    czx = tn.einsum(
                        'zr,rmnR,ZR->zmnZ', Phis[k], tn.reshape(u@v.t(), [rx[k], M[k], N[k], rx[k+1]]), Phiz[k+1])
                    # shape is rzp x MN x rz
                    czAB = _local_AB(
                        Phis_rhs[k], Phiz_rhs[k+1], A_cores[k], B_cores[k], bandsA[k], bandsB[k])

                    uk = czAB*nrmsc - czx

                    u, Rmat = QR(
                        tn.cat((u, tn.reshape(uk, [u.shape[0], -1])), 1))
                    r_add = uk.shape[-1]
                    v = tn.cat(
                        (v, tn.zeros([rx[k+1], r_add],  dtype=dtype, device=device)), 1)
                    v = v @ Rmat.t()

                r = u.shape[1]
                v = tn.einsum('ji,jklm->iklm', v, x_cores[k+1])
                # remove norm correction
                nrmsc = nrmsc * normA[k] * normx[k] / normb[k]

                norm_now = tn.linalg.norm(v)

                if norm_now > 0:
                    v = v / norm_now
                else:
                    norm_now = 1.0
                normx[k] = normx[k] * norm_now

                x_cores[k] = tn.reshape(u, [rx[k], M[k], N[k], r])
                x_cores[k+1] = tn.reshape(v, [r, M[k+1], N[k+1], rx[k+2]])
                rx[k+1] = r

                # next phis with norm correction
                Phis[k+1] = _compute_phi_fwd_x(Phis[k], x_cores[k], x_cores[k])
                Phis_rhs[k+1] = _compute_phi_AB('fwd',
                                                Phis_rhs[k], A_cores[k], B_cores[k], x_cores[k], bandsA[k], bandsB[k])

                # ... and norms
                # norm = tn.linalg.norm(Phis[k+1])
                # norm = norm if norm > 0 else 1.0
                # normA[k] = norm
                # Phis[k+1] = Phis[k+1] / norm
                norm = tn.linalg.norm(Phis_rhs[k+1])
                norm = norm if norm > 0 else 1.0
                normb[k] = norm
                Phis_rhs[k+1] = Phis_rhs[k+1] / norm

                # norm correction
                nrmsc = nrmsc * normb[k] / (normA[k] * normx[k])

                # next phiz
                if not last:
                    Phiz[k+1] = _compute_phi_fwd_x(Phiz[k],
                                                   z_cores[k], x_cores[k]) / normA[k]
                    Phiz_rhs[k+1] = _compute_phi_AB('fwd',
                                                    Phiz_rhs[k], A_cores[k], B_cores[k], z_cores[k], bandsA[k], bandsB[k]) / normb[k]
            else:
                x_cores[k] = tn.reshape(
                    u@tn.diag(s[:r]) @ v[:r, :].t(), [rx[k], M[k], N[k], rx[k+1]])

        if verbose:
            print('Solution rank is', rx)
            print('Maxdx ', max_dx)
            tme_sweep = datetime.datetime.now()-tme_sweep
            print('Time ', tme_sweep)

        if last:
            break

        if max_dx < eps:
            last = True

    if verbose:
        time_total = datetime.datetime.now() - time_total
        print()
        print('Finished after', swp+1, ' sweeps and ', time_total)
        print()
    normx = np.exp(np.sum(np.log(normx))/d)

    for k in range(d):
        x_cores[k] = x_cores[k] * normx

    if to_ttm:
        x = torchtt.TT(x_cores)
    else:
        x = torchtt.TT(
            [tn.reshape(c, [c.shape[0], c.shape[1], c.shape[-1]]) for c in x_cores])

    return x


def _compute_phi_bck_x(Phi_now, core_left, core_right):
    """
    Compute the phi backwards for the form dot(left,A @ right)

    Args:
        Phi_now (torch.tensor): The current phi. Has shape r1_k+1 x r2_k+1
        core_left (torch.tensor): the core on the left. Has shape r1_k x M_k x N_k x r1_k+1 
        core_right (torch.tensor): the core to the right. Has shape r2_k x M_k x N_k x r2_k+1 

    Returns:
        torch.tensor: The following phi (backward). Has shape r1_k x r2_k
    """

    Phi = oe.contract('LR,lmnL,rmnR->lr', Phi_now, core_left, core_right)

    return Phi


def _compute_phi_fwd_x(Phi_now, core_left, core_right):
    """
    Compute the phi forward for the form dot(left,A @ right)

    Args:
        Phi_now (torch.tensor): The current phi. Has shape r1_k x R_k x r2_k
        core_left (torch.tensor): the core on the left. Has shape r1_k x M_k x N_k x r1_k+1 
        core_right (torch.tensor): the core to the right. Has shape r2_k x M_k x N_k x r2_k+1 

    Returns:
        torch.tensor: The following phi (backward). Has shape r1_k+1 x r2_k+1
    """

    Phi_next = oe.contract('lr,lMNL,rMNR->LR', Phi_now, core_left, core_right)

    return Phi_next


def _compute_phi_AB(order, Phi_now, coreA, coreB, core, bandA=-1, bandB=-1):
    """


    Args:
        order(str): fwd - for forward phi or bck - for backward phi
        Phi_now (torch.tensor): The current phi. 
                                If order is fwd has shape r_k x rA_k x rB_k.
                                If order is bck has shape  r_k+1 x rA_k+1 x rB_k+1
        coreA (torch.tensor): The current core of the rhs. Has shape rA_k x M_k x K_k x rA_k+1
        coreB (torch.tensor): The current core of the rhs. Has shape rB_k x K_k x N_k x rB_k+1
        core (torch.tensor): The current core. Has shape r_k x M_k x N_k x r_k+1

    Returns:
        torch.tensor: If order is fwd: the forward phi corresponding to the rhs. Has shape r_k+1 x rA_k+1 x rB_k+1
                      If order is bck: the backward phi corresponding to the rhs. Has shape r_k x rA_k+1 x rB_k+1
    """
    if coreA.shape[1] != coreA.shape[2]:
        bandA = -1
    if coreB.shape[1] != coreB.shape[2]:
        bandB = -1
    if (bandA >= 0) and (bandB >= 0):
        if (bandA >= bandB):
            bandA = -1
        else:
            bandB = -1

    if bandA < 0:
        if bandB < 0:
            if order == 'bck':
                Phi = oe.contract('RAB,amkA,bknB,rmnR->rab',
                                  Phi_now, coreA, coreB, core)
            else:
                Phi = oe.contract('rab,amkA,bknB,rmnR->RAB',
                                  Phi_now, coreA, coreB, core)
        else:
            if order == 'bck':
                sizes = 'RAB,amkA,lbBk,lrmkR->rab'
            else:
                sizes = 'rab,amkA,lbBk,lrmkR->RAB'
            diagonals_B = tn.stack([tnf.pad(tn.diagonal(coreB, i, 1, 2), (-i, 0))for i in range(-bandB, 0)] +
                                   [tnf.pad(tn.diagonal(coreB, i, 1, 2), (0, i)) for i in range(0, bandB+1)])
            cores = tn.stack([tnf.pad(core[:, :, :i, :], (0, 0, -i, 0)) for i in range(-bandB, 0)] +
                             [tnf.pad(core[:, :, i:, :], (0, 0, 0, i)) for i in range(0, bandB+1)])
            Phi = oe.contract(sizes, Phi_now, coreA, diagonals_B, cores)
    else:
        if order == 'bck':
            sizes = 'RAB,laAk,bknB,lrknR->rab'
        else:
            sizes = 'rab,laAk,bknB,lrknR->RAB'
        diagonals_A = tn.stack([tnf.pad(tn.diagonal(coreA, i, 1, 2), (0, -i))for i in range(-bandA, 0)] +
                               [tnf.pad(tn.diagonal(coreA, i, 1, 2), (i, 0)) for i in range(0, bandA+1)])
        cores = tn.stack([tnf.pad(core[:, -i:, :, :], (0, 0, 0, 0, 0, -i)) for i in range(-bandA, 1)] +
                         [tnf.pad(core[:, :-i, :, :], (0, 0, 0, 0, i, 0)) for i in range(1, bandA+1)])
        Phi = oe.contract(sizes, Phi_now, diagonals_A, coreB, cores)
    return Phi


# def _compute_phi_AB(order, Phi_now, coreA, coreB, core, bandA=-1, bandB=-1):
#     """


#     Args:
#         order(str): fwd - for forward phi or bck - for backward phi
#         Phi_now (torch.tensor): The current phi.
#                                 If order is fwd has shape r_k x rA_k x rB_k.
#                                 If order is bck has shape  r_k+1 x rA_k+1 x rB_k+1
#         coreA (torch.tensor): The current core of the rhs. Has shape rA_k x M_k x K_k x rA_k+1
#         coreB (torch.tensor): The current core of the rhs. Has shape rB_k x K_k x N_k x rB_k+1
#         core (torch.tensor): The current core. Has shape r_k x M_k x N_k x r_k+1

#     Returns:
#         torch.tensor: If order is fwd: the forward phi corresponding to the rhs. Has shape r_k+1 x rA_k+1 x rB_k+1
#                       If order is bck: the backward phi corresponding to the rhs. Has shape r_k x rA_k+1 x rB_k+1
#     """
#     #print("last var")
#     if order == 'bck':
#         Phi = tn.zeros(core.shape[0], coreA.shape[0], coreB.shape[0], dtype=coreA.dtype, device=coreA.device)
#     elif order == 'fwd':
#         Phi = tn.zeros(core.shape[-1], coreA.shape[-1], coreB.shape[-1], dtype=coreA.dtype, device=coreA.device)
#     else:
#         assert(0)

#     if coreA.shape[1] != coreA.shape[2]:
#         bandA = -1
#     if coreB.shape[1] != coreB.shape[2]:
#         bandB = -1

#     if bandA < 0:
#         if bandB < 0:
#             if order == 'bck':
#                 Phi = oe.contract('RAB,amkA,bknB,rmnR->rab', Phi_now, coreA, coreB, core)
#             else:
#                 Phi = oe.contract('rab,amkA,bknB,rmnR->RAB', Phi_now, coreA, coreB, core)
#         else:
#             if order == 'bck':
#                 sizes = 'RAB,amkA,bBk,rmkR->rab'
#             else:
#                 sizes = 'rab,amkA,bBk,rmkR->RAB'
#             for i in range(-bandB, bandB+1):
#                 diag_B = tn.diagonal(coreB, i, 1, 2)
#                 if i < 0:
#                     Phi += oe.contract(sizes, Phi_now, coreA[:, :, -i:, :], diag_B, core[:, :, :i, :])
#                 elif i > 0:
#                     Phi += oe.contract(sizes, Phi_now, coreA[:, :, :-i, :], diag_B, core[:, :, i:, :])
#                 else:
#                     Phi += oe.contract(sizes, Phi_now, coreA, diag_B, core)
#     else:
#         if bandB < 0:
#             if order == 'bck':
#                 sizes = 'RAB,aAk,bknB,rknR->rab'
#             else:
#                 sizes = 'rab,aAk,bknB,rknR->RAB'
#             for i in range(-bandA, bandA+1):
#                 diag_A = tn.diagonal(coreA, i, 1, 2)
#                 if i < 0:
#                     Phi += oe.contract(sizes, Phi_now, diag_A, coreB[:, :i , :, :], core[:, -i:, :, :])
#                 elif i > 0:
#                     Phi += oe.contract(sizes, Phi_now, diag_A, coreB[:, i:, :, :], core[:, :-i, :, :])
#                 else:
#                     Phi += oe.contract(sizes, Phi_now, diag_A, coreB, core)
#         else:
#             band = min(bandA + bandB, coreA.shape[1] - 1)
#             band_min = min(bandA, bandB)
#             if order == 'bck':
#                 sizes = 'RAB,aAk,bBk,rRk->rab'
#             else:
#                 sizes = 'rab,aAk,bBk,rRk->RAB'
#             for k in range(-band, band+1):
#                 diag_core = tn.diagonal(core, k, 1, 2)
#                 if k > 0:
#                     b = bandA - k
#                     d = bandB - k
#                     for l in [*range(-min(d, bandA), 0),  *range(max(0, k-bandB), min(k, bandA)+1), *range(k+1, k+min(b, bandB)+1)]:
#                         diag_A = tn.diagonal(coreA, l, 1, 2)
#                         diag_B = tn.diagonal(coreB, k-l, 1, 2)
#                         if l < k:
#                             diag_A = diag_A[:, :, :max(l, 0)-k]
#                         diag_B = diag_B[:, :, min(k, max(l, 0)):]
#                         diag_core_slice = diag_core[:, :, max(-l, 0):]
#                         if l > k:
#                             diag_core_slice = diag_core_slice[:, :, :k-l]
#                         Phi += oe.contract(sizes, Phi_now, diag_A, diag_B, diag_core_slice)
#                 elif k < 0:
#                     d = bandB + k
#                     b = bandA + k
#                     for l in [*range(min(k-d, -bandA), k), *range(max(k, -bandB), min(0, k+bandA)+1), *range(1, min(b, bandB)+1)]:
#                         diag_B = tn.diagonal(coreB, l, 1, 2)
#                         diag_A = tn.diagonal(coreA, k-l, 1, 2)
#                         diag_A = diag_A[:, :, max(-max(l, k), 0):]
#                         if l > k:
#                             diag_B = diag_B[:, :, :k-min(l, 0)]
#                         diag_core_slice = diag_core[:, :, max(0, l):]
#                         if l < k:
#                             diag_core_slice = diag_core_slice[:, :, :l-k]
#                         Phi += oe.contract(sizes, Phi_now, diag_A, diag_B, diag_core_slice)
#                 else:
#                     for l in range(-band_min, band_min+1):
#                         diag_A = tn.diagonal(coreA, l, 1, 2)
#                         diag_B = tn.diagonal(coreB, -l, 1, 2)
#                         if l > 0:
#                             Phi += oe.contract(sizes, Phi_now, diag_A, diag_B, diag_core[:, :, :-l])
#                         else:
#                             Phi += oe.contract(sizes, Phi_now, diag_A, diag_B, diag_core[:, :, -l:])
#     return Phi
