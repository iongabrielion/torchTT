"""
System solvers in the TT format.

"""

import torch as tn
import numpy as np
import torchtt
import datetime
from torchtt._decomposition import QR, SVD, lr_orthogonal, rl_orthogonal
from torchtt._iterative_solvers import BiCGSTAB_reset, gmres_restart
import opt_einsum as oe
from .errors import *
import torch.nn.functional as tnf

import time

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

def _local_product(Phi_right, Phi_left, Summ_data, x, band_max):
    """
    Compute local matvec product
    
    Args:
     Phi (torch.tensor): right tensor of shape r x R x r.
     Psi (torch.tensor): left tensor of shape lp x Rp x lp.
     coreA (torch.tensor): current core of A, shape is rp x N x N x r.
     x (torch.tensor): the current core of x, shape is rp x N x r.
     bandA (int): if positive specifies number of diagonals in the matrix. 0 means diagonal structure, 1 means tridiagonal, ...
    
    Returns:
     torch.tensor: the result.
    """
    w = 0
    for data, band, s_beg, s_end, S_beg, S_end in Summ_data:
        if band < 0:
            # data is a full core
            w += oe.contract('lsr,smnS,LSR,rnR->lmL',
                              Phi_left[:, s_beg:s_end, :], data, Phi_right[:, S_beg:S_end, :], x)
        else:
            # data is diagonals of a core
            tmp = oe.contract('lsr,ksSm,LSR,rmR->klmL', 
                              Phi_left[:, s_beg:s_end, :], data, Phi_right[:, S_beg:S_end, :], x)
            w += tmp[band, ...]
            for i in range(1, band+1):
                w[:, :-i, :] += tmp[i + band, :, i:, :]
                w[:, i:, :] += tmp[-i + band, :, :-i, :]
    return w


class _LinearOp():
    def __init__(self, Phi_left, Phi_right, Summ_data, A_shape, shape, prec, band_max):
        self.Phi_left = Phi_left
        self.Phi_right = Phi_right
        self.A_shape = A_shape
        self.shape = shape
        self.prec = prec
        self.summ_data = Summ_data
        self.band_max = band_max
            
        if prec == 'c':
            Jl = tn.zeros(Phi_left.shape[-1], A_shape[1], A_shape[2], A_shape[-1], dtype=Phi_left.dtype, device=Phi_left.device)
            for data, band, s_beg, s_end, S_beg, S_end in Summ_data:
                if band < 0:
                    Jl[..., S_beg:S_end] += tn.einsum('sd,smnS->dmnS',tn.diagonal(Phi_left[:, s_beg:s_end, :], 0, 0, 2), data)
                else:
                    diags_Jl = tn.einsum('sd,ksSn->kdSn', tn.diagonal(Phi_left[:, s_beg:s_end, :], 0, 0, 2), data)
                    for i in range(-band, band+1):
                        diag_i_Jl = tn.diagonal(Jl, i, 1, 2)
                        if i > 0:
                            diag_i_Jl[:, S_beg:S_end, :] += diags_Jl[i + band, :, :, :-i]
                        else:
                            diag_i_Jl[:, S_beg:S_end, :] += diags_Jl[i + band, :, :, -i:]
            Jr = tn.diagonal(Phi_right, 0, 0, 2)
            self.J = tn.einsum('dmnS,SD->dDmn', Jl, Jr)
            self.J = tn.linalg.inv(self.J)
            
            self.contraction = (shape[0]*shape[1]*shape[2] > 1e5)
            
        elif prec == 'r':
            Jl = tn.zeros(Phi_left.shape[-1], A_shape[1], A_shape[2], A_shape[-1], dtype=Phi_left.dtype, device = Phi_left.device)
            for data, band, s_beg, s_end, S_beg, S_end in Summ_data:
                if band < 0:
                    Jl[:, :, :, S_beg:S_end] += tn.einsum('sd,smnS->dmnS', tn.diagonal(Phi_left[:, s_beg:s_end, :], 0, 0, 2), data)
                else:
                    diags_Jl = tn.einsum('sd,ksSn->kdSn', tn.diagonal(Phi_left[:, s_beg:s_end, :], 0, 0, 2), data)
                    for i in range(-band, band+1):
                        diag_i_Jl = tn.diagonal(Jl, i, 1, 2)
                        if i > 0:
                            diag_i_Jl[:, S_beg:S_end, :] += diags_Jl[i + band, :, :, :-i]
                        else:
                            diag_i_Jl[:, S_beg:S_end, :] += diags_Jl[i + band, :, :, -i:]
            self.J = tn.einsum('dmnS,LSR->dmLnR', Jl, Phi_right)
            sh = self.J.shape
            self.J = tn.reshape(self.J, [-1, self.J.shape[1]*self.J.shape[2], self.J.shape[3]*self.J.shape[4]])
            self.J = tn.reshape(tn.linalg.inv(self.J), sh)
                
            #self.contraction = (shape[0]*shape[1]*shape[2] > 2*1e4)
            self.contraction=False
        else:
            pass

    def apply_prec(self, x):
        if self.prec == 'c':
            # no improvement using opt_einsum
            y = tn.einsum('rRmn, rnR->rmR', self.J, x)
            return y
        elif self.prec == 'r':
            y = tn.einsum('rnR,rmLnR->rmL', x, self.J)
            return y.reshape(x.shape)

    def matvec(self, x, apply_prec=True):
        x = x.reshape(self.shape)
        if self.prec == None or not apply_prec:
            #w = tn.einsum('lsr,smnS,LSR,rnR->lmL', self.Phi_left, self.coreA, self.Phi_right, x)
            Phi_left_x = tn.tensordot(x, self.Phi_left, ([0], [2])) # rnR,lsr-> nRls
            w = tn.zeros(self.Phi_right.shape[-1], self.Phi_left.shape[0], self.A_shape[1], self.A_shape[-1], dtype=x.dtype, device=x.device)
            for data, band, s_beg, s_end, S_beg, S_end in self.summ_data:
                if band >= 0:
                    # data is diagonals of core
                    tmp = tn.einsum('nRls, ksSn -> kRlnS', Phi_left_x[..., s_beg:s_end], data)
                    w[..., S_beg:S_end] += tmp[band, ...]
                    for i in range(1, band+1):
                        w[:, :, :-i, S_beg:S_end] += tmp[i + band, :, :, i:, :]
                        w[:, :, i:, S_beg:S_end] += tmp[-i + band, :, :, :-i, :]
                else:
                    # data is a full core
                    w [:, :, :, S_beg:S_end] += tn.tensordot(Phi_left_x[..., s_beg:s_end], data, ([0, 3], [2, 0])) # nRls,smnS->RlmS
            w = tn.tensordot(w, self.Phi_right, ([0, 3], [2, 1]))  # RlmS,LSR->lmL

        elif self.prec == 'c' or self.prec == 'r':
            if self.contraction:
                w = 0
                for data, band, s_beg, s_end, S_beg, S_end in self.summ_data:
                    if band < 0:
                        w += oe.contract('lsr,smnS,LSR,rab,rnRab->lmL' if self.prec == 'r' else 'lsr,smnS,LSR,raR,rRna->lmL', 
                                         self.Phi_left[:, s_beg:s_end, :], data, self.Phi_right[:, S_beg:S_end, :], x, self.J)
                    else:
                        tmp = oe.contract('lsr,ksSn,LSR,rab,rnRab->lnL' if self.prec == 'r' else 'lsr,ksSn,LSR,raR,rRna->klnL',
                                               self.Phi_left[:, s_beg:s_end, :], data, self.Phi_right[:, S_beg:S_end, :], x, self.J)
                        w += tmp[band, ...]
                        for i in range(1, band+1):
                            w[:, :-i, :] += tmp[i + band, :, i:, :]
                            w[:, i:, :] += tmp[i + band, :, :-i, :]
            else:
                y = self.apply_prec(x)
                #w = tn.einsum('lsr,smnS,LSR,rnR->lmL', self.Phi_left, self.coreA, self.Phi_right, y)
                
                Phi_left_y = tn.tensordot(y, self.Phi_left, ([0], [2])) # rnR,lsr-> nRls
                w = tn.zeros(self.Phi_right.shape[-1], self.Phi_left.shape[0], self.A_shape[1], self.A_shape[-1], dtype=x.dtype, device=x.device)
                for data, band, s_beg, s_end, S_beg, S_end in self.summ_data:
                    if band >= 0:
                        # data is diagonals of core
                        tmp = tn.einsum('nRls, ksSn -> kRlnS', Phi_left_y[..., s_beg:s_end], data)
                        w[..., S_beg:S_end] += tmp[band, ...]
                        for i in range(1, band+1):
                            w[:, :, :-i, S_beg:S_end] += tmp[i + band, :, :, i:, :]
                            w[:, :, i:, S_beg:S_end] += tmp[-i + band, :, :, :-i, :]
                    else:
                        # data is a full core
                        w [..., S_beg:S_end] += tn.tensordot(Phi_left_y[..., s_beg:s_end], data, ([0, 3], [2, 0])) # nRls,smnS->RlmS
                w = tn.tensordot(w, self.Phi_right, ([0, 3], [2, 1]))  # RlmS,LSR->lmL

        else:
            raise Exception('Preconditioner '+str(self.prec)+' not defined.')
        return tn.reshape(w, [-1, 1])

def amen_solve(Matrices, b, nswp=22, x0=None, eps=1e-10, rmax=32768, max_full=500, kickrank=4, kick2=0, trunc_norm='res', local_solver=1, local_iterations=40, resets=2, verbose=False, preconditioner=None, use_cpp=True, use_single_precision=False, bandsMatrices=None):
    """
    Solve a multilinear system :math:`\\mathsf{Ax} = \\mathsf{b}` in the Tensor Train format.

    This method implements the algorithm from `Sergey V Dolgov, Dmitry V Savostyanov, Alternating minimal energy methods for linear systems in higher dimensions <https://epubs.siam.org/doi/abs/10.1137/140953289>`_.

    Example:

        .. code-block:: python

            import torchtt
            A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1]) # create random matrix
            x = torchtt.random([4,5,6],[1,2,3,1]) # invent a random solution
            b = A @ x # compute the rhs
            xx = torchtt.solvers.amen_solve(A,b) # solve
            print((xx-x).norm()/x.norm()) # error


    Args:
        A (torchtt.TT): the system matrix in TT.
        b (torchtt.TT): the right hand side in TT.
        nswp (int, optional): number of sweeps. Defaults to 22.
        x0 (torchtt.TT, optional): initial guess. In None is provided the initial guess is a ones tensor. Defaults to None.
        eps (float, optional): relative residual. Defaults to 1e-10.
        rmax (int, optional): maximum rank. Defaults to 100000.
        max_full (int, optional): the maximum size of the core until direct solver is used for the local subproblem. Defaults to 500.
        kickrank (int, optional): rank enrichment. Defaults to 4.
        kick2 (int, optional): [description]. Defaults to 0.
        trunc_norm (str, optional): [description]. Defaults to 'res'.
        local_solver (int, optional): choose local iterative solver: 1 for GMRES and 2 for BiCGSTAB. Defaults to 1.
        local_iterations (int, optional): number of GMRES iterations for the local subproblems. Defaults to 40.
        resets (int, optional): number of resets in the GMRES. Defaults to 2.
        verbose (bool, optional): choose whether to display or not additional information during the runtime. Defaults to True.
        preconditioner (string, optional): Choose the preconditioner for the local system. Possible values are None, 'c' (central Jacobi preconditioner). No preconditioner is used if None is provided. Defaults to None.
        use_cpp (bool, optional): use the C++ implementation of AMEn. Defaults to True.
        bandsA (list, optional): list of bands for the TT cores of the matrix. Defaults to None.

    Raises:
        InvalidArguments: A and b must be TT instances.
        InvalidArguments: Invalid preconditioner.
        IncompatibleTypes: A must be TT-matrix and b must be vector.
        ShapeMismatch: A is not quadratic.
        ShapeMismatch: Dimension mismatch.

    Returns:
        torchtt.TT: the approximation of the solution in TT format.
    """
    
    # A = Matrices[0] + ... + Matrices[-1]
    
    # perform checks of the input data
    if not isinstance(Matrices, list):
        Matrices = [Matrices, ]
        bandsMatrices = [bandsMatrices,]
    for i in range(len(Matrices)):
        A = Matrices[i]
        if not isinstance(A, torchtt.TT):
             raise InvalidArguments(f'Matrix {i+1}-th must a TT instance.')
        if not Matrices[i].is_ttm:
            raise IncompatibleTypes(f'Matrix {i+1}-th must be a TT-matrix.')
        if A.M != A.N:
            raise ShapeMismatch(f'Matrix {i+1}-th is not quadratic.')
        if i > 0:
            if A.M != Matrices[i-1].M:
                raise ShapeMismatch('Matrix shapes must be the same.')
        
    if not isinstance(b, torchtt.TT):
        raise InvalidArguments('b must be a TT instance.')
    if b.is_ttm:
        raise IncompatibleTypes(' b must be a vector.')
    if Matrices[-1].N != b.N:
        raise ShapeMismatch('Dimension mismatch.')

    if use_cpp and _flag_use_cpp:
        A = Matrices[0]
        for i in range(1, len(Matrices)):
            A += Matrices[i]
        if x0 == None:
            x_cores = []
            x_R = [1]*(1+len(A.N))
        else:
            x_cores = x0.cores
            x_R = x0.R
        if preconditioner == None:
            prec = 0
        elif preconditioner == 'c':
            prec = 1
        elif preconditioner == 'r':
            prec = 2
        else:
            raise InvalidArguments("Invalid preconditioner.")
        cores = torchttcpp.amen_solve(A.cores, b.cores, x_cores, b.N, A.R, b.R, x_R, nswp,
                                      eps, rmax, max_full, kickrank, kick2, local_iterations, resets, verbose, prec)
        return torchtt.TT(list(cores))
        return -1
    else:
        if bandsMatrices == None:
            bandsMatrices = [[-1] * len(Matrices[0].N)] * len(Matrices)
        else:
            assert(len(Matrices) == len(bandsMatrices))
            for i in range(len(Matrices)):
                if bandsMatrices[i] == None:
                    bandsMatrices = [[-1] * len(Matrices[0].N)] * len(Matrices)
                    break
                
        return _amen_solve_python(Matrices, b, nswp, x0, eps, rmax, max_full, kickrank, kick2, trunc_norm, local_solver, local_iterations, resets, verbose, preconditioner, use_single_precision, np.array(bandsMatrices))


def _amen_solve_python(Matrices, b, nswp=22, x0=None, eps=1e-10, rmax=1024, max_full=500, kickrank=4, kick2=0, trunc_norm='res', local_solver=1, local_iterations=40, resets=2, verbose=False, preconditioner=None, use_single_precision=False, bandsMatrices=None):
    N = b.N
    d = len(N)

    Summ = Matrices[0]
    for i in range(1, len(Matrices)):
        Summ += Matrices[i]
            
    Summ_data = []
    for k in range(d):
        k_th_cores = []
        s_begs = [0] * len(Matrices)
        s_ends = [1] * len(Matrices)
        S_begs = [0] * len(Matrices)
        S_ends = [1] * len(Matrices)
        for i in range((len(Matrices))):
            core = Matrices[i].cores[k]
            s_ends[i] = s_begs[i] + core.shape[0]
            S_ends[i] = S_begs[i] + core.shape[-1]
            if (i < len(Matrices) - 1):
                if k != 0:
                    s_begs[i+1] = s_ends[i]
                if k != (d - 1):
                    S_begs[i+1] = S_ends[i]
            band = bandsMatrices[i, k]
            if band < 0:
                k_th_cores.append(core)
            else:
                k_th_cores.append(tn.stack([tnf.pad(tn.diagonal(core, i, 1, 2), (0, -i)) for i in range(-band, 0)] + 
                                           [tnf.pad(tn.diagonal(core, i, 1, 2), (i, 0)) for i in range(0, band+1)]))
        Summ_data.append(list(zip(k_th_cores, bandsMatrices[:, k], s_begs, s_ends, S_begs, S_ends)))

        
    A_ranks = np.array(Matrices[0].R)
    for i in range(1, len(Matrices)):
        A_ranks[1:-1] += np.array(Matrices[i].R)[1:-1]
    A_ranks = list(A_ranks)
    A_sizes = (Matrices[0].N).copy()
                    
    
    if verbose:
        time_total = datetime.datetime.now()

    dtype = b.cores[0].dtype
    device = b.cores[0].device
    rank_search = 1  # binary rank search
    damp = 2

    if x0 == None:
        x = torchtt.ones(b.N, dtype=dtype, device=device)
    else:
        x = x0

    # kkt = torchttcpp.amen_solve(A.cores, b.cores, x.cores, b.N, A.R, b.R, x.R, nswp, eps, rmax, max_full, kickrank, kick2, local_iterations, resets, verbose, 0)
    x_cores = x.cores.copy()
    rx = x.R.copy()
    # check if rmax is a list
    if isinstance(rmax, int):
        rmax = [1] + (d-1) * [rmax] + [1]

    # z cores
    rz = [1]+(d-1)*[kickrank+kick2]+[1]
    z_tt = torchtt.random(N, rz, dtype, device=device)
    z_cores = z_tt.cores
    z_cores, rz = rl_orthogonal(z_cores, rz, False)

    norms = np.zeros(d)
    Phiz = [tn.ones((1, 1, 1), dtype=dtype, device=device)] + [None] * (d-1) + \
        [tn.ones((1, 1, 1), dtype=dtype, device=device)
         ]  # size is rzk x Rk x rxk
    Phiz_b = [tn.ones((1, 1), dtype=dtype, device=device)] + [None] * (d-1) + \
        [tn.ones((1, 1), dtype=dtype, device=device)]   # size is rzk x rzbk

    Phis = [tn.ones((1, 1, 1), dtype=dtype, device=device)] + [None] * (d-1) + \
        [tn.ones((1, 1, 1), dtype=dtype, device=device)
         ]  # size is rk x Rk x rk
    Phis_b = [tn.ones((1, 1), dtype=dtype, device=device)] + [None] * (d-1) + \
        [tn.ones((1, 1), dtype=dtype, device=device)]  # size is rk x rbk

    last = False

    normA = np.ones((d-1))
    normb = np.ones((d-1))
    normx = np.ones((d-1))
    nrmsc = 1.0

    if verbose:
        print('Starting AMEn solve with:\n\tepsilon: %g\n\tsweeps: %d\n\tlocal iterations: %d\n\tresets: %d\n\tpreconditioner: %s' % (
            eps, nswp, local_iterations, resets, str(preconditioner)))
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
                    # shape rzp x N x rz
                    czA = _local_product(Phiz[k+1], Phiz[k], Summ_data[k], x_cores[k], np.max(bandsMatrices[:, k]))
                    # shape is rzp x N x rz
                    czy = tn.einsum('br,bnB,BR->rnR',
                                    Phiz_b[k], b.cores[k], Phiz_b[k+1])
                    cz_new = czy*nrmsc - czA
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
                z_cores[k] = tn.reshape(qz.t(), [rz[k], N[k], rz[k+1]])

            # norm correction ?
            if swp > 0:
                nrmsc = nrmsc * normA[k-1] * normx[k-1] / normb[k-1]

            core = tn.reshape(x_cores[k], [rx[k], N[k]*rx[k+1]]).t()
            Qmat, Rmat = QR(core)

            core_prev = tn.einsum('ijk,km->ijm', x_cores[k-1], Rmat.T)
            rx[k] = Qmat.shape[1]

            current_norm = tn.linalg.norm(core_prev)
            if current_norm > 0:
                core_prev = core_prev / current_norm
            else:
                current_norm = 1.0
            normx[k-1] = normx[k-1]*current_norm

            x_cores[k] = tn.reshape(Qmat.t(), [rx[k], N[k], rx[k+1]])
            x_cores[k-1] = core_prev[:]

            # update phis (einsum)
            Phis[k] = _compute_phi_A('bck', Phis[k+1], x_cores[k], Summ_data[k], x_cores[k], A_ranks[k], A_ranks[k+1],
                                     np.max(bandsMatrices[:, k]))
            Phis_b[k] = _compute_phi_bck_rhs(Phis_b[k+1], b.cores[k], x_cores[k])

            # ... and norms
            norm = tn.linalg.norm(Phis[k])
            norm = norm if norm > 0 else 1.0
            normA[k-1] = norm
            Phis[k] = Phis[k] / norm
            norm = tn.linalg.norm(Phis_b[k])
            norm = norm if norm > 0 else 1.0
            normb[k-1] = norm
            Phis_b[k] = Phis_b[k]/norm

            # norm correction
            nrmsc = nrmsc * normb[k-1] / (normA[k-1] * normx[k-1])

            # compute phis_z
            if not last:
                Phiz[k] = _compute_phi_A('bck', Phiz[k+1], z_cores[k], Summ_data[k], 
                                         x_cores[k], A_ranks[k], A_ranks[k+1], 
                                         np.max(bandsMatrices[:, k])) / normA[k-1]
                Phiz_b[k] = _compute_phi_bck_rhs(
                    Phiz_b[k+1], b.cores[k], z_cores[k]) / normb[k-1]

        # start loop
        max_res = 0
        max_dx = 0

        for k in range(d):
            if verbose:
                print('\tCore', k)
            previous_solution = tn.reshape(x_cores[k], [-1, 1])

            # assemble rhs
            rhs = tn.einsum('br,bmB,BR->rmR',
                            Phis_b[k], b.cores[k] * nrmsc, Phis_b[k+1])
            rhs = tn.reshape(rhs, [-1, 1])
            norm_rhs = tn.linalg.norm(rhs)

            # residuals
            real_tol = (eps/np.sqrt(d))/damp

            # solve the local system
            use_full = rx[k]*N[k]*rx[k+1] < max_full
            if use_full:
                # solve the full system
                if verbose:
                    print('\t\tChoosing direct solver (local size %d)....' %
                          (rx[k]*N[k]*rx[k+1]))
                # shape is Rp x N x N x r x r
                Bp = tn.zeros(A_ranks[k], A_sizes[k], A_sizes[k], Phis[k+1].shape[-1], Phis[k+1].shape[0], dtype=dtype, device=device)
                for data, band, s_beg, s_end, S_beg, S_end in Summ_data[k]:
                    if band < 0:
                        Bp[s_beg:s_end, ...] += tn.einsum('smnS,LSR->smnRL', data, Phis[k+1][:, S_beg:S_end, :])
                    else:
                        diags_Bp = tn.einsum('ksSn,LSR->ksRLn', data, Phis[k+1][:, S_beg:S_end, :])
                        for i in range(-band, band+1):
                            diag_i_Bp = tn.diagonal(Bp[s_beg:s_end, ...], i, 1, 2)
                            if i >= 0:
                                diag_i_Bp += diags_Bp[i + band, :, :, :, i:]
                            else:
                                diag_i_Bp += diags_Bp[i + band, :, :, :, :i]
                B = tn.einsum('lsr,smnRL->lmLrnR', Phis[k], Bp)
                B = tn.reshape(B, [rx[k]*N[k]*rx[k+1], rx[k]*N[k]*rx[k+1]])

                solution_now = tn.linalg.solve(B, rhs)

                res_old = tn.linalg.norm(B@previous_solution-rhs)/norm_rhs
                res_new = tn.linalg.norm(B@solution_now-rhs)/norm_rhs
            else:
                # iterative solver
                if verbose:
                    print('\t\tChoosing iterative solver %s (local size %d)....' % (
                        'GMRES' if local_solver == 1 else 'BiCGSTAB_reset', rx[k]*N[k]*rx[k+1]))
                    time_local = datetime.datetime.now()
                shape_now = [rx[k], N[k], rx[k+1]]

                if use_single_precision:
                    Op = _LinearOp(Phis[k], Phis[k+1],
                                   Summ_data[k], (A_ranks[k], A_sizes[k], A_sizes[k], A_ranks[k+1]),
                                   shape_now, preconditioner, np.max(bandsMatrices[:, k]))

                    # solution_now, flag, nit, res_new = BiCGSTAB_reset(Op, rhs,previous_solution[:], eps_local, local_iterations)
                    eps_local = real_tol * norm_rhs
                    drhs = Op.matvec(previous_solution.to(tn.float32), False)
                    drhs = rhs.to(tn.float32)-drhs
                    eps_local = eps_local / tn.linalg.norm(drhs)
                    if local_solver == 1:
                        solution_now, flag, nit = gmres_restart(Op, drhs, previous_solution.to(
                            tn.float32)*0, rhs.shape[0], local_iterations+1, eps_local, resets)
                    elif local_solver == 2:
                        solution_now, flag, nit, _ = BiCGSTAB_reset(
                            Op, drhs, previous_solution.to(tn.float32)*0, eps_local, local_iterations)
                    else:
                        raise InvalidArguments('Solver not implemented.')

                    if preconditioner != None:
                        solution_now = Op.apply_prec(
                            tn.reshape(solution_now, shape_now))
                        solution_now = tn.reshape(solution_now, [-1, 1])

                    solution_now = previous_solution + solution_now.to(dtype)
                    res_old = tn.linalg.norm(Op.matvec(previous_solution.to(
                        tn.float32), False).to(dtype)-rhs)/norm_rhs
                    res_new = tn.linalg.norm(Op.matvec(solution_now.to(
                        tn.float32), False).to(dtype)-rhs)/norm_rhs
                else:
                    Op = _LinearOp(Phis[k], Phis[k+1],
                                   Summ_data[k], (A_ranks[k], A_sizes[k], A_sizes[k], A_ranks[k+1]),
                                   shape_now, preconditioner, np.max(bandsMatrices[:, k]))

                    # solution_now, flag, nit, res_new = BiCGSTAB_reset(Op, rhs,previous_solution[:], eps_local, local_iterations)
                    eps_local = real_tol * norm_rhs
                    drhs = Op.matvec(previous_solution, False)
                    drhs = rhs-drhs
                    eps_local = eps_local / tn.linalg.norm(drhs)
                    if local_solver == 1:
                        solution_now, flag, nit = gmres_restart(
                            Op, drhs, previous_solution*0, rhs.shape[0], local_iterations+1, eps_local, resets)
                    elif local_solver == 2:
                        solution_now, flag, nit, _ = BiCGSTAB_reset(
                            Op, drhs, previous_solution*0, eps_local, local_iterations)
                    else:
                        raise InvalidArguments('Solver not implemented.')

                    if preconditioner != None:
                        solution_now = Op.apply_prec(
                            tn.reshape(solution_now, shape_now))
                        solution_now = tn.reshape(solution_now, [-1, 1])

                    solution_now = previous_solution + solution_now
                    res_old = tn.linalg.norm(
                        Op.matvec(previous_solution, False)-rhs) / norm_rhs
                    res_new = tn.linalg.norm(
                        Op.matvec(solution_now, False)-rhs) / norm_rhs

                if verbose:
                    print('\t\tFinished with flag %d after %d iterations with relres %g (from %g)' % (
                        flag, nit, res_new, real_tol * norm_rhs))
                    time_local = datetime.datetime.now() - time_local
                    print('\t\tTime needed ', time_local)
            # residual damp check
            if res_old/res_new < damp and res_new > real_tol:
                if verbose:
                    print('WARNING: residual increases. res_old %g, res_new %g, real_tol %g' % (
                        res_old, res_new, real_tol))  # warning (from tt toolbox)

            # compute residual and step size
            dx = tn.linalg.norm(solution_now - previous_solution) / \
                tn.linalg.norm(solution_now)
            if verbose:
                print('\t\tdx = %g, res_now = %g, res_old = %g' %
                      (dx, res_new, res_old))

            max_dx = max(dx, max_dx)
            max_res = max(max_res, res_old)

            solution_now = tn.reshape(solution_now, [rx[k]*N[k], rx[k+1]])
            # truncation
            if k < d-1:
                u, s, v = SVD(solution_now)
                if trunc_norm == 'fro':
                    pass
                else:
                    # search for a rank such that offeres small enough residuum
                    # TODO: binary search?
                    r = 0
                    for r in range(u.shape[1]-1, 0, -1):
                        # solution has the same size
                        solution = u[:, :r] @ tn.diag(s[:r]) @ v[:r, :]
                        # res = tn.linalg.norm(tn.reshape(local_product(Phis[k+1],Phis[k],A.cores[k],tn.reshape(solution,[rx[k],N[k],rx[k+1]]),solution_now.shape),[-1,1]) - rhs)/norm_rhs

                        if use_full:
                            res = tn.linalg.norm(
                                B@tn.reshape(solution, [-1, 1])-rhs)/norm_rhs
                        else:
                            # res = tn.linalg.norm(tn.reshape(local_product(Phis[k+1],Phis[k],A.cores[k],tn.reshape(solution,[rx[k],N[k],rx[k+1]]),solution_now.shape),[-1,1]) - rhs)/norm_rhs
                            res = tn.linalg.norm(Op.matvec(solution.to(
                                tn.float32 if use_single_precision else dtype)).to(dtype)-rhs)/norm_rhs
                        if res > max(real_tol*damp, res_new):
                            break
                    r += 1

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
                czA = _local_product(Phiz[k+1], Phiz[k], Summ_data[k], tn.reshape(
                                     u@v.t(), [rx[k], N[k], rx[k+1]]), np.max(bandsMatrices[:, k]))  # shape rzp x N x rz

                # shape is rzp x N x rz
                czy = tn.einsum('br,bnB,BR->rnR',
                                Phiz_b[k], b.cores[k]*nrmsc, Phiz_b[k+1])
                cz_new = czy - czA

                uz, _, _ = SVD(tn.reshape(cz_new, [rz[k]*N[k], rz[k+1]]))
                # truncate to kickrank
                cz_new = uz[:, :min(kickrank, uz.shape[1])]
                if k < d-1:  # extend cz_new with random elements
                    cz_new = tn.cat(
                        (cz_new, tn.randn((cz_new.shape[0], kick2),  dtype=dtype, device=device)), 1)

                qz, _ = QR(cz_new)
                rz[k+1] = qz.shape[1]
                z_cores[k] = tn.reshape(qz, [rz[k], N[k], rz[k+1]])

            if k < d-1:
                if not last:
                    left_res = _local_product(Phiz[k+1], Phis[k], Summ_data[k], tn.reshape(
                                              u@v.t(), [rx[k], N[k], rx[k+1]]), np.max(bandsMatrices[:, k]))
                    left_b = tn.einsum(
                        'br,bmB,BR->rmR', Phis_b[k], b.cores[k]*nrmsc, Phiz_b[k+1])
                    uk = left_b - left_res  # rx_k x N_k x rz_k+1
                    u, Rmat = QR(
                        tn.cat((u, tn.reshape(uk, [u.shape[0], -1])), 1))
                    r_add = uk.shape[2]
                    v = tn.cat(
                        (v, tn.zeros([rx[k+1], r_add],  dtype=dtype, device=device)), 1)
                    v = v @ Rmat.t()

                r = u.shape[1]
                v = tn.einsum('ji,jkl->ikl', v, x_cores[k+1])
                # remove norm correction
                nrmsc = nrmsc * normA[k] * normx[k] / normb[k]

                norm_now = tn.linalg.norm(v)

                if norm_now > 0:
                    v = v / norm_now
                else:
                    norm_now = 1.0
                normx[k] = normx[k] * norm_now

                x_cores[k] = tn.reshape(u, [rx[k], N[k], r])
                x_cores[k+1] = tn.reshape(v, [r, N[k+1], rx[k+2]])
                rx[k+1] = r

                # next phis with norm correction
                Phis[k+1] = _compute_phi_A('fwd', Phis[k], x_cores[k], Summ_data[k], x_cores[k], A_ranks[k], A_ranks[k+1],  
                                           np.max(bandsMatrices[:, k]))
                Phis_b[k+1] = _compute_phi_fwd_rhs(Phis_b[k], b.cores[k], x_cores[k])

                # ... and norms
                norm = tn.linalg.norm(Phis[k+1])
                norm = norm if norm > 0 else 1.0
                normA[k] = norm
                Phis[k+1] = Phis[k+1] / norm
                norm = tn.linalg.norm(Phis_b[k+1])
                norm = norm if norm > 0 else 1.0
                normb[k] = norm
                Phis_b[k+1] = Phis_b[k+1] / norm

                # norm correction
                nrmsc = nrmsc * normb[k] / (normA[k] * normx[k])

                # next phiz
                if not last:
                    Phiz[k+1] = _compute_phi_A('fwd', Phiz[k], z_cores[k], Summ_data[k], x_cores[k], A_ranks[k], A_ranks[k+1], 
                                               np.max(bandsMatrices[:, k])) / normA[k]
                    Phiz_b[k+1] = _compute_phi_fwd_rhs(Phiz_b[k], b.cores[k], z_cores[k], ) / normb[k]
            else:
                x_cores[k] = tn.reshape(
                    u@tn.diag(s[:r]) @ v[:r, :].t(), [rx[k], N[k], rx[k+1]])

        if verbose:
            print('Solution rank is', rx)
            print('Maxres ', max_res)
            tme_sweep = datetime.datetime.now()-tme_sweep
            print('Time ', tme_sweep)

        if last:
            break

        if max_res < eps:
            last = True

    if verbose:
        time_total = datetime.datetime.now() - time_total
        print()
        print('Finished after', swp+1, ' sweeps and ', time_total)
        print()
    normx = np.exp(np.sum(np.log(normx))/d)

    for k in range(d):
        x_cores[k] = x_cores[k] * normx

    x = torchtt.TT(x_cores)

    return x

def _compute_phi_A(order, Phi_now, core_left, Summ_data, core_right, s, S, band_max):
    """
    Compute the phi backwards for the form dot(left,A @ right)

    Args:
        order(str): fwd or bck
        Phi_now (torch.tensor): The current phi. If order == 'bck' has shape r1_k+1 x R_k+1 x r2_k+1
                                                 If order == 'fwd' has shape r1_k x R_k x r2_k
        core_left (torch.tensor): the core on the left. Has shape r1_k x N_k x r1_k+1 
        core_A (torch.tensor): the core of the matrix. Has shape  R_k x N_k x N_k x R_k
        core_right (torch.tensor): the core to the right. Has shape r2_k x N_k x r2_k+1 

    Returns:
        torch.tensor: The following phi (backward).  If order == 'bck' has shape r1_k x R_k x r2_k
                                                     If order == 'fwd' has shape r1_k+1 x R_k+1 x r2_k+1
    """
    if order == 'bck':
        Phi = tn.zeros(core_left.shape[0], s, core_right.shape[0], device=core_left.device, dtype=core_left.dtype)
    elif order == 'fwd':
        Phi = tn.zeros(core_left.shape[-1], S, core_right.shape[-1], device=core_left.device, dtype=core_left.dtype)
    for data, band, s_beg, s_end, S_beg, S_end in Summ_data:
        if band < 0:
            # data is a full core
            if order == 'bck':
                Phi[:, s_beg:s_end, :] += oe.contract('LSR,lML,sMNS,rNR->lsr', 
                                                       Phi_now[:, S_beg:S_end, :], core_left, data, core_right)
            elif order == 'fwd':
                Phi[:, S_beg:S_end, :] += oe.contract('lsr,lML,sMNS,rNR->LSR', 
                                                       Phi_now[:, s_beg:s_end, :], core_left, data, core_right)
        else:            
            cores_left = tn.stack([tnf.pad(core_left[:, -i:, :], (0, 0, 0, -i)) for i in range(-band, 1)] + 
                                  [tnf.pad(core_left[:, :-i, :], (0, 0, i, 0)) for i in range(1, band+1)])

            # data is diagonals of a core
            if order == 'bck':
                Phi[:, s_beg:s_end, :] += oe.contract('LSR,zlKL,zsSK,rKR->lsr', Phi_now[:, S_beg:S_end, :],
                                                      cores_left, data, core_right)
            elif order == 'fwd':
                Phi[:, S_beg:S_end, :] += oe.contract('lsr,zlKL,zsSK,rKR->LSR', Phi_now[:, s_beg:s_end, :],
                                                      cores_left, data, core_right)
    return Phi



def _compute_phi_bck_rhs(Phi_now, core_b, core):
    """


    Args:
        Phi_now (torch.tensor): The current phi. Has shape rb_k+1 x r_k+1
        core_b (torch.tensor): The current core of the rhs. Has shape rb_k x N_k x rb_k+1
        core (torch.tensor): The current core. Has shape r_k x N_k x r_k+1

    Returns:
        torch.tensor: The backward phi corresponding to the rhs. Has shape rb_k x r_k
    """
    # Phit = tn.einsum('ij,abj->iba',Phi_now,core_b)
    # Phi = tn.einsum('ijk,kjc->ic',core,Phit)
    Phi = oe.contract('BR,bnB,rnR->br', Phi_now, core_b, core)
    return Phi


def _compute_phi_fwd_rhs(Phi_now, core_rhs, core):
    """


    Args:
        Phi_now (torch.tensor): The current phi. Has shape  rb_k x r_k
        core_b (torch.tensor): The current core of the rhs. Has shape rb_k x N_k+1 x rb_k+1
        core (torch.tensor): The current core. Has shape r_k x N_k x r_k+1

    Returns:
        torch.tensor: The forward computer phi for the rhs. Has shape rb_k+1 x r_k+1
    """
    # tmp = tn.einsum('ij,jbc->ibc',Phi_now,core_rhs) # shape rk-1 x Nk x rbk
    # Phi_next = tn.einsum('ijk,ijc->kc',core,tmp)
    Phi_next = oe.contract('br,bnB,rnR->BR', Phi_now, core_rhs, core)
    return Phi_next
