"""
System solvers in the TT format.

"""

import torch as tn
import numpy as np
import torchtt
import datetime
from torchtt._decomposition import QR, SVD, lr_orthogonal, rl_orthogonal
from torchtt._iterative_solvers import BiCGSTAB_reset, gmres_restart, gmres
import opt_einsum as oe
from .errors import *
import torch.nn.functional as tnf
import scipy


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

def _local_product(Phi_right_list, Phi_left_list, Summ_data, x):
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
    for data, band, j in Summ_data:
        if band < 0:
            # data is a full core
            w += oe.contract('lsr, smnS, LSR, rnR -> lmL', 
                              Phi_left_list[j], data, Phi_right_list[j], x)
        else:
            # data is diagonals of a core
            tmp = oe.contract('lsr, ksSn, LSR, rnR -> klnL',
                               Phi_left_list[j], data, Phi_right_list[j], x)
            w += tmp[band, ...]
            for i in range(1, band + 1):
                w[:, :-i, :] += tmp[i + band, :, i:, :]
                w[:, i:, :] += tmp[-i + band, :, :-i, :]
        
    return w


class _LinearOp():
    def __init__(self, Phi_left_list, Phi_right_list, Summ_data, shape, prec):
        self.shape = shape
        self.prec = prec
        if prec is not None:
            if prec == 'c' or prec == 'r':
                self.prec = prec
            else:
                raise Exception('Preconditioner ' + str(prec) + ' not defined.')
        self.summ_data = Summ_data
        self.expressions = []
        for data, band, j in Summ_data:
            sizes = 'lsr, smnS, LSR, rnR-> lmL' if band < 0 else 'lsr, ksSn, LSR, rnR -> klnL'
            expr = oe.contract_expression(sizes,
                                          Phi_left_list[j].contiguous(), data.contiguous(), Phi_right_list[j].contiguous(), 
                                          shape,
                                          constants=[0, 1, 2], optimize='optimal')
            self.expressions.append(expr)
        if prec == 'c':
            J = tn.zeros(Phi_left_list[0].shape[-1], Phi_right_list[0].shape[-1], shape[1], shape[1],
                              dtype=Phi_left_list[0].dtype, device=Phi_left_list[0].device)
            self.band_J = 0
            for data, band, j in Summ_data:
                if band < 0:
                    J += oe.contract('sr, smnS, SR -> rRmn',
                                     tn.diagonal(Phi_left_list[j], 0, 0, 2).contiguous(),
                                     data,
                                     tn.diagonal(Phi_right_list[j], 0, 0, 2).contiguous())
                else:
                    diags_J = oe.contract('sr, ksSn, SR -> krRn',
                                     tn.diagonal(Phi_left_list[j], 0, 0, 2).contiguous(),
                                     data,
                                     tn.diagonal(Phi_right_list[j], 0, 0, 2).contiguous())
                    for i in range(-band, band + 1):
                        diag_i_J = tn.diagonal(J, i, 2, 3)
                        if i >= 0:
                            diag_i_J += diags_J[i + band, ..., i:]
                        else:
                            diag_i_J += diags_J[i + band, ..., :i]
            self.J_LU_factors, self.J_pivots = tn.linalg.lu_factor(J)
            
        elif prec == 'r':
            J = tn.zeros(Phi_left_list[0].shape[-1], shape[1], Phi_right_list[0].shape[0], shape[1], Phi_right_list[0].shape[-1],
                              dtype=Phi_left_list[0].dtype, device = Phi_left_list[0].device)
            self.J_band = 0
            for data, band, j in Summ_data:
                if band < 0:
                    J += oe.contract('sr, smnS, LSR- > rmLnR', 
                                     tn.diagonal(Phi_left_list[j], 0, 0, 2).contiguous(),
                                     data, 
                                     Phi_right_list[j])
                else:
                    diags_J = oe.contract('sr, ksSn, LSR -> krLRn',
                                          tn.diagonal(Phi_left_list[j], 0, 0, 2).contiguous(),
                                          data, 
                                          Phi_right_list[j])
                    for i in range(-band, band + 1):
                        diag_i_J = tn.diagonal(J, i, 1, 3)
                        if i >= 0:
                            diag_i_J += diags_J[i + band, ..., i:]
                        else:
                            diag_i_J += diags_J[i + band, ..., :i]
            sh = J.shape
            J = tn.reshape(J, [-1, J.shape[1] * J.shape[2], J.shape[3] * J.shape[4]])
            self.J_LU_factors, self.J_pivots = tn.linalg.lu_factor(J)
        else:
            pass

    def apply_prec(self, x):
        if self.prec == 'c':
            y = tn.reshape(x, self.shape + [1]) # rnR1
            y = tn.permute(y, (0, 2, 1, 3)).contiguous() # rRn1
            z = tn.linalg.lu_solve(self.J_LU_factors, self.J_pivots, y)
            return tn.permute(z, (0, 2, 1, 3)).reshape(x.shape).contiguous()
        else:
            # rnR, rmLnR -> rmL
            y = x.reshape([self.shape[0], -1, 1]) # r(nR)1
            y = tn.linalg.lu_solve(self.J_LU_factors, self.J_pivots, y)
            return y.reshape(x.shape).contiguous()

    def matvec(self, x, apply_prec=True):
        full_time = datetime.datetime.now()
        y = x.reshape(self.shape)
        w = 0
        if (self.prec is not None) and apply_prec:
            y = self.apply_prec(y)
        y = y.contiguous()
        for _, band, j in self.summ_data:
            if band >= 0:
                # data is diagonals of core
                tmp = self.expressions[j](y)
                w += tmp[band, ...]
                for i in range(1, band + 1):
                     w[:, :-i, :] += tmp[i + band, :, i:, :]
                     w[:, i:, :] += tmp[-i + band, :, :-i, :]
            else:
                # data is a full core
                w += self.expressions[j](y)
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
            x_R = [1] * (1 + len(A.N))
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
    
    Summ_data = []
    for k in range(d):
        k_th_cores = []
        for i in range((len(Matrices))):
            core = Matrices[i].cores[k]
            band = bandsMatrices[i, k]
            if band < 0:
                k_th_cores.append(core.clone())
            else:
                k_th_cores.append(tn.stack([tnf.pad(tn.diagonal(core, i, 1, 2), (0, -i)).contiguous() for i in range(-band, 0)] + 
                                           [tnf.pad(tn.diagonal(core, i, 1, 2), (i, 0)).contiguous() for i in range(0, band + 1)]))
        Summ_data.append(list(zip(k_th_cores, bandsMatrices[:, k], list(range(len(Matrices))))))      
    
    
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
        
    x_cores = x.cores.copy()
    rx = x.R.copy()
    
    # check if rmax is a list
    if isinstance(rmax, int):
        rmax = [1] + (d - 1) * [rmax] + [1]

    # z cores
    rz = [1] + [kickrank + kick2] * (d - 1) + [1]
    z_cores = torchtt.random(N, rz, dtype, device=device).cores
    z_cores, rz = rl_orthogonal(z_cores, rz, False)

    norms = np.zeros(d)
    Phiz = [[tn.ones((1, 1, 1), dtype=dtype, device=device) for i in range(len(Matrices))]] +  \
           [[None] * len(Matrices)] * (d - 1) + \
           [[tn.ones((1, 1, 1), dtype=dtype, device=device) for i in range(len(Matrices))]]  # size is rzk x Rk x rxk
    Phiz_b = [tn.ones((1, 1), dtype=dtype, device=device)] +  [None] * (d - 1) + \
             [tn.ones((1, 1), dtype=dtype, device=device)]   # size is rzk x rzbk

    Phis = [[tn.ones((1, 1, 1), dtype=dtype, device=device) for i in range(len(Matrices))]] + \
           [[None] * len(Matrices)] * (d - 1) + \
           [[tn.ones((1, 1, 1), dtype=dtype, device=device) for i in range(len(Matrices))]]  # size is rk x Rk x rk
    Phis_b = [tn.ones((1, 1), dtype=dtype, device=device)] + [None] * (d - 1) + \
             [tn.ones((1, 1), dtype=dtype, device=device)]  # size is rk x rbk

    last = False

    normA = np.ones(d - 1)
    normb = np.ones(d - 1)
    normx = np.ones(d - 1)
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

        #orthogonalization 
        for k in range(d - 1, 0, -1):
            # update the z part (ALS) update
            if not last:
                if swp > 0:
                    # shape rzp x N x rz
                    czA = _local_product(Phiz[k + 1], Phiz[k], Summ_data[k], x_cores[k])
                    # shape is rzp x N x rz
                    czy = oe.contract('br, bnB, BR -> rnR',
                                    Phiz_b[k], b.cores[k], Phiz_b[k + 1])
                    cz_new = czy*nrmsc - czA
                    _, _, vz = SVD(tn.reshape(cz_new, [cz_new.shape[0], -1]))
                    # truncate to kickrank
                    cz_new = vz[:min(kickrank, vz.shape[0]), :].t()
                    if k < d - 1:  # extend cz_new with random elements
                        cz_new = tn.cat(
                            (cz_new, tn.randn((cz_new.shape[0], kick2),  dtype=dtype, device=device)), 1)
                else:
                    cz_new = tn.reshape(z_cores[k], [rz[k], -1]).t().contiguous()

                qz, _ = QR(cz_new)
                rz[k] = qz.shape[1]
                z_cores[k] = tn.reshape(qz.t(), [rz[k], N[k], rz[k + 1]]).contiguous()

            # norm correction ?
            if swp > 0:
                nrmsc = nrmsc * normA[k - 1] * normx[k - 1] / normb[k - 1]

            core = tn.reshape(x_cores[k], [rx[k], N[k] * rx[k + 1]]).t().contiguous()
            Qmat, Rmat = QR(core)

            core_prev = oe.contract('ijk, mk -> ijm', x_cores[k - 1], Rmat)
            rx[k] = Qmat.shape[1]
            

            current_norm = tn.linalg.norm(core_prev)
            if current_norm > 0:
                core_prev = core_prev / current_norm
            else:
                current_norm = 1.0
            normx[k - 1] = normx[k - 1] * current_norm

            x_cores[k] = tn.reshape(Qmat.t(), [rx[k], N[k], rx[k + 1]]).contiguous()
            x_cores[k - 1] = core_prev[:]

            # update phis (einsum)
            Phis[k] = _compute_phi_A('bck', Phis[k + 1], x_cores[k], Summ_data[k], x_cores[k])
            Phis_b[k] = _compute_phi_bck_rhs(Phis_b[k + 1], b.cores[k], x_cores[k])

            # ... and norms
            norm = sum([tn.linalg.norm(Phis[k][i]) ** 2 for i in range(len(Matrices))])
            norm = tn.sqrt(norm)
            norm = norm if norm > 0 else 1.0
            normA[k - 1] = norm
            Phis[k] = [Phis[k][i] / norm for i in range(len(Matrices))]
            norm = tn.linalg.norm(Phis_b[k])
            norm = norm if norm > 0 else 1.0
            normb[k - 1] = norm
            Phis_b[k] = Phis_b[k] / norm

            # norm correction
            nrmsc = nrmsc * normb[k - 1] / (normA[k - 1] * normx[k - 1])

            # compute phis_z
            if not last:
                Phiz[k] = _compute_phi_A('bck', Phiz[k + 1], z_cores[k], Summ_data[k], x_cores[k])
                Phiz[k] = [Phiz[k][i] / normA[k - 1] for i in range(len(Matrices))]
                Phiz_b[k] = _compute_phi_bck_rhs(Phiz_b[k + 1], b.cores[k], z_cores[k]) / normb[k - 1]

        # start loop
        max_res = 0
        max_dx = 0

        for k in range(d):
            if verbose:
                print('\tCore', k)
            previous_solution = tn.reshape(x_cores[k], [-1, 1])

            # assemble rhs
            rhs = oe.contract('br, bmB, BR -> rmR',
                              Phis_b[k], b.cores[k] * nrmsc, Phis_b[k + 1])
            rhs = tn.reshape(rhs, [-1, 1])
            norm_rhs = tn.linalg.norm(rhs)

            # residuals
            real_tol = (eps / np.sqrt(d)) / damp

            # solve the local system
            use_full = rx[k] * N[k] * rx[k + 1] < max_full
            if use_full:
                # solve the full system
                if verbose:
                    print('\t\tChoosing direct solver (local size %d)....' %
                          (rx[k] * N[k] * rx[k + 1]))
                # shape is Rp x N x N x r x r
                B = tn.zeros(rx[k], N[k], rx[k + 1], rx[k], N[k], rx[k + 1], dtype=dtype, device=device)
                for data, band, j in Summ_data[k]:
                    if band < 0:
                        Bp = oe.contract('smnS, LSR -> smnRL', data, Phis[k + 1][j])
                        B += oe.contract('lsr,smnRL->lmLrnR', Phis[k][j], Bp)
                    else:
                        diags_Bp = oe.contract('ksSn, LSR -> ksRLn', data, Phis[k + 1][j])
                        diags_B = oe.contract('lsr, ksRLn -> klLrRn', Phis[k][j], diags_Bp)
                        for i in range(-band, band + 1):
                            diag_i_B = tn.diagonal(B, i, 1, 4)
                            if i >= 0:
                                diag_i_B += diags_B[i + band, ..., i:]
                            else:
                                diag_i_B += diags_B[i + band, ..., :i]
                B = tn.reshape(B, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])

                solution_now = tn.linalg.solve(B, rhs)

                res_old = tn.linalg.norm(B @ previous_solution - rhs) / norm_rhs
                res_new = tn.linalg.norm(B @ solution_now - rhs) / norm_rhs
            else:
                # iterative solver
                if verbose:
                    print('\t\tChoosing iterative solver %s (local size %d)....' % (
                        'GMRES' if local_solver == 1 else 'BiCGSTAB_reset', rx[k] * N[k] * rx[k + 1]))
                    time_local = datetime.datetime.now()
                shape_now = [rx[k], N[k], rx[k + 1]]

                if use_single_precision:
                    Op = _LinearOp(Phis[k], Phis[k + 1],
                                   Summ_data[k], shape_now, preconditioner)

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
                        solution_now = Op.apply_prec(solution_now)

                    solution_now = previous_solution + solution_now.to(dtype)
                    res_old = tn.linalg.norm(Op.matvec(previous_solution.to(
                        tn.float32), False).to(dtype) - rhs) / norm_rhs
                    res_new = tn.linalg.norm(Op.matvec(solution_now.to(
                        tn.float32), False).to(dtype) - rhs) / norm_rhs
                else:
                    Op = _LinearOp(Phis[k], Phis[k + 1],
                                   Summ_data[k], shape_now, preconditioner)

                    # solution_now, flag, nit, res_new = BiCGSTAB_reset(Op, rhs,previous_solution[:], eps_local, local_iterations)
                    eps_local = real_tol * norm_rhs
                    drhs = Op.matvec(previous_solution, False)
                    drhs = rhs - drhs
                    eps_local = eps_local / tn.linalg.norm(drhs)
                    if local_solver == 1:
                        solution_now, flag, nit = gmres_restart(
                            Op, drhs, previous_solution*0, rhs.shape[0], local_iterations + 1, eps_local, resets)
                    elif local_solver == 2:
                        solution_now, flag, nit, _ = BiCGSTAB_reset(
                            Op, drhs, previous_solution*0, eps_local, local_iterations)
                    else:
                        raise InvalidArguments('Solver not implemented.')

                    if preconditioner != None:
                        solution_now = Op.apply_prec(solution_now)

                    solution_now = previous_solution + solution_now
                    res_old = tn.linalg.norm(
                        Op.matvec(previous_solution, False) - rhs) / norm_rhs
                    res_new = tn.linalg.norm(Op.matvec(solution_now, False) - rhs) / norm_rhs

                if verbose:
                    print('\t\tFinished with flag %d after %d iterations with relres %g (from %g)' % (
                        flag, nit, res_new, real_tol * norm_rhs))
                    time_local = datetime.datetime.now() - time_local
                    print('\t\tTime needed ', time_local)
                    
            # residual damp check
            if res_old / res_new < damp and res_new > real_tol:
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

            solution_now = tn.reshape(solution_now, [rx[k] * N[k], rx[k + 1]])
            # truncation
            if k < d:
                u, s, v = SVD(solution_now)
                if trunc_norm == 'fro':
                    pass
                else:
                    # search for a rank such that offeres small enough residual
                    # TODO: binary search?
                    r = 0
                    for r in range(u.shape[1] - 1, 0, -1):
                        # solution has the same size
                        solution = u[:, :r] @ tn.diag(s[:r]) @ v[:r, :]

                        if use_full:
                            res = tn.linalg.norm(B @ tn.reshape(solution, [-1, 1]) - rhs)
                        else:
                            res = tn.linalg.norm(Op.matvec(solution.to(
                                tn.float32 if use_single_precision else dtype), False).to(dtype) - rhs)
                        if res > real_tol * norm_rhs:
                            break
                    r += 1
                    r = min([r, tn.numel(s), rmax[k + 1]])
                u = u[:, :r]
                v = tn.diag(s[:r]) @ v[:r, :]
                

            if not last:
                czA = _local_product(Phiz[k + 1], Phiz[k], Summ_data[k], 
                                     tn.reshape(u @ v, [rx[k], N[k], rx[k + 1]]))  # shape rzp x N x rz
                # shape is rzp x N x rz
                czy = oe.contract('br, bnB , BR -> rnR',
                                Phiz_b[k], b.cores[k] * nrmsc, Phiz_b[k + 1])
                cz_new = czy - czA

                uz, _, _ = SVD(tn.reshape(cz_new, [rz[k] * N[k], rz[k + 1]]))
                # truncate to kickrank
                cz_new = uz[:, :min(kickrank, uz.shape[1])]
                if k < d - 1:  # extend cz_new with random elements
                    cz_new = tn.cat(
                        (cz_new, tn.randn((cz_new.shape[0], kick2),  dtype=dtype, device=device)), 1)

                qz, _ = QR(cz_new)
                rz[k + 1] = qz.shape[1]
                z_cores[k] = tn.reshape(qz, [rz[k], N[k], rz[k + 1]]).contiguous()

            if k < d - 1:
                if not last:
                    left_res = _local_product(Phiz[k + 1], Phis[k], Summ_data[k],
                                              tn.reshape(u @ v, [rx[k], N[k], rx[k + 1]]))
                    left_b = oe.contract('br, bmB, BR -> rmR', Phis_b[k],  nrmsc * b.cores[k], Phiz_b[k + 1])
                    uk = left_b - left_res  # rx_k x N_k x rz_k+1
                    u, Rmat = QR(tn.cat((u, tn.reshape(uk, [u.shape[0], -1])), 1))
                    r_add = uk.shape[2]
                    v = Rmat[:, :v.shape[0]] @ v

                r = u.shape[1]
                v = oe.contract('ij, jkl -> ikl', v, x_cores[k + 1])
                # remove norm correction
                nrmsc = nrmsc * normA[k] * normx[k] / normb[k]

                norm_now = tn.linalg.norm(s[:r]) # v and x_cores[k+1] are orthogonal
                norm_now = norm_now if norm_now > 0 else 1.0
                v = v / norm_now
                normx[k] *= norm_now

                x_cores[k] = tn.reshape(u, [rx[k], N[k], r]).contiguous()
                x_cores[k + 1] = tn.reshape(v, [r, N[k + 1], rx[k + 2]]).contiguous()
                rx[k + 1] = r

                # next phis with norm correction
                Phis[k + 1] = _compute_phi_A('fwd', Phis[k], x_cores[k], Summ_data[k], x_cores[k])
                
                Phis_b[k + 1] = _compute_phi_fwd_rhs(Phis_b[k], b.cores[k], x_cores[k])

                # ... and norms
                norm = sum([tn.linalg.norm(Phis[k + 1][i]) ** 2 for i in range(len(Matrices))])
                norm = tn.sqrt(norm)
                norm = norm if norm > 0 else 1.0
                normA[k] = norm
                Phis[k + 1] = [Phis[k + 1][i] / norm for i in range(len(Matrices))]
                norm = tn.linalg.norm(Phis_b[k + 1])
                norm = norm if norm > 0 else 1.0
                normb[k] = norm
                Phis_b[k + 1] = Phis_b[k + 1] / norm

                # norm correction
                nrmsc = nrmsc * normb[k] / (normA[k] * normx[k])

                # next phiz
                if not last:
                    Phiz[k + 1] = _compute_phi_A('fwd', Phiz[k], z_cores[k], Summ_data[k], x_cores[k])
                    Phiz[k + 1] = [Phiz[k + 1][i] / normA[k] for i in range(len(Phiz[k + 1]))]
                    Phiz_b[k + 1] = _compute_phi_fwd_rhs(Phiz_b[k], b.cores[k], z_cores[k]) / normb[k]
            else:
                x_cores[k] = tn.reshape(u @ v, [rx[k], N[k], rx[k + 1]]).contiguous()

        if verbose:
            print('Solution rank is', rx)
            print('Maxres ', max_res)
            tme_sweep = datetime.datetime.now()-tme_sweep
            print('Time ', tme_sweep)

        if last:
            #print("swp number = ", swp)
            break

        last = max_res < eps

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

def _compute_phi_A(order, Phi_now_list, core_left, Summ_data, core_right):
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
    Phi_list = []
    if order == 'fwd':
        core_right_tmp = tn.transpose(core_right, 0, 2).contiguous()
        core_left_tmp = tn.transpose(core_left, 0, 2).contiguous()
    band_max = -1
    for _, band, _ in Summ_data:
        if band_max < band:
            band_max = band
    if band_max >= 0:
        cores_left = tn.stack([tnf.pad(core_left[:, -i:, :], (0, 0, 0, -i)) for i in range(-band_max, 1)] + 
                                [tnf.pad(core_left[:, :-i, :], (0, 0, i, 0)) for i in range(1, band_max + 1)])
    for data, band, j in Summ_data:
        if band < 0:
            # data is a full core
            if order == 'bck':
                Phi_list.append(oe.contract('LSR, lmL, smnS, rnR -> lsr',
                                            Phi_now_list[j], core_left, data, core_right))
            elif order == 'fwd':
                Phi_list.append(oe.contract('lsr, Lml, smnS, Rnr -> LSR',
                                            Phi_now_list[j], core_left_tmp, data, core_right_tmp))
        else:            
            # data is diagonals of a core
            if order == 'bck':
                Phi_list.append(oe.contract('LSR, klnL, ksSn, rnR -> lsr',
                                            Phi_now_list[j], cores_left[band_max-band:band_max+band+1, ...], data, core_right))
            elif order == 'fwd':
                Phi_list.append(oe.contract('lsr, klnL, ksSn, rnR -> LSR',
                                            Phi_now_list[j], cores_left[band_max-band:band_max+band+1, ...], data, core_right))
    return Phi_list



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
    Phi = tn.einsum('BR,bnB,rnR->br', Phi_now, core_b, core)
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
    Phi_next = tn.einsum('br,bnB,rnR->BR', Phi_now, core_rhs, core)
    return Phi_next
