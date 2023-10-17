'''
Taken from ttpy
'''
import torch as tn
import numpy as np
import torchtt as tntt
import datetime
from torchtt._decomposition import QR, SVD, lr_orthogonal, rl_orthogonal
from torchtt._iterative_solvers import BiCGSTAB_reset, gmres_restart
import opt_einsum as oe
from ._amen_approx_contractors import mv_local_op, mv_multiple_local_op, mvm_multiple_local_op, mm_multiple_local_op

def _reshape(a, shape):
    return _np.reshape(a, shape, order='F')


def _tconj(a):
    return a.T.conjugate()


def _my_chop2(sv, eps):
    if eps <= 0.0:
        r = len(sv)
        return r
    sv0 = np.cumsum(abs(sv[::-1]) ** 2)[::-1]
    ff = [i for i in range(len(sv0)) if sv0[i] < eps ** 2]
    if len(ff) == 0:
        return len(sv)
    else:
        return np.amin(ff)


def amen_approx(contractor, tol, shape_out_M, shape_out_N, y=None, z=None, nswp=20, kickrank=4,
                kickrank2=0, verb=0, init_qr=True, fkick=False, device=None):
    """
    AMEn to solve the following minimization problem problem ||y-(expr)||_F^2. The expr can be anything from matvec to matmat or multiple matvecs or matmats.
    The behavior is defined by the contraftor argument.
    
    Args:
        contractor (object): the contractor
        tol (flaot): relative tolerance.
        shape_out_M (_type_): row shape of the output. Can be None.
        shape_out_N (_type_): column shape of the output. 
        y (_type_, optional): initial guess. None menas random guess is made. Defaults to None.
        z (_type_, optional): initial guess for the z. None means random guess is made. Defaults to None.
        nswp (int, optional): number of sweeps. Defaults to 20.
        kickrank (int, optional): rank enrichment. Defaults to 4.
        kickrank2 (int, optional): random enrichment. Defaults to 0.
        verb (int, optional): verbose flag. Defaults to 0.
        init_qr (bool, optional): Perform QR of the input. Defaults to True.
        fkick (bool, optional):  Perform solution enrichment during forward sweeps. Defaults to False.
        device (_type_, optional): _description_. Defaults to None.

    Returns:
        tntt.TT: the result.
    """
    dtype = tn.float64 
    d = len(shape_out_N)
    M = shape_out_M
    N = shape_out_N
    ttm = shape_out_M is not None
    MN = [m*n for m, n in zip(M, N)] if ttm else N

    if y is None:
        ry = [1]+[2]*(d-1) + [1]
        if shape_out_M is None:
            cores_y = tntt.randn(shape_out_N, ry).cores
        else:
            cores_y = tntt.randn(
                [(m, n) for m, n in zip(shape_out_M, shape_out_N)], ry).cores
    else:
        cores_y = y.cores
        ry = y.R

    if kickrank + kickrank2 > 0:
        if z is None:
            rz = [1]+[kickrank+kickrank2]*(d-1) + [1]
            if shape_out_M is None:
                cores_z = tntt.randn(shape_out_N, rz).cores
            else:
                cores_z = tntt.randn(
                    [(m, n) for m, n in zip(shape_out_M, shape_out_N)], rz).cores
        else:
            cores_z = z.cores
            ry = z.R

        phizy = [tn.ones((1,1), device=device, dtype=dtype)] + [None] * \
            (d-1) + [tn.ones((1,1), device=device, dtype=dtype)]

    nrms = tn.ones(d, dtype=dtype)

    # Orthogonalization
    for i in range(d - 1):
        if init_qr:
            cr = tn.reshape(cores_y[i], (ry[i] * MN[i], ry[i + 1]))
            [cr, R] = QR(cr)
            nrmr = tn.linalg.norm(R)  # , 'fro')
            if (nrmr > 0):
                R = R / nrmr
            # cr2 = tn.reshape(cores_y[i + 1], (ry[i + 1], n[i + 1] * ry[i + 2]))
            # cr2 = _np.dot(R, cr2)
            cr2 = tn.tensordot(R, cores_y[i + 1], ([1], [0]))
            ry[i + 1] = cr.shape[1]
            cores_y[i] = tn.reshape(
                cr, (ry[i], M[i], N[i], ry[i + 1]) if ttm else (ry[i], N[i], ry[i + 1]))
            cores_y[i+1] = cr2.clone()

        nrms[i] = contractor.update_phi_y(cores_y[i], i, 'lr', 1, True)

        if (kickrank + kickrank2 > 0):
            [cr, R] = QR(tn.reshape(cores_z[i], [-1, rz[i+1]]))
            nrmr = tn.linalg.norm(R)  # , 'fro')
            if (nrmr > 0):
                R = R / nrmr

            cores_z[i + 1] = tn.tensordot(R, cores_z[i+1],
                                    ([1], [0]))  # _np.dot(R, cr2)
            rz[i + 1] = cr.shape[1]
            cores_z[i] = tn.reshape(cr, (rz[i], M[i], N[i], rz[i + 1])
                              if ttm else (rz[i], N[i], rz[i + 1]))

            contractor.update_phi_z(cores_z[i], i, 'lr', nrms[i], False)

            phizy[i+1] = compute_phiy_lr(phizy[i], cores_z[i], cores_y[i])

    i = d - 1
    direct = -1
    swp = 1
    max_dx = 0

    if verb > 0:
        tme_total = datetime.datetime.now()

    while swp <= nswp:
        if (verb > 0):
            if ((direct < 0) and (i == d - 1)) or ((direct > 0) and (i == 0)):
                print("Sweep %d, direction %d"%(swp, direct))
                tme_swp = datetime.datetime.now()
        # Project the MatVec generating vector
        cry = contractor.b_fun(i, 'y', 'y')

        nrms[i] = tn.linalg.norm(cry)  # , 'fro')
        if (nrms[i] > 0):
            cry = cry / nrms[i]
        else:
            nrms[i] = 1

        dx = tn.linalg.norm(cry - cores_y[i])
        max_dx = max(max_dx, dx)

        # Truncation and enrichment
        if ((direct > 0) and (i < d - 1)): 
            cry = tn.reshape(cry, (-1, ry[i + 1]))

            [u, s, vt] = tn.linalg.svd(cry, full_matrices=False)

            r = _my_chop2(s.cpu().numpy(), tol *
                          tn.linalg.norm(s).cpu().numpy() / d**0.5)
            u = u[:, :r]
            v =tn.einsum('i,ij->ji', s[:r], tn.conj(vt[:r, :]))

            # Prepare enrichment, if needed
            if (kickrank + kickrank2 > 0):
                cry = tn.tensordot(u, v, ([1], [1]))  # CONJUGATE??
                cry = tn.reshape(
                    cry, (ry[i], M[i], N[i], ry[i + 1]) if ttm else (ry[i], N[i], ry[i + 1]))
                # Z core update
                crz = contractor.b_fun(i, 'z', 'z')

                ys = tn.reshape(cry, [-1, ry[i+1]]) @ phizy[i + 1]
                yz = tn.reshape(ys, (ry[i], -1))
                yz = phizy[i] @ yz
                yz = tn.reshape(
                    yz, (rz[i], M[i], N[i], rz[i + 1]) if ttm else (rz[i], N[i], rz[i + 1]))
                crz = crz / nrms[i] - yz
                crz = tn.reshape(crz, [-1, rz[i+1]])
                nrmz = tn.linalg.norm(crz)
                if (kickrank2 > 0):
                    crz, _, _ = tn.linalg.svd(crz, full_matrices=False)
                    crz = crz[:, : min(crz.shape[1], kickrank)]
                    crz = tn.hstack(
                        (crz, tn.randn([crz.shape[0], kickrank2], device=device, dtype=dtype)))

                # For adding into solution
                if fkick:
                    crs = contractor.b_fun(i, 'y', 'z')

                    crs = tn.reshape(crs, (-1, rz[i + 1]))
                    crs = crs / nrms[i] - ys
                    u = tn.hstack((u, crs))

                    u, R = QR(u)

                    v = tn.hstack((v, tn.zeros((ry[i + 1], rz[i + 1]), device=device, dtype=dtype)))
                    v = tn.tensordot(v, R, ([1], [1]))  # _np.dot(v, R.T)
                    r = u.shape[1]
                    
            cores_y[i] = tn.reshape(u, (ry[i], M[i], N[i], r) if ttm else (ry[i], N[i], r))
            
            v = tn.reshape(v, (ry[i + 1], r))
            cores_y[i+1] = tn.tensordot(v, cores_y[i+1], ([0], [0]))

            ry[i + 1] = r

            nrms[i] = contractor.update_phi_y(cores_y[i], i, 'lr', 1, True)
        
            if (kickrank + kickrank2 > 0):
                crz, _ = QR(crz)

                rz[i + 1] = crz.shape[1]
                cores_z[i] = tn.reshape(crz, (rz[i], M[i], N[i], rz[i + 1]) if ttm else (rz[i], N[i], rz[i + 1]))

                contractor.update_phi_z(cores_z[i], i, 'lr', nrms[i], False)

                phizy[i+1] = compute_phiy_lr(phizy[i], cores_z[i], cores_y[i])

        elif ((direct < 0) and (i > 0)):
            cry = tn.reshape(cry, (ry[i], -1))

            u, s, vt = tn.linalg.svd(cry, full_matrices=False)

            
            r = _my_chop2(s.cpu().numpy(), tol *
                          tn.linalg.norm(s).cpu().numpy() / d**0.5)

            vt = vt[:r, :]
            v = tn.conj(vt.t())
            u = tn.einsum('ij,j->ij', u[:, :r], s[:r])
            # Prepare enrichment, if needed
            if (kickrank + kickrank2 > 0):
                cry = u @ v.t()
                cry = tn.reshape(cry, (ry[i], -1))

                crz = contractor.b_fun(i, 'z', 'z')
                crz = tn.reshape(crz, (rz[i], -1))
                ys = phizy[i] @ cry
                yz = tn.reshape(ys, (-1, ry[i + 1]))
                yz = yz @ phizy[i + 1]
                yz = tn.reshape(yz, (rz[i], -1))
                crz = crz / nrms[i] - yz
                nrmz = tn.linalg.norm(crz)
                if (kickrank2 > 0):
                    _, _, crz = tn.linalg.svd(crz, full_matrices=False)
                    crz = crz[:, : min(crz.shape[1], kickrank)]
                    crz = tn.conj(crz.t())
                    crz = tn.vstack((crz, tn.randn([kickrank2, crz.shape[1]], device=device, dtype=dtype)))

                crs = contractor.b_fun(i, 'z', 'y')

                crs = tn.reshape(crs, (rz[i], -1))
                crs = crs / nrms[i] - ys
                v = tn.hstack((v, crs.T))
                [v, R] = QR(v)

                u = tn.hstack((u, tn.zeros((ry[i], rz[i]), device=device, dtype=dtype)))
                u = tn.tensordot(u, R, ([1], [1]))
                r = v.shape[1]
            
            cr2 = tn.tensordot(cores_y[i-1], u, ([3 if ttm else 2], [0]))
            cores_y[i - 1] = tn.reshape(cr2, (ry[i - 1], M[i-1],
                                        N[i - 1], r) if ttm else (ry[i - 1], N[i - 1], r))
            cores_y[i] = tn.reshape(
                v.t(), (r, M[i], N[i], ry[i + 1]) if ttm else (r, N[i], ry[i + 1]))

            ry[i] = r

            nrms[i] = contractor.update_phi_y(cores_y[i], i, 'rl', 1, True)

            if (kickrank + kickrank2 > 0):
                
                crz, R = QR(crz.t())

                rz[i] = crz.shape[1]
                cores_z[i] = tn.reshape(
                    crz.t(), (rz[i], M[i], N[i], rz[i + 1]) if ttm else (rz[i], N[i], rz[i + 1]))

                contractor.update_phi_z(cores_z[i], i, 'rl', nrms[i], False)

                phizy[i] = compute_phiy_rl(phizy[i+1], cores_z[i], cores_y[i])

        if (verb > 1):
            print('\t\tcore %d, dx %e, rank %d:' %(i, dx, r))

        # Stopping or reversing
        if ((direct > 0) and (i == d - 1)) or ((direct < 0) and (i == 0)):
            if (verb > 0):
                tme_swp = datetime.datetime.now() - tme_swp
                print('\tfinished after %s, max dx %.3e, max rank %d' %(str(tme_swp), max_dx, max(ry)))
            if ((max_dx < tol) or (swp == nswp)) and (direct > 0):
                break
            else:
                # We are at the terminal block
                cores_y[i] = tn.reshape(
                    cry, (ry[i], M[i], N[i], ry[i + 1]) if ttm else (ry[i], N[i], ry[i + 1]))
                if (direct > 0):
                    swp = swp + 1
                    
            max_dx = 0
            direct = -direct
        else:
            i = i + direct

    cores_y[d - 1] = tn.reshape(cry, (ry[d - 1], M[d - 1],
                                N[d - 1], ry[d]) if ttm else (ry[d - 1], N[d - 1], ry[d]))

    # Distribute norms
    nrms = tn.exp(tn.sum(tn.log(nrms)) / d)
    for i in range(d):
        cores_y[i] = cores_y[i] * nrms

    if verb > 0:
        tme_total = datetime.datetime.now() - tme_total
        print("Finished in %s"%(str(tme_total)))
        print()
    return tntt.TT(cores_y)


def compute_phiy_lr(phi_prev, z, y):
    """
    Compute the interface for the core y from left to right.

    Args:
        phi_prev (tn.tensor): thr previus phi having the shape rz x ry.
        z (tn.tensor): the z core with shape rz x n x rz' or rz x m x n x rz'. 
        y (tn.tensor): the y core with shape ry x n x ry' or ry x m x n x ry'.

    Returns:
        the next phi: shape is rz' x ry'.
    """
    if len(z.shape) == 3:
        return oe.contract('znZ,zy,ynY->ZY', z, phi_prev, y)
    else:
        return oe.contract('zmnZ,zy,ymnY->ZY', z, phi_prev, y)


def compute_phiy_rl(phi_prev, z, y):
    """
    Compute the interface for the core y from right to left.

    Args:
        phi_prev (tn.tensor): thr previus phi having the shape rz' x ry'.
        z (tn.tensor): the z core with shape rz x n x rz' or rz x m x n x rz'. 
        y (tn.tensor): the y core with shape ry x n x ry' or ry x m x n x ry'.

    Returns:
        the next phi: shape is rz x ry.
    """
    if len(z.shape) == 3:
        return oe.contract('znZ,YZ,ynY->yz', z, phi_prev, y)
    else:
        return oe.contract('zmnZ,YZ,ymnY->yz', z, phi_prev, y)


