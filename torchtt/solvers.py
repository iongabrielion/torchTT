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

class _SymbolGenerator():
    def __init__(self):
        self.symbol_idx = 0
        self.l = oe.get_symbol(self.symbol_idx)
        self.symbol_idx += 1
        self.L = oe.get_symbol(self.symbol_idx)
        self.symbol_idx += 1
        self.r = oe.get_symbol(self.symbol_idx)
        self.symbol_idx += 1
        self.R = oe.get_symbol(self.symbol_idx)
        self.symbol_idx += 1
        
    def generate_symbol(self):
        s = oe.get_symbol(self.symbol_idx)
        self.symbol_idx += 1
        return s

gen = _SymbolGenerator()

class _LinearOp_prod():
    def __init__(self, Phi_left, Phi_right, Operator, shape, prec):
        self.shape = shape #rnR
        self.prec = prec
        self.summands_number = Operator.summands_number
        self.expressions = []
         # lsr, smnS, LSR, rnR -> lmL
        self.expressions = [oe.contract_expression(Operator.local_product_str[j],
                                                   Phi_left[j], 
                                                   *Operator.data[j], 
                                                   Phi_right[j], 
                                                   shape,
                                                   constants=list(range(len(Operator.data[j]) + 2)),
                                                   optimize='optimal') for j in range(self.summands_number)]
        if prec == 'c':
            J = 0
            for j in range(self.summands_number):
                sl = Operator.s[j] + gen.l
                SL = Operator.S[j] + gen.L
                lLmn = gen.l + gen.L + Operator.smnS[j][1] + Operator.smnS[j][-2]
                J += oe.contract(sl + ',' + Operator.smnS[j] + ',' + SL + '->' + lLmn,
                                tn.diagonal(Phi_left[j], 0, 0, -1), *Operator.data[j], tn.diagonal(Phi_right[j], 0, 0, -1))
            self.J_LU_factors, self.J_pivots = tn.linalg.lu_factor(J)
            
        elif prec == 'r':
            J = 0
            for j in range(self.summands_number):
                sl = Operator.s[j] + gen.l
                LSR = gen.L + Operator.S[j] + gen.R
                lmLnR = gen.l + Operator.smnS[j][1] + gen.L + Operator.smnS[j][-2] + gen.R
                #sl,smnS, LSR-> lmLnR'
                J += oe.contract(sl + ',' + Operator.smnS[j] + ',' + LSR + '->' + lmLnR,
                                tn.diagonal(Phi_left[j], 0, 0, -1), *Operator.data[j], Phi_right[j])
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
            y = x.reshape([self.shape[0], -1, 1]).contiguous() # r(nR)1
            y = tn.linalg.lu_solve(self.J_LU_factors, self.J_pivots, y)
            return y.reshape(x.shape).contiguous()

    def matvec(self, x, apply_prec=True):
        y = x.reshape(self.shape)
        if (self.prec is not None) and apply_prec:
            if self.prec == 'c' or self.prec == 'r':
                y = self.apply_prec(y)
            else:
                raise Exception('Preconditioner '+str(self.prec)+' not defined.')
        y = y.contiguous()
        w = 0
        for j in range(self.summands_number):
            w += self.expressions[j](y)
        
        return tn.reshape(w, [-1, 1])


class _Phi_prod():
    def __init__(self, Matrices, k, d, dtype, device):
        if 0 < k < d:
            self.data = [None] * len(Matrices)
        else:
            # size is rk x Rk x rk
            self.data = [tn.ones(tuple([1] * (len(Matrices[i]) + 2)), dtype=dtype, device=device) for i in range(len(Matrices))]

    def normalize(self):
        norm = sum([tn.linalg.norm(self.data[i]) ** 2 for i in range(len(self.data))])
        norm = tn.sqrt(norm)
        norm = norm if norm > 0 else 1.0
        self.data = [self.data[i] / norm for i in range(len(self.data))]
        return norm
        

class _Operator_prod():
    def __init__(self, Matrices, k, dtype, device, x_shape):
        # Matrices = [(A1, A2, A3), (B1, B2), (C1, C2, C3)]
        self.summands_number = len(Matrices)
        self.local_product_str = []
        self.full_subscripts = []
        self.data = [] # [(A1.cores[k], A2.cores[k], A3.cores[k]), (B1.cores[k], B2.cores[k]), (C1.cores[k], C2.cores[k], C3.cores[k])]
        self.s = []
        self.S = []
        self.smnS = []
        self.k = k
        for j in range(self.summands_number):
            self.data.append(tuple([A.cores[k] for A in Matrices[j]]))
            Rank1 = [M.R[k] for M in Matrices[j]]
            Rank2 = [M.R[k + 1] for M in Matrices[j]]
            
            operands_number = len(Matrices[j])
            ranks_left = ''.join([gen.generate_symbol() for _ in range(operands_number)])
            ranks_right = ''.join([gen.generate_symbol() for _ in range(operands_number)])
            sizes = ''.join([gen.generate_symbol() for _ in range(operands_number + 1)])
                
            smnS = ranks_left[0] + sizes[0] + sizes[1] + ranks_right[0] # s is a set of left, S is a set of right ranks
            for i in range(1, operands_number):
                smnS += ',' + ranks_left[i] + sizes[i] + sizes[i + 1] + ranks_right[i]

            self.s.append(ranks_left)
            self.S.append(ranks_right)
            self.smnS.append(smnS)
            lsr = gen.l + ranks_left + gen.r
            LSR = gen.L + ranks_right + gen.R
            rnR = gen.r + sizes[-1] + gen.R
            lmL = gen.l + sizes[0] + gen.L
            #lsr, smnS, LSR -> lmLrnR
            self.full_subscripts.append(lsr + ',' + smnS + ',' + LSR + '->' + lmL + rnR)
            # lsr, smnS, LSR, rnR -> lmL
            subscripts = lsr + ',' + smnS + ','  + LSR + ',' + rnR + '->' + lmL
            self.local_product_str.append(subscripts)
            
    def get_full_operator(self, Phis_left, Phis_right, size):
        B = 0
        for j in range(self.summands_number):
            B += oe.contract(self.full_subscripts[j], #lsr, smnS, LSR -> lmLrnR
                                 Phis_left[j], 
                                 *self.data[j],
                                 Phis_right[j])
        return B

    def local_product(self, Phi_right, Phi_left, x):
        """
        Compute local matvec product
        
        Args:
         Phi (torch.tensor): right tensor of shape r x R x r.
         Psi (torch.tensor): left tensor of shape lp x Rp x lp.
         coreA (torch.tensor): current core of A, shape is rp x N x N x r.
         x (torch.tensor): the current core of x, shape is rp x N x r.
        
        Returns:
         torch.tensor: the result.
        """
        w = 0
        for j in range(self.summands_number):
            w += oe.contract(self.local_product_str[j], Phi_left[j],  *self.data[j], Phi_right[j], x)
        return w


    def compute_phi_A_fwd(self, Phi_bck, core_left, core_right):
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
        #lmL, smnS, rnR, lsr -> LSR
        subscripts = [gen.l + self.smnS[j][1] + gen.L + ',' + self.smnS[j] + ',' +
                      gen.r + self.smnS[j][-2] + gen.R + ',' + gen.l + self.s[j] + gen.r + '->' + gen.L + self.S[j] + gen.R
                      for j in range(self.summands_number)]
        with oe.shared_intermediates():
            Phi_fwd = [oe.contract(subscripts[j],
                                    core_left,
                                    *self.data[j],
                                    core_right, Phi_bck[j]) for j in range(self.summands_number)]
        return Phi_fwd
    
    def compute_phi_A_bck(self, Phi_fwd, core_left, core_right):
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
        #lmL, smnS, rnR, LSR -> lsr
        core_left = core_left.contiguous()
        core_right = core_right.contiguous()
        subscripts = [gen.l + self.smnS[j][1] + gen.L + ',' + self.smnS[j] + ',' +
                      gen.r + self.smnS[j][-2] + gen.R + ',' + gen.L + self.S[j] + gen.R + '->' + gen.l + self.s[j] + gen.r
                      for j in range(self.summands_number)]
        Phi_bck = [oe.contract(subscripts[j],
                                core_left,
                                *self.data[j],
                                core_right, Phi_fwd[j]) for j in range(self.summands_number)]
        return Phi_bck


class _LinearOp_sum():
    def __init__(self, Phi_left_list, Phi_right_list, Operator, shape, prec):
        self.shape = shape
        self.prec = prec
        self.summands_number = Operator.summands_number
        self.bands = Operator.bands
        if prec is not None:
            if prec == 'c' or prec == 'r':
                self.prec = prec
            else:
                raise Exception('Preconditioner ' + str(prec) + ' not defined.')
        #self.summ_data = Summ_data
        self.expressions = []
        for j in range(Operator.summands_number):
            band = Operator.bands[j]
            sizes = 'lsr, smnS, LSR, rnR-> lmL' if band < 0 else 'lsr, ksSn, LSR, rnR -> klnL'
            expr = oe.contract_expression(sizes,
                                          Phi_left_list[j].contiguous(), Operator.data[j].contiguous(), Phi_right_list[j].contiguous(), 
                                          shape,
                                          constants=[0, 1, 2], optimize='optimal')
            self.expressions.append(expr)
        if prec == 'c':
            J = tn.zeros(Phi_left_list[0].shape[-1], Phi_right_list[0].shape[-1], shape[1], shape[1],
                              dtype=Phi_left_list[0].dtype, device=Phi_left_list[0].device)
            for j in range(Operator.summands_number):
                band = Operator.bands[j]
                if band < 0:
                    J += oe.contract('sr, smnS, SR -> rRmn',
                                     tn.diagonal(Phi_left_list[j], 0, 0, 2).contiguous(),
                                     Operator.data[j],
                                     tn.diagonal(Phi_right_list[j], 0, 0, 2).contiguous())
                else:
                    diags_J = oe.contract('sr, ksSn, SR -> krRn',
                                     tn.diagonal(Phi_left_list[j], 0, 0, 2).contiguous(),
                                     Operator.data[j],
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
            for j in range(Operator.summands_number):
                band = Operator.bands[j]
                if band < 0:
                    J += oe.contract('sr, smnS, LSR- > rmLnR', 
                                     tn.diagonal(Phi_left_list[j], 0, 0, 2).contiguous(),
                                     Operator.data[j], 
                                     Phi_right_list[j])
                else:
                    diags_J = oe.contract('sr, ksSn, LSR -> krLRn',
                                          tn.diagonal(Phi_left_list[j], 0, 0, 2).contiguous(),
                                          Operator.data[j], 
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
            y = x.reshape([self.shape[0], -1, 1]).contiguous() # r(nR)1
            y = tn.linalg.lu_solve(self.J_LU_factors, self.J_pivots, y)
            return y.reshape(x.shape).contiguous()

    def matvec(self, x, apply_prec=True):
        y = x.reshape(self.shape)
        w = 0
        if (self.prec is not None) and apply_prec:
            y = self.apply_prec(y)
        y = y.contiguous()
        for j in range(self.summands_number):
            band = self.bands[j]
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


class _Operator_sum():
    def __init__(self, Matrices, bandsMatrices, k, dtype, device):
        # Matrices = [A1, A2, ..., An]
        self.summands_number = len(Matrices)
        self.k = k
        self.data = [] # [A1.cores[k], A2.cores[k], ..., An.cores[k]]
        self.dtype = dtype
        self.device = device
        for i in range((len(Matrices))):
            core = Matrices[i].cores[k]
            band = bandsMatrices[i, k]
            if band < 0:
                self.data.append(core.clone())
            else:
                self.data.append(tn.stack([tnf.pad(tn.diagonal(core, i, 1, 2), (0, -i)).contiguous() for i in range(-band, 0)] + 
                                           [tnf.pad(tn.diagonal(core, i, 1, 2), (i, 0)).contiguous() for i in range(0, band + 1)]))
        self.bands = bandsMatrices[:, k]
            
                

    def get_full_operator(self, Phis_left, Phis_right, sizes):
        B = tn.zeros(sizes, dtype=self.dtype, device=self.device)
        for j in range(self.summands_number):
            band = self.bands[j]
            if band < 0:
                Bp = oe.contract('smnS, LSR -> smnRL', self.data[j], Phis_right[j])
                B += oe.contract('lsr,smnRL->lmLrnR', Phis_left[j], Bp)
            else:
                diags_Bp = oe.contract('ksSn, LSR -> ksRLn', self.data[j], Phis_right[j])
                diags_B = oe.contract('lsr, ksRLn -> klLrRn', Phis_left[j], diags_Bp)
                for i in range(-band, band + 1):
                    diag_i_B = tn.diagonal(B, i, 1, 4)
                    if i >= 0:
                        diag_i_B += diags_B[i + band, ..., i:]
                    else:
                        diag_i_B += diags_B[i + band, ..., :i]
        return B

    def local_product(self, Phi_right_list, Phi_left_list, x):
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
        for j in range(self.summands_number):
            band = self.bands[j]
            if band < 0:
                # data is a full core
                w += oe.contract('lsr, smnS, LSR, rnR -> lmL', 
                                  Phi_left_list[j], self.data[j], Phi_right_list[j], x)
            else:
                # data is diagonals of a core
                tmp = oe.contract('lsr, ksSn, LSR, rnR -> klnL',
                                   Phi_left_list[j], self.data[j], Phi_right_list[j], x)
                w += tmp[band, ...]
                for i in range(1, band + 1):
                    w[:, :-i, :] += tmp[i + band, :, i:, :]
                    w[:, i:, :] += tmp[-i + band, :, :-i, :]
        return w


    def compute_phi_A_bck(Operator, Phi_now_list, core_left, core_right):
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
        band_max = max(Operator.bands)
        if band_max >= 0:
            cores_left = tn.stack([tnf.pad(core_left[:, -i:, :], (0, 0, 0, -i)) for i in range(-band_max, 1)] + 
                                    [tnf.pad(core_left[:, :-i, :], (0, 0, i, 0)) for i in range(1, band_max + 1)])
        for j in range(Operator.summands_number):
            band = Operator.bands[j]
            if band < 0:
                # data is a full core
                Phi_list.append(oe.contract('LSR, lmL, smnS, rnR -> lsr',
                                            Phi_now_list[j], core_left, Operator.data[j], core_right))
            else:            
                # data is diagonals of a core
                Phi_list.append(oe.contract('LSR, klnL, ksSn, rnR -> lsr',
                                            Phi_now_list[j], cores_left[band_max-band:band_max+band+1, ...], Operator.data[j], core_right))
        return Phi_list
    
    
    def compute_phi_A_fwd(Operator, Phi_now_list, core_left, core_right):
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
        core_right_tmp = tn.transpose(core_right, 0, 2).contiguous()
        core_left_tmp = tn.transpose(core_left, 0, 2).contiguous()
        band_max = max(Operator.bands)
        if band_max >= 0:
            cores_left = tn.stack([tnf.pad(core_left[:, -i:, :], (0, 0, 0, -i)) for i in range(-band_max, 1)] + 
                                    [tnf.pad(core_left[:, :-i, :], (0, 0, i, 0)) for i in range(1, band_max + 1)])
        for j in range(Operator.summands_number):
            band = Operator.bands[j]
            if band < 0:
                # data is a full core
                Phi_list.append(oe.contract('lsr, Lml, smnS, Rnr -> LSR',
                                            Phi_now_list[j], core_left_tmp, Operator.data[j], core_right_tmp))
            else:            
                # data is diagonals of a core
                Phi_list.append(oe.contract('lsr, klnL, ksSn, rnR -> LSR',
                                            Phi_now_list[j], cores_left[band_max-band:band_max+band+1, ...], Operator.data[j], core_right))
        return Phi_list


class _Phi_sum():
    def __init__(self, Matrices, k, d, dtype, device):
        if 0 < k < d:
            self.data = [None] * len(Matrices)
        else:
            # size is rk x Rk x rk
            self.data = [tn.ones((1, 1, 1), dtype=dtype, device=device) for i in range(len(Matrices))]

    def normalize(self):
        norm = sum([tn.linalg.norm(self.data[i]) ** 2 for i in range(len(self.data))])
        norm = tn.sqrt(norm)
        norm = norm if norm > 0 else 1.0
        self.data = [self.data[i] / norm for i in range(len(self.data))]
        return norm
    
        
def amen_solve_sum(Matrices, b, nswp=22, x0=None, eps=1e-10, rmax=32768, max_full=500, kickrank=4, kick2=0, trunc_norm='res', local_solver=1, local_iterations=40, resets=2, verbose=False, preconditioner=None, use_cpp=True, use_single_precision=False, bandsMatrices=None):
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
                
        return _amen_solve_python(Matrices, b, nswp, x0, eps, rmax, max_full, kickrank, kick2, trunc_norm, local_solver, local_iterations, resets, verbose, preconditioner, use_single_precision, np.array(bandsMatrices), 'sum')
        
    
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

    Raises:
        InvalidArguments: A and b must be TT instances.
        InvalidArguments: Invalid preconditioner.
        IncompatibleTypes: A must be TT-matrix and b must be vector.
        ShapeMismatch: A is not quadratic.
        ShapeMismatch: Dimension mismatch.

    Returns:
        torchtt.TT: the approximation of the solution in TT format.
    """
    
    # Matrices = [(A1, A2, A3), (B1, B2), (C1, C2, C3)]
    # A = A1 @ A2 @ A3 + B1 @ B2 + C1 @ C2 @ C3
    
    #perform checks of the input data
    ord = 'prod'
    if isinstance(Matrices, torchtt.TT):
        Matrices = [Matrices, ]
        if bandsMatrices is not None:
            bandsMatrices = [bandsMatrices,]
        ord = 'sum'
    if isinstance(Matrices[0], torchtt.TT):
        ord = 'sum'
    if not isinstance(b, torchtt.TT):
        raise InvalidArguments('b must be a TT instance.')
    if b.is_ttm:
        raise IncompatibleTypes(' b must be a vector.')
    if ord == 'prod':
        for i in range(len(Matrices)):
            for j in range(len(Matrices[i])):
                A = Matrices[i][j]
                if not isinstance(A, torchtt.TT):
                     raise InvalidArguments(f'Matrix {i+1}-th {j+1}-th must a TT instance.')
                if not A.is_ttm:
                    raise IncompatibleTypes(f'Matrix {i+1}-th {j+1}-th  must be a TT-matrix.')
                if i > 0:
                    if (j > 0):
                        if A.M != Matrices[i][j - 1].N:
                            raise ShapeMismatch('Matrix shapes must match for product.')
                    else:
                        if (i > 0):
                            if A.M != Matrices[i - 1][j - 1].M:
                                raise ShapeMismatch('Matrix shapes must the same for sum.')
            if Matrices[i][0].M != Matrices[i][-1].N:
                raise IncompatibleTypes('Matrices must be square')
        if Matrices[0][-1].N != b.N:
            raise ShapeMismatch('Dimension mismatch.')
    else:
        for i in range(len(Matrices)):
            A = Matrices[i]
            if not isinstance(A, torchtt.TT):
                 raise InvalidArguments(f'Matrix {i+1}-th must a TT instance.')
            if not Matrices[i].is_ttm:
                raise IncompatibleTypes(f'Matrix {i+1}-th must be a TT-matrix.')
            if A.M != A.N:
                raise ShapeMismatch(f'Matrix {i+1}-th is not quadratic.')
            if i > 0:
                if A.M != Matrices[i - 1].M:
                    raise ShapeMismatch('Matrix shapes must be the same.')
        if Matrices[-1].N != b.N:
            raise ShapeMismatch('Dimension mismatch.')
    

    if use_cpp and _flag_use_cpp:
        A = 0
        if ord == 'prod':
            for i in range(len(Matrices)):
                B = Matrices[i][0]
                for j in range(1, len(Matrices[i])):
                    B = B @ Matrices[i][j]
                A += B.round(eps)
        else:
            for i in range(len(Matrices)):
                A += Matrices[i].round(eps)
        if x0 == None:
            x_cores = []
            x_R = [1]*(1 + len(A.N))
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
    else:
        if ord == 'sum':
            if bandsMatrices == None:
                bandsMatrices = [[-1] * len(Matrices[0].N)] * len(Matrices)
            else:
                assert(len(Matrices) == len(bandsMatrices))
                for i in range(len(Matrices)):
                    if bandsMatrices[i] == None:
                        bandsMatrices = [[-1] * len(Matrices[0].N)] * len(Matrices)
                        break
            bandsMatrices = np.array(bandsMatrices)
        return _amen_solve_python(Matrices, b, nswp, x0, eps, rmax, max_full, kickrank, kick2, trunc_norm, local_solver, local_iterations, resets, verbose, preconditioner, use_single_precision, bandsMatrices, ord)


def _amen_solve_python(Matrices, b, nswp=22, x0=None, eps=1e-10, rmax=1024, max_full=500, kickrank=4, kick2=0, trunc_norm='res', local_solver=1, local_iterations=40, resets=2, verbose=False, preconditioner=None, use_single_precision=False, bandsMatrices=None, ord='sum'):

    N = b.N.copy()
    d = len(N)
    
    if verbose:
        time_total = datetime.datetime.now()

    dtype = b.cores[0].dtype
    device = b.cores[0].device
    damp = 2

    if x0 is None:
        x0 = torchtt.ones(b.N, dtype=dtype, device=device)
    x_cores = x0.cores.copy()
    rx = x0.R.copy()
    
    # check if rmax is a list
    if isinstance(rmax, int):
        rmax = [1,] + (d - 1) * [rmax] + [1,]

    if ord == 'prod':
        Phis = [_Phi_prod(Matrices, k, d, dtype, device) for k in range(0, d + 1)]
        Phiz = [_Phi_prod(Matrices, k, d, dtype, device) for k in range(0, d + 1)]
    elif ord == 'sum':
        Phis = [_Phi_sum(Matrices, k, d, dtype, device) for k in range(0, d + 1)]
        Phiz = [_Phi_sum(Matrices, k, d, dtype, device) for k in range(0, d + 1)]

    Phiz_b = [tn.ones((1, 1), dtype=dtype, device=device)] +  [None] * (d-1) + \
             [tn.ones((1, 1), dtype=dtype, device=device)]   # size is rzk x rzbk
    
    Phis_b = [tn.ones((1, 1), dtype=dtype, device=device)] + [None] * (d-1) + \
             [tn.ones((1, 1), dtype=dtype, device=device)]  # size is rk x rbk

    if ord == 'prod':
        Operators = [_Operator_prod(Matrices, k, dtype, device, x_cores[k].shape) for k in range(d)]
    elif ord == 'sum':
         Operators = [_Operator_sum(Matrices, bandsMatrices, k, dtype, device) for k in range(d)]

    # z cores
    rz = [1] + (d - 1) * [kickrank + kick2]+ [1]
    z_cores = torchtt.random(N, rz, dtype, device=device).cores
    z_cores, rz = rl_orthogonal(z_cores, rz, False)
    
    Phiz_b = [tn.ones((1, 1), dtype=dtype, device=device)] +  [None] * (d - 1) + \
             [tn.ones((1, 1), dtype=dtype, device=device)]   # size is rzk x rzbk

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

        # left to right orthogonalization
        for k in range(d - 1, 0, -1):
            # update the z part (ALS) update
            if not last:
                if swp > 0:
                    # shape rzp x N x rz
                    czA = Operators[k].local_product(Phiz[k + 1].data, Phiz[k].data,  x_cores[k])
                    # shape is rzp x N x rz
                    czy = tn.einsum('br, bnB, BR -> rnR',
                                    Phiz_b[k], b.cores[k], Phiz_b[k + 1])
                    cz_new = czy * nrmsc - czA
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

            core = tn.reshape(x_cores[k], [rx[k], N[k] * rx[k+1]]).t()
            Qmat, Rmat = QR(core)

            core_prev = tn.einsum('ijk, mk -> ijm', x_cores[k - 1], Rmat)
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
            Phis[k].data = Operators[k].compute_phi_A_bck(Phis[k + 1].data, x_cores[k], x_cores[k])
            Phis_b[k] = _compute_phi_bck_rhs(Phis_b[k + 1], b.cores[k], x_cores[k])

            # ... and norms
            normA[k - 1] = Phis[k].normalize()
            norm = tn.linalg.norm(Phis_b[k])
            norm = norm if norm > 0 else 1.0
            normb[k - 1] = norm
            Phis_b[k] = Phis_b[k] / norm

            # norm correction
            nrmsc = nrmsc * normb[k - 1] / (normA[k - 1] * normx[k - 1])

            # compute phis_z
            if not last:
                Phiz[k].data = Operators[k].compute_phi_A_bck(Phiz[k + 1].data, z_cores[k], x_cores[k])
                Phiz[k].data = [Phiz[k].data[i] / normA[k - 1] for i in range(len(Matrices))]
                Phiz_b[k] = _compute_phi_bck_rhs(
                    Phiz_b[k + 1], b.cores[k], z_cores[k]) / normb[k - 1]

        # start loop
        max_res = 0
        max_dx = 0

        for k in range(d):
            if verbose:
                print('\tCore', k)
            previous_solution = tn.reshape(x_cores[k], [-1, 1])

            # assemble rhs
            rhs = tn.einsum('br, bmB, BR -> rmR',
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
                          (rx[k]*N[k]*rx[k+1]))
                # shape is Rp x N x N x r x r
                B = Operators[k].get_full_operator(Phis[k].data, Phis[k + 1].data, [rx[k], N[k], rx[k + 1], rx[k], N[k], rx[k + 1]])
                B = tn.reshape(B, [rx[k] * N[k] * rx[k + 1], rx[k] * N[k] * rx[k + 1]])

                solution_now = tn.linalg.solve(B, rhs)

                res_old = tn.linalg.norm(B @ previous_solution - rhs) / norm_rhs
                res_new = tn.linalg.norm(B @ solution_now - rhs) / norm_rhs
            else:
                # iterative solver
                if verbose:
                    print('\t\tChoosing iterative solver %s (local size %d)....' % (
                        'GMRES' if local_solver == 1 else 'BiCGSTAB_reset', rx[k]*N[k]*rx[k+1]))
                    time_local = datetime.datetime.now()

                shape_now = [rx[k], N[k], rx[k + 1]]
                if ord == 'prod':
                    Op = _LinearOp_prod(Phis[k].data, Phis[k + 1].data,
                                       Operators[k],
                                       shape_now, preconditioner)
                elif ord == 'sum':
                    Op = _LinearOp_sum(Phis[k].data, Phis[k + 1].data,
                                   Operators[k],
                                   shape_now, preconditioner)
                if use_single_precision:

                    # solution_now, flag, nit, res_new = BiCGSTAB_reset(Op, rhs,previous_solution[:], eps_local, local_iterations)
                    eps_local = real_tol * norm_rhs
                    drhs = Op.matvec(previous_solution.to(tn.float32), False)
                    drhs = rhs.to(tn.float32 ) -drhs
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
                        tn.float32), False).to(dtype)-rhs)/norm_rhs
                    res_new = tn.linalg.norm(Op.matvec(solution_now.to(
                        tn.float32), False).to(dtype)-rhs)/norm_rhs
                else:
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
                    res_new = tn.linalg.norm(
                        Op.matvec(solution_now, False) - rhs) / norm_rhs

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
            dx = tn.linalg.norm(solution_now - previous_solution) / tn.linalg.norm(solution_now)
            if verbose:
                print('\t\tdx = %g, res_now = %g, res_old = %g' %
                      (dx, res_new, res_old))

            max_dx = max(dx, max_dx)
            max_res = max(max_res, res_old)

            solution_now = tn.reshape(solution_now, [rx[k] * N[k], rx[k + 1]])
            # truncation
            if k < d - 1:
                u, s, v = SVD(solution_now)
                if trunc_norm == 'fro':
                    pass
                else:
                    # search for a rank such that offeres small enough residuum
                    # TODO: binary search?
                    r = 0
                    for r in range(u.shape[1] - 1, 0, -1):
                        # solution has the same size
                        solution = u[:, :r] @ tn.diag(s[:r]) @ v[:r, :]

                        if use_full:
                            res = tn.linalg.norm(
                                B @ tn.reshape(solution, [-1, 1]) - rhs) / norm_rhs
                        else:
                            res = tn.linalg.norm(Op.matvec(solution.to(
                                tn.float32 if use_single_precision else dtype)).to(dtype) - rhs) / norm_rhs
                        if res > max(real_tol * damp, res_new):
                            break
                    r += 1

                    r = min([r, tn.numel(s), rmax[k + 1]])
                    u = u[:, :r]
                    v = tn.diag(s[:r]) @ v[:r, :]
                    v_norm = tn.linalg.norm(tn.diag(s[:r]))
                    solution = tn.reshape(u @ v, [rx[k], N[k], rx[k + 1]])
                    
            else:
                solution = tn.reshape(solution_now, [rx[k], N[k], rx[k + 1]])
                x_cores[k] = solution

            if not last:
                czA = Operators[k].local_product(Phiz[k + 1].data, Phiz[k].data, solution)  # shape rzp x N x rz

                # shape is rzp x N x rz
                czy = tn.einsum('br, bnB ,BR->rnR',
                                Phiz_b[k], b.cores[k] * nrmsc, Phiz_b[k + 1])
                cz_new = czy - czA

                uz, _, _ = SVD(tn.reshape(cz_new, [rz[k] * N[k], rz[k + 1]]))
                # truncate to kickrank
                cz_new = uz[:, :min(kickrank, uz.shape[1])]
                if k < d - 1 :  # extend cz_new with random elements
                    cz_new = tn.cat(
                        (cz_new, tn.randn((cz_new.shape[0], kick2),  dtype=dtype, device=device)), 1)

                qz, _ = QR(cz_new)
                rz[k + 1] = qz.shape[1]
                z_cores[k] = tn.reshape(qz, [rz[k], N[k], rz[k + 1]]).contiguous()

            if k < d - 1:
                if not last:
                    left_res = Operators[k].local_product(Phiz[k + 1].data, Phis[k].data, solution)
                    left_b = tn.einsum('br, bmB, BR -> rmR', 
                                       Phis_b[k], b.cores[k] * nrmsc, Phiz_b[k + 1])
                    uk = left_b - left_res  # rx_k x N_k x rz_k+1
                    u, Rmat = QR(tn.cat((u, tn.reshape(uk, [u.shape[0], -1])), 1))
                    v = Rmat[:, :v.shape[0]] @ v

                r = u.shape[1]
                v = tn.einsum('ij, jkl -> ikl', v, x_cores[k + 1]) # x_cores[k + 1] is orthogonal matrix
                # remove norm correction
                nrmsc = nrmsc * normA[k]  / normb[k]

                if v_norm > 0:
                    v = v / v_norm
                else:
                    v_norm = 1.0
                normx[k] = normx[k] * v_norm

                x_cores[k] = tn.reshape(u, [rx[k], N[k], r]).contiguous()
                x_cores[k + 1] = tn.reshape(v, [r, N[k + 1], rx[k + 2]]).contiguous()
                rx[k + 1] = r

                # next phis with norm correction
                Phis[k + 1].data = Operators[k].compute_phi_A_fwd(Phis[k].data, x_cores[k], x_cores[k])
                Phis_b[k + 1] = _compute_phi_fwd_rhs(Phis_b[k], b.cores[k], x_cores[k])

                # ... and norms
                normA[k] = Phis[k + 1].normalize()
                norm = tn.linalg.norm(Phis_b[k + 1])
                norm = norm if norm > 0 else 1.0
                normb[k] = norm
                Phis_b[k + 1] = Phis_b[k + 1] / norm

                # norm correction
                nrmsc = nrmsc * normb[k] / (normA[k] * v_norm)

                # next phiz
                if not last:
                    Phiz[k + 1].data = Operators[k].compute_phi_A_fwd(Phiz[k].data, z_cores[k], x_cores[k])
                    Phiz[k + 1].data = [Phiz[k + 1].data[i] / normA[k] for i in range(len(Phiz[k + 1].data))]
                    Phiz_b[k + 1] = _compute_phi_fwd_rhs(Phiz_b[k], b.cores[k], z_cores[k]) / normb[k]
        
        if verbose:
            print('Solution rank is', rx)
            print('Maxres ', max_res)
            tme_sweep = datetime.datetime.now() - tme_sweep
            print('Time ', tme_sweep)

        if last:
            break

        last = max_res < eps

    if verbose:
        time_total = datetime.datetime.now() - time_total
        print()
        print('Finished after', swp+1, ' sweeps and ', time_total)
        print()
    normx = np.exp(np.sum(np.log(normx)) / d)

    for k in range(d):
        x_cores[k] = x_cores[k] * normx

    return torchtt.TT(x_cores)


def _compute_phi_bck_rhs(Phi_now, core_b, core):
    """


    Args:
        Phi_now (torch.tensor): The current phi. Has shape rb_k+1 x r_k+1
        core_b (torch.tensor): The current core of the rhs. Has shape rb_k x N_k x rb_k+1
        core (torch.tensor): The current core. Has shape r_k x N_k x r_k+1

    Returns:
        torch.tensor: The backward phi corresponding to the rhs. Has shape rb_k x r_k
    """
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
    Phi_next = tn.einsum('br,bnB,rnR->BR', Phi_now, core_rhs, core)
    return Phi_next
