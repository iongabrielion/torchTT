import torch as tn 
import opt_einsum as oe 
import torch.nn.functional as tnf

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
            #path = oe.contract_path('rab,amkA,bknB,RAB->rmnR', Phi_left, coreA, coreB, Phi_right)
            #print(path[1])
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


class mv_local_op():

    def __init__(self, A, x, M, N, has_z=True):

        self.A = A
        self.x = x
        self.M = M
        self.N = N
        self.has_z = has_z

        self.phis_y = [tn.ones([1, 1, 1], device=A.cores[0].device, dtype=A.cores[0].dtype)] + \
            [None]*(len(M)-1) + [tn.ones([1, 1, 1], device=A.cores[0].device, dtype=A.cores[0].dtype)]
        if has_z:
            self.phis_z = [tn.ones([1, 1, 1], device=A.cores[0].device, dtype=A.cores[0].dtype)] + [None]*(
                len(M)-1) + [tn.ones([1, 1, 1], device=A.cores[0].device, dtype=A.cores[0].dtype)]


    def b_fun(self, k, first, second):
        # print()
        # print(k, first, second)
        # path_info = oe.contract_path('yax,YAX,amnA,xnX->ymY', self.phis_y[k] if first == 'y' else self.phis_z[k], self.phis_y[k+1] if second == 'y' else self.phis_z[k+1], self.A.cores[k], self.x.cores[k])
        # print(path_info[1])
        return oe.contract('yax,YAX,amnA,xnX->ymY', self.phis_y[k] if first == 'y' else self.phis_z[k], self.phis_y[k+1] if second == 'y' else self.phis_z[k+1], self.A.cores[k], self.x.cores[k])

    def update_phi_z(self, z, k, mode, norm, return_norm):

        if mode == 'rl':
            phi = oe.contract(
                'ZAX,amnA,xnX,zmZ->zax', self.phis_z[k+1], self.A.cores[k], self.x.cores[k], z)
            if return_norm:
                nrm_phi = tn.linalg.norm(phi)
                norm *= nrm_phi
            else:
                nrm_phi = 1
            if norm != 1:
                phi = phi / norm
            self.phis_z[k] = phi.clone()
        else:
            # print()
            # print('Z', mode)
            # path_info = oe.contract_path('zax,zmZ,amnA,xnX->ZAX',
            #                   self.phis_z[k], z, self.A.cores[k], self.x.cores[k]); 
            # print(path_info[1])
            
            phi = oe.contract('zax,zmZ,amnA,xnX->ZAX',
                              self.phis_z[k], z, self.A.cores[k], self.x.cores[k])
            if return_norm:
                nrm_phi = tn.linalg.norm(phi)
                norm *= nrm_phi
            else:
                nrm_phi = 1
            if norm != 1:
                phi = phi / norm
            self.phis_z[k+1] = phi.clone()
        return nrm_phi

    def update_phi_y(self, y, k, mode, norm, return_norm):

        if mode == 'rl':
            phi = oe.contract(
                'YAX,amnA,xnX,ymY->yax', self.phis_y[k+1], self.A.cores[k], self.x.cores[k], y)
            if return_norm:
                nrm_phi = tn.linalg.norm(phi)
                norm *= nrm_phi
            else:
                nrm_phi = 1
            if norm != 1:
                phi = phi / norm
            self.phis_y[k] = phi.clone()
        else:
            phi = oe.contract('yax,ymY,amnA,xnX->YAX',
                              self.phis_y[k], y, self.A.cores[k], self.x.cores[k])
            if return_norm:
                nrm_phi = tn.linalg.norm(phi)
                norm *= nrm_phi
            else:
                nrm_phi = 1
            if norm != 1:
                phi = phi / norm
            self.phis_y[k+1] = phi.clone()
        return nrm_phi

class mv_multiple_local_op():

    def __init__(self, A, xs, M, N, has_z=True):

        self.A = A
        self.xs = xs
        self.M = M
        self.N = N
        self.has_z = has_z
        self.len = len(xs)

        self.phis_y = [[tn.ones([1, 1, 1], device=A.cores[0].device, dtype=A.cores[0].dtype)] + [None]*(len(M)-1) + [tn.ones([1, 1, 1], device=A.cores[0].device, dtype=A.cores[0].dtype)] for i in range(self.len)]
        if has_z:
            self.phis_z = [[tn.ones([1, 1, 1], device=A.cores[0].device, dtype=A.cores[0].dtype)] + [None]*(len(M)-1) + [tn.ones([1, 1, 1], device=A.cores[0].device, dtype=A.cores[0].dtype)] for i in range(self.len)]


    def b_fun(self, k, first, second):
        # print()
        # print(k, first, second)
        # path_info = oe.contract_path('yax,YAX,amnA,xnX->ymY', self.phis_y[k] if first == 'y' else self.phis_z[k], self.phis_y[k+1] if second == 'y' else self.phis_z[k+1], self.A.cores[k], self.x.cores[k])
        # print(path_info[1])
        ss = [oe.contract('yax,YAX,amnA,xnX->ymY', self.phis_y[i][k] if first == 'y' else self.phis_z[i][k], self.phis_y[i][k+1] if second == 'y' else self.phis_z[i][k+1], self.A.cores[k], self.xs[i].cores[k]) for i in range(self.len)]
        return sum(ss)
    
    def update_phi_z(self, z, k, mode, norm, return_norm):

        if mode == 'rl':
            nrms = []
            for i in range(self.len):
                phi = oe.contract(
                    'ZAX,amnA,xnX,zmZ->zax', self.phis_z[i][k+1], self.A.cores[k], self.xs[i].cores[k], z)
                if return_norm:
                    nrm_phi = tn.linalg.norm(phi)
                    nrms.append(nrm_phi)

                if norm != 1:
                    phi = phi / norm
                self.phis_z[i][k] = phi.clone()
            if return_norm:
                nrm_phi = max(nrms)
                for i in range(self.len):
                    self.phis_z[i][k] /= nrm_phi
            else:
                nrm_phi = 1
        else:
            nrms = []
            for i in range(self.len):
                phi = oe.contract('zax,zmZ,amnA,xnX->ZAX',
                                self.phis_z[i][k], z, self.A.cores[k], self.xs[i].cores[k])
                if return_norm:
                    nrm_phi = tn.linalg.norm(phi)
                    nrms.append(nrm_phi)
                if norm != 1:
                    phi = phi / norm
                self.phis_z[i][k+1] = phi.clone()
            
            if return_norm:
                nrm_phi = max(nrms)
                for i in range(self.len):
                    self.phis_z[i][k+1] /= nrm_phi
            else:
                nrm_phi = 1
        return nrm_phi

    def update_phi_y(self, y, k, mode, norm, return_norm):

        if mode == 'rl':
            nrms = []
            for i in range(self.len):
                phi = oe.contract(
                    'YAX,amnA,xnX,ymY->yax', self.phis_y[i][k+1], self.A.cores[k], self.xs[i].cores[k], y)
                if return_norm:
                    nrm_phi = tn.linalg.norm(phi)
                    nrms.append(nrm_phi)

                if norm != 1:
                    phi = phi / norm
                self.phis_y[i][k] = phi.clone()
            
            if return_norm:
                nrm_phi = max(nrms)
                for i in range(self.len):
                    self.phis_y[i][k] /= nrm_phi
        else:
            nrms = []
            for i in range(self.len): 
                phi = oe.contract('yax,ymY,amnA,xnX->YAX',
                                self.phis_y[i][k], y, self.A.cores[k], self.xs[i].cores[k])
                if return_norm:
                    nrm_phi = tn.linalg.norm(phi)
                    nrms.append(nrm_phi)

                if norm != 1:
                    phi = phi / norm
                self.phis_y[i][k+1] = phi.clone()
            if return_norm:
                nrm_phi = max(nrms)
                for i in range(self.len):
                    self.phis_y[i][k+1] /= nrm_phi
        return nrm_phi

class mvm_multiple_local_op():

    def __init__(self, As, xs, Bs, M, N, has_z=True):

        self.As = As
        self.xs = xs
        self.Bs = Bs
        self.M = M
        self.N = N
        self.has_z = has_z
        self.len = len(xs)
        dt = As[0].cores[0].dtype
        dev = As[0].cores[0].device
        self.phis_y = [[tn.ones([1, 1, 1, 1], device=dev, dtype=dt)] + [None]*(len(M)-1) + [tn.ones([1, 1, 1, 1], device=dev, dtype=dt)] for i in range(self.len)]
        if has_z:
            self.phis_z = [[tn.ones([1, 1, 1, 1], device=dev, dtype=dt)] + [None]*(len(M)-1) + [tn.ones([1, 1, 1, 1], device=dev, dtype=dt)] for i in range(self.len)]


    def b_fun(self, k, first, second):
        # i = 0
        # print()
        # print(k, first, second)
        # path_info = oe.contract_path('yabc,YABC,amkA,bkB,cknC->ymnY', self.phis_y[i][k] if first == 'y' else self.phis_z[i][k], self.phis_y[i][k+1] if second == 'y' else self.phis_z[i][k+1], self.As[i].cores[k], self.xs[i].cores[k], self.Bs[i].cores[k])
        # print(path_info[1])
        ss = [oe.contract('yabc,YABC,amkA,bkB,cknC->ymnY', self.phis_y[i][k] if first == 'y' else self.phis_z[i][k], self.phis_y[i][k+1] if second == 'y' else self.phis_z[i][k+1], self.As[i].cores[k], self.xs[i].cores[k], self.Bs[i].cores[k]) for i in range(self.len)]

        return sum(ss)

    def update_phi_z(self, z, k, mode, norm, return_norm):

        if mode == 'rl':
            nrms = []
            for i in range(self.len):
                phi = oe.contract(
                    'ZABC,amkA,bkB,cknC,zmnZ->zabc', self.phis_z[i][k+1], self.As[i].cores[k], self.xs[i].cores[k], self.Bs[i].cores[k], z)
                if return_norm:
                    nrm_phi = tn.linalg.norm(phi)
                    nrms.append(nrm_phi)

                if norm != 1:
                    phi = phi / norm
                self.phis_z[i][k] = phi.clone()
            if return_norm:
                nrm_phi = max(nrms)
                for i in range(self.len):
                    self.phis_z[i][k] /= nrm_phi
            else:
                nrm_phi = 1
        else:
            nrms = []
            for i in range(self.len):
                phi = oe.contract('zabc,zmnZ,amkA,bkB,cknC->ZABC',
                                self.phis_z[i][k], z, self.As[i].cores[k], self.xs[i].cores[k], self.Bs[i].cores[k])
                if return_norm:
                    nrm_phi = tn.linalg.norm(phi)
                    nrms.append(nrm_phi)
                if norm != 1:
                    phi = phi / norm
                self.phis_z[i][k+1] = phi.clone()
            
            if return_norm:
                nrm_phi = max(nrms)
                for i in range(self.len):
                    self.phis_z[i][k+1] /= nrm_phi
            else:
                nrm_phi = 1
        return nrm_phi

    def update_phi_y(self, y, k, mode, norm, return_norm):

        if mode == 'rl':
            nrms = []
            for i in range(self.len):
                phi = oe.contract(
                    'YAXB,amkA,xkX,bknB,ymnY->yaxb', self.phis_y[i][k+1], self.As[i].cores[k], self.xs[i].cores[k], self.Bs[i].cores[k], y)
                if return_norm:
                    nrm_phi = tn.linalg.norm(phi)
                    nrms.append(nrm_phi)

                if norm != 1:
                    phi = phi / norm
                self.phis_y[i][k] = phi.clone()
            
            if return_norm:
                nrm_phi = max(nrms)
                for i in range(self.len):
                    self.phis_y[i][k] /= nrm_phi
        else:
            nrms = []
            for i in range(self.len): 
                phi = oe.contract('yaxb,ymnY,amkA,xkX,bknB->YAXB',
                                self.phis_y[i][k], y, self.As[i].cores[k], self.xs[i].cores[k], self.Bs[i].cores[k])
                if return_norm:
                    nrm_phi = tn.linalg.norm(phi)
                    nrms.append(nrm_phi)

                if norm != 1:
                    phi = phi / norm
                self.phis_y[i][k+1] = phi.clone()
            if return_norm:
                nrm_phi = max(nrms)
                for i in range(self.len):
                    self.phis_y[i][k+1] /= nrm_phi
        return nrm_phi

class mm_multiple_local_op():

    def __init__(self, As, Bs, M, N, has_z, bandsA, bandsB):
        self.bandsA = bandsA 
        self.bandsB = bandsB
        self.As = As
        self.Bs = Bs
        self.M = M
        self.N = N
        self.has_z = has_z
        self.len = len(As)
        dt = As[0].cores[0].dtype
        dev = As[0].cores[0].device
        self.phis_y = [[tn.ones([1, 1, 1], device=dev, dtype=dt)] + [None]*(len(M)-1) + [tn.ones([1, 1, 1], device=dev, dtype=dt)] for i in range(self.len)]
        if has_z:
            self.phis_z = [[tn.ones([1, 1, 1], device=dev, dtype=dt)] + [None]*(len(M)-1) + [tn.ones([1, 1, 1], device=dev, dtype=dt)] for i in range(self.len)]


    def b_fun(self, k, first, second):
        # print()
        # print(k, first, second)
        # path_info = oe.contract_path('yax,YAX,amnA,xnX->ymY', self.phis_y[k] if first == 'y' else self.phis_z[k], self.phis_y[k+1] if second == 'y' else self.phis_z[k+1], self.A.cores[k], self.x.cores[k])
        # print(path_info[1])
        #ss = [oe.contract('yab,YAB,amkA,bknB->ymnY', self.phis_y[i][k] if first == 'y' else self.phis_z[i][k], self.phis_y[i][k+1] if second == 'y' else self.phis_z[i][k+1], self.As[i].cores[k], self.Bs[i].cores[k]) for i in range(self.len)]
        #return sum(ss)

        ss = [_local_AB(self.phis_y[i][k] if first == 'y' else self.phis_z[i][k], self.phis_y[i][k+1] if second == 'y' else self.phis_z[i][k+1], self.As[i].cores[k], self.Bs[i].cores[k], bandA=self.bandsA[i][k], bandB=self.bandsB[i][k]) for i in range(self.len)]
        return sum(ss)

    def update_phi_z(self, z, k, mode, norm, return_norm):

        if mode == 'rl':
            nrms = []
            for i in range(self.len):
                phi = _compute_phi_AB('bck', self.phis_z[i][k+1], self.As[i].cores[k], self.Bs[i].cores[k], z, self.bandsA[i][k], self.bandsB[i][k])
                # phi = oe.contract(
                #    'ZAB,amkA,bknB,zmnZ->zab', self.phis_z[i][k+1], self.As[i].cores[k], self.Bs[i].cores[k], z)
                if return_norm:
                    nrm_phi = tn.linalg.norm(phi)
                    nrms.append(nrm_phi)

                if norm != 1:
                    phi = phi / norm
                self.phis_z[i][k] = phi.clone()
            if return_norm:
                nrm_phi = max(nrms)
                for i in range(self.len):
                    self.phis_z[i][k] /= nrm_phi
            else:
                nrm_phi = 1
        else:
            nrms = []
            for i in range(self.len):
                phi = _compute_phi_AB('fwd', self.phis_z[i][k], self.As[i].cores[k], self.Bs[i].cores[k], z, self.bandsA[i][k], self.bandsB[i][k])
                #phi = oe.contract('zab,zmnZ,amkA,bknB->ZAB',
                #                self.phis_z[i][k], z, self.As[i].cores[k], self.Bs[i].cores[k])
                if return_norm:
                    nrm_phi = tn.linalg.norm(phi)
                    nrms.append(nrm_phi)
                if norm != 1:
                    phi = phi / norm
                self.phis_z[i][k+1] = phi.clone()
            
            if return_norm:
                nrm_phi = max(nrms)
                for i in range(self.len):
                    self.phis_z[i][k+1] /= nrm_phi
            else:
                nrm_phi = 1
        return nrm_phi

    def update_phi_y(self, y, k, mode, norm, return_norm):

        if mode == 'rl':
            nrms = []
            for i in range(self.len):
                phi = _compute_phi_AB('bck', self.phis_y[i][k+1], self.As[i].cores[k], self.Bs[i].cores[k], y, self.bandsA[i][k], self.bandsB[i][k])
                #phi = oe.contract(
                #    'YAB,amkA,bknB,ymnY->yab', self.phis_y[i][k+1], self.As[i].cores[k], self.Bs[i].cores[k], y)
                if return_norm:
                    nrm_phi = tn.linalg.norm(phi)
                    nrms.append(nrm_phi)

                if norm != 1:
                    phi = phi / norm
                self.phis_y[i][k] = phi.clone()
            
            if return_norm:
                nrm_phi = max(nrms)
                for i in range(self.len):
                    self.phis_y[i][k] /= nrm_phi
        else:
            nrms = []
            for i in range(self.len): 
                phi = _compute_phi_AB('fwd', self.phis_y[i][k], self.As[i].cores[k], self.Bs[i].cores[k], y, self.bandsA[i][k], self.bandsB[i][k])
                #phi = oe.contract('yab,ymnY,amkA,bknB->YAB',
                #                self.phis_y[i][k], y, self.As[i].cores[k], self.Bs[i].cores[k])
                if return_norm:
                    nrm_phi = tn.linalg.norm(phi)
                    nrms.append(nrm_phi)

                if norm != 1:
                    phi = phi / norm
                self.phis_y[i][k+1] = phi.clone()
            if return_norm:
                nrm_phi = max(nrms)
                for i in range(self.len):
                    self.phis_y[i][k+1] /= nrm_phi
        return nrm_phi
