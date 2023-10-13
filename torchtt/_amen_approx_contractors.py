import torch as tn 
import opt_einsum as oe 

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

        self.phis_y = [[tn.ones([1, 1, 1, 1], device=A.cores[0].device, dtype=A.cores[0].dtype)] + [None]*(len(M)-1) + [tn.ones([1, 1, 1, 1], device=A.cores[0].device, dtype=A.cores[0].dtype)] for i in range(self.len)]
        if has_z:
            self.phis_z = [[tn.ones([1, 1, 1, 1], device=A.cores[0].device, dtype=A.cores[0].dtype)] + [None]*(len(M)-1) + [tn.ones([1, 1, 1, 1], device=A.cores[0].device, dtype=A.cores[0].dtype)] for i in range(self.len)]


    def b_fun(self, k, first, second):

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
                phi = oe.contract('yaxb,ymnY,amkA,xkX,cknC->YAXB',
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
