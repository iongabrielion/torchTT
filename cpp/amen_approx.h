#include "define.h"
#include "ortho.h"
#include <cmath>
#include "amen_contractors.h"

namespace AMEn
{


    /**
     * Compute the interface for the core y from left to right.
     */
    at::Tensor compute_phiy_lr(at::Tensor &phi_prev, at::Tensor &z, at::Tensor &y)
    {
        if (z.sizes().size() == 3)
        {
            if (phi_prev.sizes()[0] > phi_prev.sizes()[1])
            {
                auto tmp = at::tensordot(phi_prev, z, {0}, {0});
                return at::tensordot(tmp, y, {0, 1}, {0, 1});
            }
            else
            {
                auto tmp = at::tensordot(phi_prev, y, {1}, {0});
                return at::tensordot(z, tmp, {0, 1}, {0, 1});
            }
        }
        else
        {
            if (phi_prev.sizes()[0] > phi_prev.sizes()[1])
            {
                auto tmp = at::tensordot(phi_prev, z, {0}, {0});
                return at::tensordot(tmp, y, {0, 1, 2}, {0, 1, 2});
            }
            else
            {
                auto tmp = at::tensordot(phi_prev, y, {1}, {0});
                return at::tensordot(z, tmp, {0, 1, 2}, {0, 1, 2});
            }
        }
    }

    /**
     * Compute the interface for the core y from right to left.
     */
    at::Tensor compute_phiy_rl(at::Tensor &phi_prev, at::Tensor &z, at::Tensor &y) 
    {
        if (z.sizes().size() == 3)
        {
            if (phi_prev.sizes()[0] < phi_prev.sizes()[1])
            {
                auto tmp = at::tensordot(z, phi_prev, {2}, {1});
                return at::tensordot(y, tmp, {1, 2}, {1, 2});
            }
            else
            {
                auto tmp = at::tensordot(y, phi_prev, {2}, {0});
                return at::tensordot(tmp, z, {0, 1}, {0, 1});
            }
        }
        else
        {
            if (phi_prev.sizes()[0] < phi_prev.sizes()[1])
            {
                auto tmp = at::tensordot(z, phi_prev, {3}, {1});
                return at::tensordot(y, tmp, {1, 2, 3}, {1, 2, 3});
            }
            else
            {
                auto tmp = at::tensordot(y, phi_prev, {3}, {0});
                return at::tensordot(tmp, z, {0, 1, 2}, {0, 1, 2});
            }
        }
    }

    std::vector<at::Tensor> amen_approx(AMEn::IContractor &contractor, double tol, std::vector<uint64_t> shape_out_M, std::vector<uint64_t> shape_out_N, std::vector<at::Tensor> &y, std::vector<at::Tensor> &z, int32_t nswp, int32_t kickrank, int32_t kickrank2, int32_t verb, bool init_qr, bool fkick, at::TensorOptions options) 
    {

        torch::NoGradGuard no_grad;
        options = options.requires_grad(false);

        int32_t d = shape_out_N.size();
        std::vector<uint64_t> &M = shape_out_M;
        std::vector<uint64_t> &N = shape_out_N;
        bool ttm = M.size() != 0;

        std::vector<at::Tensor> cores_y(d);
        std::vector<at::Tensor> cores_z(d);
        std::vector<int64_t> ry(d + 1);
        std::vector<int64_t> rz(d + 1);
        std::vector<at::Tensor> phizy(d);

        if (y.size() == 0)
        {
            ry[d] = 1;
            ry[0] = 1;
            for (int32_t i = 1; i < d; ++i)
                ry[i] = 2;

            for (int32_t i = 0; i < d; ++i)
            {
                if (ttm)
                    cores_y[i] = torch::randn({ry[i], M[i], N[i], ry[i + 1]}, options);
                else
                    cores_y[i] = torch::randn({ry[i], N[i], ry[i + 1]}, options);
            }
        }
        else
        {
            for (int32_t i = 0; i < d; ++i)
            {
                ry[i] = y[i].sizes()[0];
                cores_y[i] = y[i].clone();
            }
            ry[d] = 1;
        }

        if (kickrank + kickrank2 > 0)
        {
            if (z.size() == 0)
            {
                rz[d] = 1;
                rz[0] = 1;
                for (int32_t i = 1; i < d; ++i)
                    rz[i] = 2;

                for (int32_t i = 0; i < d; ++i)
                {
                    if (ttm)
                        cores_z[i] = torch::randn({rz[i], M[i], N[i], rz[i + 1]}, options);
                    else
                        cores_z[i] = torch::randn({rz[i], N[i], rz[i + 1]}, options);
                }
            }
            else
            {
                for (int32_t i = 0; i < d; ++i)
                {
                    rz[i] = z[i].sizes()[0];
                    cores_z[i] = z[i].clone();
                }
                rz[d] = 1;
            }
        }

        double *nrms = new double[d];
        for (int k = 0; k < d; ++k)
        {
            nrms[k] = 1.0;
        }

        for (int32_t i = 0; i < d; ++i)
        {
            if (init_qr)
            {
                auto cr = cores_y[i].reshape({-1, ry[i + 1]});
                std::tuple<at::Tensor, at::Tensor> QR = at::linalg_qr(cr);
                double nrmr = torch::norm(std::get<1>(QR)).item<double>();
                auto cr2 = at::tensordot(nrmr > 0 ? std::get<1>(QR) / nrmr : std::get<1>(QR), cores_y[i + 1], {1}, {0});
                ry[i + 1] = cr.sizes()[1];
                if (ttm)
                    cores_y[i] = std::get<0>(QR).reshape({ry[i], M[i], N[i], ry[i + 1]});
                else
                    cores_y[i] = std::get<0>(QR).reshape({ry[i], N[i], ry[i + 1]});
            }
            nrms[i] = contractor.update_phi_y(cores_y[i], i, "lr", 1, true);

            if (kickrank+kickrank2 > 0)
            {
                std::tuple<at::Tensor, at::Tensor> QR = at::linalg_qr(cores_z[i].reshape({-1, rz[i + 1]}));
                double nrmr = torch::norm(std::get<1>(QR)).item<double>();
                cores_z[i + 1] = at::tensordot(nrmr > 0 ? std::get<1>(QR) / nrmr : std::get<1>(QR), cores_z[i + 1], {1}, {0});
                rz[i + 1] = std::get<0>(QR).sizes()[1];
                if (ttm)
                    cores_z[i] = std::get<0>(QR).reshape({rz[i], M[i], N[i], rz[i + 1]});
                else
                    cores_z[i] = std::get<0>(QR).reshape({rz[i], N[i], rz[i + 1]});
                contractor.update_phi_z(cores_z[i], i, "lr", nrms[i], false);

                phizy[i + 1] = compute_phiy_lr(phizy[i], cores_z[i], cores_y[i]);
            }
        }

        int32_t i = d - 1;
        int direct = -1;
        uint32_t swp = 1;
        double max_dx;

        while (swp <= nswp)
        {
            at::Tensor cry = contractor.b_fun(i, 'y', 'y');
            nrms[i] = torch::norm(cry).item<double>();
            if (nrms[i] > 0)
                cry = cry / nrms[i];
            else
                nrms[i] = 1.0;

            double dx = torch::norm(cry - cores_y[i]).item<double>();
            max_dx = std::max(max_dx, dx);

            if ((direct > 0) && (i < d - 1))
            {
                std::tuple<at::Tensor, at::Tensor, at::Tensor> USV = at::linalg_svd(cry.reshape({-1, ry[i + 1]}), false);
                
                auto scpu = std::get<1>(USV).cpu();
                uint64_t r = rank_chop(scpu, tol * torch::norm(std::get<1>(USV)).item<double>() / std::sqrt((double)d));
                
                at::Tensor u = std::get<0>(USV).index({torch::indexing::Ellipsis, torch::indexing::Slice(0, r, 1)});
                at::Tensor v = at::conj(std::get<2>(USV).index({torch::indexing::Slice(0, r, 1), torch::indexing::Ellipsis})).t() * std::get<1>(USV).index({torch::indexing::Slice(0, r, 1)});

                if (kickrank + kickrank2 > 0)
                {
                    cry = at::tensordot(u, v, {1}, {1});
                    if (ttm)
                        cry = cry.reshape({ry[i], M[i], N[i], ry[i + 1]});
                    else
                        cry = cry.reshape({ry[i], N[i], ry[i + 1]});
                    auto crz = contractor.b_fun(i, 'z', 'z');
                }
            }
        }

        delete[] nrms;
        return cores_y;
    }
}