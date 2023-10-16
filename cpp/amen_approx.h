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
                auto tmp = at::tensordot(phi_prev, z, {0}, {0}); // ynZ
                return at::tensordot(tmp, y, {0, 1}, {0, 1}); // ZY
            }
            else
            {
                auto tmp = at::tensordot(phi_prev, y, {1}, {0}); // znY
                return at::tensordot(z, tmp, {0, 1}, {0, 1}); // ZY
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
                auto tmp = at::tensordot(z, phi_prev, {2}, {1}); // znZ, YZ ->znY
                return at::tensordot(y, tmp, {1, 2}, {1, 2}); // ynY, znY -> yz
            }
            else
            {
                auto tmp = at::tensordot(y, phi_prev, {2}, {0}); // ynY, YZ -> ynZ
                return at::tensordot(tmp, z, {1, 2}, {1, 2}); // ynZ, znZ -> yz
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
                return at::tensordot(tmp, z, {1, 2, 3}, {1, 2, 3});
            }
        }
    }

    std::vector<at::Tensor> amen_approx(AMEn::IContractor &contractor, double tol, std::vector<uint64_t> shape_out_M, std::vector<uint64_t> shape_out_N, std::vector<at::Tensor> &y, std::vector<at::Tensor> &z, int32_t nswp, int32_t kickrank, int32_t kickrank2, int32_t verb, bool init_qr, bool fkick, at::TensorOptions options)
    {

        torch::NoGradGuard no_grad;
        options = options.requires_grad(false);

        std::chrono::time_point<std::chrono::high_resolution_clock> tme_swp, tme_total;
        if (verb > 0)
            tme_total = std::chrono::high_resolution_clock::now();

        int32_t d = shape_out_N.size();
        std::vector<uint64_t> &M = shape_out_M;
        std::vector<uint64_t> &N = shape_out_N;
        bool ttm = M.size() != 0;

        std::vector<at::Tensor> cores_y(d);
        std::vector<at::Tensor> cores_z(d);
        std::vector<int64_t> ry(d + 1);
        std::vector<int64_t> rz(d + 1);
        std::vector<at::Tensor> phizy(d+1);

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
                    rz[i] = kickrank+kickrank2;

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
            phizy[0] = torch::ones({1,1}, options);
            phizy[d] = torch::ones({1,1}, options);
        }

        double *nrms = new double[d];
        for (int k = 0; k < d; ++k)
        {
            nrms[k] = 1.0;
        }

        if (verb > 0)
            std::cout << "Initial orhtogonalization" << std::endl;
        for (int32_t i = 0; i < d-1; ++i)
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
            if (kickrank + kickrank2 > 0)
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
        at::Tensor cry;

        while (swp <= nswp)
        {
            if (verb > 0)
            {
                if (((direct < 0) && (i == d - 1)) || ((direct > 0) && (i == 0)))
                {
                    std::cout << "Sweep " << swp << ", direction " << direct << std::endl << std::flush;
                    tme_swp = std::chrono::high_resolution_clock::now();
                }
            }

            cry = contractor.b_fun(i, 'y', 'y');
            nrms[i] = torch::norm(cry).item<double>();
            if (nrms[i] > 0)
                cry = cry / nrms[i];
            else
                nrms[i] = 1.0;

            double dx = torch::norm(cry - cores_y[i]).item<double>();
            max_dx = std::max(max_dx, dx);
            uint64_t r;

            if ((direct > 0) && (i < d - 1))
            {
                std::tuple<at::Tensor, at::Tensor, at::Tensor> USV = at::linalg_svd(cry.reshape({-1, ry[i + 1]}), false);

                auto scpu = std::get<1>(USV).cpu();
                r = rank_chop(scpu, tol * torch::norm(std::get<1>(USV)).item<double>() / std::sqrt((double)d));

                at::Tensor u = std::get<0>(USV).index({torch::indexing::Ellipsis, torch::indexing::Slice(0, r, 1)});
                at::Tensor v = at::conj(std::get<2>(USV).index({torch::indexing::Slice(0, r, 1), torch::indexing::Ellipsis})).t() * std::get<1>(USV).index({torch::indexing::Slice(0, r, 1)});
                at::Tensor crz;

                if (kickrank + kickrank2 > 0)
                {
                    cry = at::tensordot(u, v, {1}, {1});
                    if (ttm)
                        cry = cry.reshape({ry[i], M[i], N[i], ry[i + 1]});
                    else
                        cry = cry.reshape({ry[i], N[i], ry[i + 1]});
                    crz = contractor.b_fun(i, 'z', 'z');
                    auto ys = at::tensordot(cry, phizy[i + 1], {ttm ? 3 : 2}, {0});
                    auto yz = at::tensordot(phizy[i], ys, {1}, {0});
                    crz = crz / nrms[i] - yz;
                    double nrmz = torch::norm(crz).item<double>();
                    crz = crz.reshape({-1, rz[i + 1]});
                    if (kickrank2 > 0)
                    {
                        std::tuple<at::Tensor, at::Tensor, at::Tensor> USV = at::linalg_svd(crz, true);
                        crz = std::get<0>(USV).index({torch::indexing::Ellipsis, torch::indexing::Slice(0, std::min((int32_t)std::get<0>(USV).sizes()[1], kickrank), 1)});
                        crz = at::cat({crz, torch::randn({crz.sizes()[0], kickrank2}, options)}, 1);
                    }
                    if (fkick)
                    {
                        auto crs = contractor.b_fun(i, 'y', 'z');

                        crs = crs / nrms[i] - ys;
                        std::tuple<at::Tensor, at::Tensor> QR = at::linalg_qr(at::cat({u, crs.reshape({-1, rz[i + 1]})}, 1));
                        u = std::get<0>(QR);
                        v = at::cat({v, torch::zeros({ry[i + 1], rz[i + 1]}, options)}, 1);
                        v = at::tensordot(v, std::get<1>(QR), {1}, {1});
                        r = u.sizes()[1];
                    }
                }

                if (ttm)
                    cores_y[i] = u.reshape({ry[i], M[i], N[i], r});
                else
                    cores_y[i] = u.reshape({ry[i], N[i], r});

                cores_y[i + 1] = at::tensordot(v.reshape({ry[i + 1], r}), cores_y[i + 1], {0}, {0});

                ry[i + 1] = r;

                nrms[i] = contractor.update_phi_y(cores_y[i], i, "lr", 1, true);

                if (kickrank + kickrank2 > 0)
                {
                    at::Tensor crz2;
                    std::tie(crz2, std::ignore) = at::linalg_qr(crz);
                    rz[i+1] = crz2.sizes()[1];
                    if (ttm)
                        cores_z[i] = crz2.reshape({rz[i], M[i], N[i], rz[i + 1]});
                    else
                        cores_z[i] = crz2.reshape({rz[i], N[i], rz[i + 1]});
                    contractor.update_phi_z(cores_z[i], i, "lr", nrms[i], false);

                    phizy[i + 1] = compute_phiy_lr(phizy[i], cores_z[i], cores_y[i]);
                }
            }
            else if ((direct < 0) && (i > 0))
            {
                std::tuple<at::Tensor, at::Tensor, at::Tensor> USV = at::linalg_svd(cry.reshape({ry[i], -1}), false);

                auto scpu = std::get<1>(USV).cpu();
                r = rank_chop(scpu, tol * torch::norm(std::get<1>(USV)).item<double>() / std::sqrt((double)d));

                at::Tensor v = at::conj(std::get<2>(USV).index({torch::indexing::Slice(0, r, 1), torch::indexing::Ellipsis}).t());
                at::Tensor u = std::get<0>(USV).index({torch::indexing::Ellipsis, torch::indexing::Slice(0, r, 1)}) * std::get<1>(USV).index({torch::indexing::Slice(0, r, 1)});
                at::Tensor crz;

                
                if (kickrank + kickrank2 > 0)
                {
                    at::Tensor cry = at::tensordot(u, v, {1}, {1}).reshape({ry[i], -1});
                    crz = contractor.b_fun(i, 'z', 'z').reshape({rz[i], -1});

                    auto ys = at::linalg_matmul(phizy[i], cry);
                    auto yz = ys.reshape({-1, ry[i + 1]});
                    yz = at::linalg_matmul(yz, phizy[i + 1]).reshape({rz[i], -1});

                    crz = crz / nrms[i] - yz;

                    double nrmz = torch::norm(crz).item<double>();

                    
                    if (kickrank2 > 0)
                    {
                        std::tie(std::ignore, std::ignore, crz) = at::linalg_svd(crz, false);
                        crz = crz.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, std::min(kickrank, (int32_t)crz.sizes()[1]), 1)});
                        crz = at::conj(crz.t());
                        crz = at::cat({crz, torch::randn({kickrank2, crz.sizes()[1]}, options)}, 0);
                    }

                    
                    auto crs = contractor.b_fun(i, 'z', 'y').reshape({rz[i], -1});
                    crs = crs / nrms[i] - ys;

                    v = at::cat({v, crs.t()}, 1);

                    std::tuple<at::Tensor, at::Tensor> QR = at::linalg_qr(v);

                    v = std::get<0>(QR);

                    u = at::cat({u, torch::zeros({ry[i], rz[i]}, options)}, 1);
                    u = at::tensordot(u, std::get<1>(QR), {1}, {1});
                    r = v.sizes()[1];
                }
                if (ttm)
                {
                    
                    cores_y[i - 1] = at::tensordot(cores_y[i - 1], u, {3}, {0}).reshape({ry[i - 1], M[i - 1], N[i - 1], r});
                    cores_y[i] = v.t().reshape({r, M[i], N[i], ry[i + 1]});
                }
                else
                {
                    cores_y[i - 1] = at::tensordot(cores_y[i - 1], u, {2}, {0}).reshape({ry[i - 1], N[i - 1], r});
                    cores_y[i] = v.t().reshape({r, N[i], ry[i + 1]});
                }
                ry[i] = r;
                
                nrms[i] = contractor.update_phi_y(cores_y[i], i, "rl", 1, true);

                if (kickrank + kickrank2 > 0)
                {
                    std::tuple<at::Tensor, at::Tensor> QR = at::linalg_qr(crz.t());

                    rz[i] = std::get<0>(QR).sizes()[1];
                    if (ttm)
                        cores_z[i] = std::get<0>(QR).t().reshape({rz[i], M[i], N[i], rz[i + 1]});
                    else
                        cores_z[i] = std::get<0>(QR).t().reshape({rz[i], N[i], rz[i + 1]});
                    contractor.update_phi_z(cores_z[i], i, "rl", nrms[i], false);
                    phizy[i] = compute_phiy_rl(phizy[i + 1], cores_z[i], cores_y[i]);
                }
            }

            if (verb)
                std::cout << "\t\tcore " << i << ", dx " << dx << ", rank " << r << std::endl;

            if (((direct > 0) && (i == d - 1)) || ((direct < 0) && (i == 0)))
            {
                if (verb > 0)
                {
                    auto diff_time = std::chrono::high_resolution_clock::now() - tme_swp;
                    auto max_ry = *std::max_element(ry.begin(), ry.end());
                    std::cout << "\tfinished after " << (double)(std::chrono::duration_cast<std::chrono::microseconds>(diff_time)).count() / 1000.0 << " ms, max dx " << max_dx << ", max rank " << max_ry << std::endl;
                }
                if (((max_dx < tol) || (swp == nswp)) && (direct > 0))
                    break;
                else
                {
                    if (ttm)
                        cores_y[i] = cry.reshape({ry[i], M[i], N[i], ry[i + 1]});
                    else
                        cores_y[i] = cry.reshape({ry[i], N[i], ry[i + 1]});
                    if (direct > 0)
                        swp++;
                }
                max_dx = 0;
                direct = -direct;
            }
            else
            {
                i += direct;
            }
        }

        cores_y[d - 1] = ttm ? cry.reshape({ry[d - 1], M[d - 1], N[d - 1], ry[d]}) : cry.reshape({ry[d - 1], N[d - 1], ry[d]});

        double nrms_dist = 0;
        for (int32_t i = 0; i < d; ++i)
            nrms_dist += std::log(nrms[i]);
        nrms_dist = std::exp(nrms_dist / d);

        for (int32_t i = 0; i < d; ++i)
            cores_y[i] *= nrms_dist;

        if (verb > 0)
        {

            auto diff_time = std::chrono::high_resolution_clock::now() - tme_swp;
            std::cout << "Finished in " << (double)(std::chrono::duration_cast<std::chrono::microseconds>(diff_time)).count() / 1000000.0 << " s" << std::endl;
        }

        delete[] nrms;
        return cores_y;
    }

    std::vector<at::Tensor> amen_mv(std::vector<at::Tensor> &A_cores, std::vector<at::Tensor> &x_cores, std::vector<uint64_t> M, std::vector<uint64_t> N, std::vector<at::Tensor> &y0, double eps, int32_t nswp, int32_t kickrank, int32_t kickrank2, int32_t verb, bool init_qr, bool fkick)
    {
        auto options = A_cores[0].options();
        ContractorMv contr(A_cores, x_cores, M, N, kickrank + kickrank2 > 0);
        std::vector<at::Tensor> z;
        std::vector<uint64_t> empty;
        return AMEn::amen_approx(contr, eps, empty, M, y0, z, nswp, kickrank, kickrank2, verb, init_qr, fkick, options);
    }
}