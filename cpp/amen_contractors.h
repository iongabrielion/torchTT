
#include "define.h"

namespace AMEn
{

    class IContractor
    {
    public:
        virtual at::Tensor b_fun(uint32_t k, char first, char second) = 0;
        virtual double update_phi_z(at::Tensor &z, uint32_t k, const char *mode, double norm, bool return_norm) = 0;
        virtual double update_phi_y(at::Tensor &y, uint32_t k, const char *mode, double norm, bool return_norm) = 0;
    };

    class ContractorMv : public IContractor
    {

    private:
        std::vector<at::Tensor> A_cores;
        std::vector<at::Tensor> x_cores;
        std::vector<uint64_t> M;
        std::vector<uint64_t> N;
        bool has_z;
        std::vector<at::Tensor> phis_y;
        std::vector<at::Tensor> phis_z;

    public:
        ContractorMv(std::vector<at::Tensor> &A_cores, std::vector<at::Tensor> &x_cores, std::vector<uint64_t> &M, std::vector<uint64_t> &N, bool has_z)
        {
            this->has_z = has_z;
            this->A_cores = A_cores;
            this->x_cores = x_cores;
            this->M = M;
            this->N = N;
            auto d = M.size();

            auto options = A_cores[0].options();

            this->phis_y = std::vector<at::Tensor>(d + 1);
            this->phis_z = std::vector<at::Tensor>(d + 1);

            this->phis_y[0] = torch::ones({1, 1, 1}, options);
            this->phis_z[0] = torch::ones({1, 1, 1}, options);
            this->phis_y[d] = torch::ones({1, 1, 1}, options);
            this->phis_z[d] = torch::ones({1, 1, 1}, options);
        }

        virtual at::Tensor b_fun(uint32_t k, char first, char second)
        {
            auto tmp = at::tensordot(this->x_cores[k], second == 'y' ? this->phis_y[k + 1] : this->phis_z[k + 1], {2}, {2}); // xnX,YAX->xnYA
            tmp = at::tensordot(tmp, this->A_cores[k], {1, 3}, {2, 3});                                                      // xnYA,amnA->xYam
            tmp = at::tensordot(first == 'y' ? this->phis_y[k] : this->phis_z[k], tmp, {1, 2}, {2, 0});                      // yax,xYam->ymY
            return tmp.permute({0, 2, 1});
        }

        virtual double update_phi_z(at::Tensor &z, uint32_t k, const char *mode, double norm, bool return_norm)
        {
            double nrm_phi = 1;

            if (mode[0] == 'r' && mode[1] == 'l')
            {
                //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                //  --------------------------------------------------------------------------------
                //  scaling        BLAS                current                             remaining
                //  --------------------------------------------------------------------------------
                //  5           GEMM          xnX,ZAX->xnZA                    amnA,zmZ,xnZA->zax
                //  6           TDOT        xnZA,amnA->xZam                         zmZ,xZam->zax
                //  5           TDOT          xZam,zmZ->zax                              zax->zax
                at::Tensor phi;

                if (this->A_cores[k].sizes()[1] * 2 < this->x_cores[k].sizes()[0] * this->x_cores[k].sizes()[2] * this->A_cores[k].sizes()[3])
                {
                    auto tmp = at::tensordot(this->x_cores[k], this->phis_z[k + 1], {2}, {2});
                    tmp = at::tensordot(tmp, this->A_cores[k], {1, 3}, {2, 3});
                    phi = at::tensordot(z, tmp, {1, 2}, {3, 1}).permute({0, 2, 1});
                }
                else
                {
                    auto tmp = at::tensordot(z, this->A_cores[k], {1}, {1});
                    tmp = at::tensordot(tmp, this->x_cores[k], {3}, {1});
                    phi = at::tensordot(tmp, this->phis_z[k + 1], {1, 3, 5}, {0, 1, 2});
                }
                // Alternative
                //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                //  --------------------------------------------------------------------------------
                //  scaling        BLAS                current                             remaining
                //  --------------------------------------------------------------------------------
                //  6           TDOT        zmZ,amnA->zZanA                    ZAX,xnX,zZanA->zax
                //  7           TDOT      zZanA,xnX->zZaAxX                       ZAX,zZaAxX->zax
                //  6    GEMV/EINSUM        zZaAxX,ZAX->zax                              zax->zax

                if (return_norm)
                {
                    nrm_phi = torch::norm(phi).item<double>();
                    norm *= nrm_phi;
                }
                if (norm != 1.0)
                    phi = phi / norm;
                this->phis_z[k] = phi.clone();
            }
            else
            {
                //   Complete contraction:  zax,zmZ,amnA,xnX->ZAX
                // --------------------------------------------------------------------------------
                // scaling        BLAS                current                             remaining
                // --------------------------------------------------------------------------------
                // 5           GEMM          xnX,zax->nXza                    zmZ,amnA,nXza->ZAX
                // 6           TDOT        nXza,amnA->XzmA                         zmZ,XzmA->ZAX
                // 5           TDOT          XzmA,zmZ->ZAX                              ZAX->ZAX

                at::Tensor phi;

                auto tmp = at::tensordot(this->x_cores[k], this->phis_z[k], {0}, {2});
                tmp = at::tensordot(tmp, this->A_cores[k], {0, 3}, {2, 0});
                phi = at::tensordot(z, tmp, {0, 1}, {1, 2}).permute({0, 2, 1}); // zmZ, XzmA ->ZXA and permute

                if (return_norm)
                {
                    nrm_phi = torch::norm(phi).item<double>();
                    norm *= nrm_phi;
                }
                if (norm != 1.0)
                    phi = phi / norm;
                this->phis_z[k + 1] = phi.clone();
            }

            return nrm_phi;
        }

        virtual double update_phi_y(at::Tensor &y, uint32_t k, const char *mode, double norm, bool return_norm)
        {

            double nrm_phi = 1;

            if (mode[0] == 'r' && mode[1] == 'l')
            {
                at::Tensor phi;
                if (this->A_cores[k].sizes()[1] * 2 < this->x_cores[k].sizes()[0] * this->x_cores[k].sizes()[2] * this->A_cores[k].sizes()[3])
                {
                    auto tmp = at::tensordot(this->x_cores[k], this->phis_y[k + 1], {2}, {2});
                    tmp = at::tensordot(tmp, this->A_cores[k], {1, 3}, {2, 3});
                    phi = at::tensordot(y, tmp, {1, 2}, {3, 1}).permute({0, 2, 1});
                }
                else
                {
                    auto tmp = at::tensordot(y, this->A_cores[k], {1}, {1});
                    tmp = at::tensordot(tmp, this->x_cores[k], {3}, {1});
                    phi = at::tensordot(tmp, this->phis_y[k + 1], {1, 3, 5}, {0, 1, 2});
                }

                if (return_norm)
                {
                    nrm_phi = torch::norm(phi).item<double>();
                    norm *= nrm_phi;
                }
                if (norm != 1.0)
                    phi = phi / norm;
                this->phis_y[k] = phi.clone();
            }
            else
            {

                at::Tensor phi;

                auto tmp = at::tensordot(this->x_cores[k], this->phis_y[k], {0}, {2});
                tmp = at::tensordot(tmp, this->A_cores[k], {0, 3}, {2, 0});
                phi = at::tensordot(y, tmp, {0, 1}, {1, 2}).permute({0, 2, 1});

                if (return_norm)
                {
                    nrm_phi = torch::norm(phi).item<double>();
                    norm *= nrm_phi;
                }
                if (norm != 1.0)
                    phi = phi / norm;
                this->phis_y[k + 1] = phi.clone();
            }

            return nrm_phi;
        }
    };

    class ContractorMvMultiple : public IContractor
    {

    private:
        std::vector<at::Tensor> A_cores;
        std::vector<std::vector<at::Tensor>> x_cores;
        std::vector<uint64_t> M;
        std::vector<uint64_t> N;
        int len;
        bool has_z;
        std::vector<std::vector<at::Tensor>> phis_y;
        std::vector<std::vector<at::Tensor>> phis_z;

    public:
        ContractorMvMultiple(std::vector<at::Tensor> &A_cores, std::vector<std::vector<at::Tensor>> &x_cores, std::vector<uint64_t> &M, std::vector<uint64_t> &N, bool has_z)
        {
            this->has_z = has_z;
            this->A_cores = A_cores;
            this->x_cores = x_cores;
            this->M = M;
            this->N = N;
            this->len = x_cores.size();
            auto d = M.size();

            auto options = A_cores[0].options();

            this->phis_y = std::vector<std::vector<at::Tensor>>(this->len);
            this->phis_z = std::vector<std::vector<at::Tensor>>(this->len);
            for (int k = 0; k < this->len; ++k)
            {
                this->phis_y[k] = std::vector<at::Tensor>(d + 1);
                this->phis_z[k] = std::vector<at::Tensor>(d + 1);

                this->phis_y[k][0] = torch::ones({1, 1, 1}, options);
                this->phis_z[k][0] = torch::ones({1, 1, 1}, options);
                this->phis_y[k][d] = torch::ones({1, 1, 1}, options);
                this->phis_z[k][d] = torch::ones({1, 1, 1}, options);
            }
        }

        virtual at::Tensor b_fun(uint32_t k, char first, char second)
        {
            at::Tensor result = torch::zeros({(first == 'y' ? this->phis_y[0][k].sizes()[0] : this->phis_z[0][k].sizes()[0]),  (second == 'y' ? this->phis_y[0][k + 1].sizes()[0] : this->phis_z[0][k + 1].sizes()[0]), this->A_cores[k].sizes()[1]}, this->A_cores[0].options());
            for (int i = 0; i < this->len; ++i)
            {
                auto tmp = at::tensordot(this->x_cores[i][k], second == 'y' ? this->phis_y[i][k + 1] : this->phis_z[i][k + 1], {2}, {2}); // xnX,YAX->xnYA
                tmp = at::tensordot(tmp, this->A_cores[k], {1, 3}, {2, 3});                                                               // xnYA,amnA->xYam
                tmp = at::tensordot(first == 'y' ? this->phis_y[i][k] : this->phis_z[i][k], tmp, {1, 2}, {2, 0});                         // yax,xYam->ymY
                result += tmp;
            }
            return result.permute({0, 2, 1});
        }

        virtual double update_phi_z(at::Tensor &z, uint32_t k, const char *mode, double norm, bool return_norm)
        {
            double nrm_phi = 1;

            if (mode[0] == 'r' && mode[1] == 'l')
            {
                std::vector<double> nrms(this->len);

                for (int i = 0; i < this->len; ++i)
                {
                    //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                    //  --------------------------------------------------------------------------------
                    //  scaling        BLAS                current                             remaining
                    //  --------------------------------------------------------------------------------
                    //  5           GEMM          xnX,ZAX->xnZA                    amnA,zmZ,xnZA->zax
                    //  6           TDOT        xnZA,amnA->xZam                         zmZ,xZam->zax
                    //  5           TDOT          xZam,zmZ->zax                              zax->zax
                    at::Tensor phi;

                    if (this->A_cores[k].sizes()[1] * 2 < this->x_cores[i][k].sizes()[0] * this->x_cores[i][k].sizes()[2] * this->A_cores[k].sizes()[3])
                    {
                        auto tmp = at::tensordot(this->x_cores[i][k], this->phis_z[i][k + 1], {2}, {2});
                        tmp = at::tensordot(tmp, this->A_cores[k], {1, 3}, {2, 3});
                        phi = at::tensordot(z, tmp, {1, 2}, {3, 1}).permute({0, 2, 1});
                    }
                    else
                    {
                        auto tmp = at::tensordot(z, this->A_cores[k], {1}, {1});
                        tmp = at::tensordot(tmp, this->x_cores[i][k], {3}, {1});
                        phi = at::tensordot(tmp, this->phis_z[i][k + 1], {1, 3, 5}, {0, 1, 2});
                    }
                    // Alternative
                    //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                    //  --------------------------------------------------------------------------------
                    //  scaling        BLAS                current                             remaining
                    //  --------------------------------------------------------------------------------
                    //  6           TDOT        zmZ,amnA->zZanA                    ZAX,xnX,zZanA->zax
                    //  7           TDOT      zZanA,xnX->zZaAxX                       ZAX,zZaAxX->zax
                    //  6    GEMV/EINSUM        zZaAxX,ZAX->zax                              zax->zax

                    if (return_norm)
                    {
                        nrms[i] = torch::norm(phi).item<double>();
                    }
                    if (norm != 1.0)
                        phi = phi / norm;
                    this->phis_z[i][k] = phi.clone();
                }
                if (return_norm)
                {
                    nrm_phi = *std::max_element(std::begin(nrms), std::end(nrms));
                    for (int i = 0; i < this->len; ++i)
                        this->phis_z[i][k] /= nrm_phi;
                }
                else
                    nrm_phi = 1;
            }
            else
            {
                std::vector<double> nrms(this->len);

                for (int i = 0; i < this->len; ++i)
                {

                    //   Complete contraction:  zax,zmZ,amnA,xnX->ZAX
                    // --------------------------------------------------------------------------------
                    // scaling        BLAS                current                             remaining
                    // --------------------------------------------------------------------------------
                    // 5           GEMM          xnX,zax->nXza                    zmZ,amnA,nXza->ZAX
                    // 6           TDOT        nXza,amnA->XzmA                         zmZ,XzmA->ZAX
                    // 5           TDOT          XzmA,zmZ->ZAX                              ZAX->ZAX

                    at::Tensor phi;

                    auto tmp = at::tensordot(this->x_cores[i][k], this->phis_z[i][k], {0}, {2});
                    tmp = at::tensordot(tmp, this->A_cores[k], {0, 3}, {2, 0});
                    phi = at::tensordot(z, tmp, {0, 1}, {1, 2}).permute({0, 2, 1}); // zmZ, XzmA ->ZXA and permute

                    if (return_norm)
                    {
                        nrms[i] = torch::norm(phi).item<double>();
                    }
                    if (norm != 1.0)
                        phi = phi / norm;
                    this->phis_z[i][k + 1] = phi.clone();
                }
                if (return_norm)
                {
                    nrm_phi = *std::max_element(std::begin(nrms), std::end(nrms));
                    for (int i = 0; i < this->len; ++i)
                        this->phis_z[i][k + 1] /= nrm_phi;
                }
                else
                    nrm_phi = 1;
            }

            return nrm_phi;
        }

        virtual double update_phi_y(at::Tensor &y, uint32_t k, const char *mode, double norm, bool return_norm)
        {

            double nrm_phi = 1;

            if (mode[0] == 'r' && mode[1] == 'l')
            {
                std::vector<double> nrms(this->len);

                for (int i = 0; i < this->len; ++i)
                {
                    //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                    //  --------------------------------------------------------------------------------
                    //  scaling        BLAS                current                             remaining
                    //  --------------------------------------------------------------------------------
                    //  5           GEMM          xnX,ZAX->xnZA                    amnA,zmZ,xnZA->zax
                    //  6           TDOT        xnZA,amnA->xZam                         zmZ,xZam->zax
                    //  5           TDOT          xZam,zmZ->zax                              zax->zax
                    at::Tensor phi;

                    if (this->A_cores[k].sizes()[1] * 2 < this->x_cores[i][k].sizes()[0] * this->x_cores[i][k].sizes()[2] * this->A_cores[k].sizes()[3])
                    {
                        auto tmp = at::tensordot(this->x_cores[i][k], this->phis_y[i][k + 1], {2}, {2});
                        tmp = at::tensordot(tmp, this->A_cores[k], {1, 3}, {2, 3});
                        phi = at::tensordot(y, tmp, {1, 2}, {3, 1}).permute({0, 2, 1});
                    }
                    else
                    {
                        auto tmp = at::tensordot(y, this->A_cores[k], {1}, {1});
                        tmp = at::tensordot(tmp, this->x_cores[i][k], {3}, {1});
                        phi = at::tensordot(tmp, this->phis_y[i][k + 1], {1, 3, 5}, {0, 1, 2});
                    }
                    // Alternative
                    //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                    //  --------------------------------------------------------------------------------
                    //  scaling        BLAS                current                             remaining
                    //  --------------------------------------------------------------------------------
                    //  6           TDOT        zmZ,amnA->zZanA                    ZAX,xnX,zZanA->zax
                    //  7           TDOT      zZanA,xnX->zZaAxX                       ZAX,zZaAxX->zax
                    //  6    GEMV/EINSUM        zZaAxX,ZAX->zax                              zax->zax

                    if (return_norm)
                    {
                        nrms[i] = torch::norm(phi).item<double>();
                    }
                    if (norm != 1.0)
                        phi = phi / norm;
                    this->phis_y[i][k] = phi.clone();
                }
                if (return_norm)
                {
                    nrm_phi = *std::max_element(std::begin(nrms), std::end(nrms));
                    for (int i = 0; i < this->len; ++i)
                        this->phis_y[i][k] /= nrm_phi;
                }
                else
                    nrm_phi = 1;
            }
            else
            {

                std::vector<double> nrms(this->len);

                for (int i = 0; i < this->len; ++i)
                {

                    at::Tensor phi;

                    auto tmp = at::tensordot(this->x_cores[i][k], this->phis_y[i][k], {0}, {2});
                    tmp = at::tensordot(tmp, this->A_cores[k], {0, 3}, {2, 0});
                    phi = at::tensordot(y, tmp, {0, 1}, {1, 2}).permute({0, 2, 1}); // zmZ, XzmA ->ZXA and permute

                    if (return_norm)
                    {
                        nrms[i] = torch::norm(phi).item<double>();
                    }
                    if (norm != 1.0)
                        phi = phi / norm;
                    this->phis_y[i][k + 1] = phi.clone();
                }
                if (return_norm)
                {
                    nrm_phi = *std::max_element(std::begin(nrms), std::end(nrms));
                    for (int i = 0; i < this->len; ++i)
                        this->phis_y[i][k + 1] /= nrm_phi;
                }
                else
                    nrm_phi = 1;
            }

            return nrm_phi;
        }
    };
 /* 
    class ContractorMvMMultiple : public IContractor
    {

    private:
        std::vector<std::vector<at::Tensor>> A_cores;
        std::vector<std::vector<at::Tensor>> x_cores;
        std::vector<std::vector<at::Tensor>> B_cores;
        std::vector<uint64_t> M;
        std::vector<uint64_t> N;
        int len;
        bool has_z;
        std::vector<std::vector<at::Tensor>> phis_y;
        std::vector<std::vector<at::Tensor>> phis_z;

    public:
        ContractorMvMultiple( std::vector<std::vector<at::Tensor>> &A_cores, std::vector<std::vector<at::Tensor>> &x_cores,  std::vector<std::vector<at::Tensor>> &A_cores, std::vector<uint64_t> &M, std::vector<uint64_t> &N, bool has_z)
        {
            this->has_z = has_z;
            this->A_cores = A_cores;
            this->x_cores = x_cores;
            this->B_cores = B_cores;
            this->M = M;
            this->N = N;
            this->len = x_cores.size();
            auto d = M.size();

            auto options = A_cores[0].options();

            this->phis_y = std::vector<std::vector<at::Tensor>>(this->len);
            this->phis_z = std::vector<std::vector<at::Tensor>>(this->len);
            for (int k = 0; k < this->len; ++k)
            {
                this->phis_y[k] = std::vector<at::Tensor>(d + 1);
                this->phis_z[k] = std::vector<at::Tensor>(d + 1);

                this->phis_y[k][0] = torch::ones({1, 1, 1, 1}, options);
                this->phis_z[k][0] = torch::ones({1, 1, 1, 1}, options);
                this->phis_y[k][d] = torch::ones({1, 1, 1, 1}, options);
                this->phis_z[k][d] = torch::ones({1, 1, 1, 1}, options);
            }
        }

        virtual at::Tensor b_fun(uint32_t k, char first, char second)
        {
            at::Tensor result = torch::zeros({(first == 'y' ? this->phis_y[0][k].sizes()[0] : this->phis_z[0][k].sizes()[0]),  (second == 'y' ? this->phis_y[0][k + 1].sizes()[0] : this->phis_z[0][k + 1].sizes()[0]), this->A_cores[k].sizes()[1]}, this->A_cores[0].options());
            for (int i = 0; i < this->len; ++i)
            {
                //  Complete contraction:  yabc,YABC,amkA,bkB,cknC->ymnY
                //         Naive scaling:  11
                //     Optimized scaling:  8
                //      Naive FLOP count:  4.493e+6
                //  Optimized FLOP count:  5.933e+4
                //   Theoretical speedup:  7.573e+1
                //  Largest intermediate:  2.340e+3 elements
                //--------------------------------------------------------------------------------
                //scaling        BLAS                current                             remaining
                //--------------------------------------------------------------------------------
                //   6           TDOT        bkB,yabc->kByac            YABC,amkA,cknC,kByac->ymnY
                //   7              0     kByac,cknC->kByanC                YABC,amkA,kByanC->ymnY
                //   8           TDOT    kByanC,amkA->BynCmA                     YABC,BynCmA->ymnY
                //   7           TDOT      BynCmA,YABC->ymnY                            ymnY->ymnY
                auto tmp = at::tensordot(this->x_cores[i][k], second == 'y' ? this->phis_y[i][k + 1] : this->phis_z[i][k + 1], {2}, {2}); // xnX,YAX->xnYA
                tmp = at::tensordot(tmp, this->A_cores[k], {1, 3}, {2, 3});                                                               // xnYA,amnA->xYam
                tmp = at::tensordot(first == 'y' ? this->phis_y[i][k] : this->phis_z[i][k], tmp, {1, 2}, {2, 0});                         // yax,xYam->ymY
                result += tmp;
            }
            return result.permute({0, 2, 1});
        }

        virtual double update_phi_z(at::Tensor &z, uint32_t k, const char *mode, double norm, bool return_norm)
        {
            double nrm_phi = 1;

            if (mode[0] == 'r' && mode[1] == 'l')
            {
                std::vector<double> nrms(this->len);

                for (int i = 0; i < this->len; ++i)
                {
                    //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                    //  --------------------------------------------------------------------------------
                    //  scaling        BLAS                current                             remaining
                    //  --------------------------------------------------------------------------------
                    //  5           GEMM          xnX,ZAX->xnZA                    amnA,zmZ,xnZA->zax
                    //  6           TDOT        xnZA,amnA->xZam                         zmZ,xZam->zax
                    //  5           TDOT          xZam,zmZ->zax                              zax->zax
                    at::Tensor phi;

                    if (this->A_cores[k].sizes()[1] * 2 < this->x_cores[i][k].sizes()[0] * this->x_cores[i][k].sizes()[2] * this->A_cores[k].sizes()[3])
                    {
                        auto tmp = at::tensordot(this->x_cores[i][k], this->phis_z[i][k + 1], {2}, {2});
                        tmp = at::tensordot(tmp, this->A_cores[k], {1, 3}, {2, 3});
                        phi = at::tensordot(z, tmp, {1, 2}, {3, 1}).permute({0, 2, 1});
                    }
                    else
                    {
                        auto tmp = at::tensordot(z, this->A_cores[k], {1}, {1});
                        tmp = at::tensordot(tmp, this->x_cores[i][k], {3}, {1});
                        phi = at::tensordot(tmp, this->phis_z[i][k + 1], {1, 3, 5}, {0, 1, 2});
                    }
                    // Alternative
                    //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                    //  --------------------------------------------------------------------------------
                    //  scaling        BLAS                current                             remaining
                    //  --------------------------------------------------------------------------------
                    //  6           TDOT        zmZ,amnA->zZanA                    ZAX,xnX,zZanA->zax
                    //  7           TDOT      zZanA,xnX->zZaAxX                       ZAX,zZaAxX->zax
                    //  6    GEMV/EINSUM        zZaAxX,ZAX->zax                              zax->zax

                    if (return_norm)
                    {
                        nrms[i] = torch::norm(phi).item<double>();
                    }
                    if (norm != 1.0)
                        phi = phi / norm;
                    this->phis_z[i][k] = phi.clone();
                }
                if (return_norm)
                {
                    nrm_phi = *std::max_element(std::begin(nrms), std::end(nrms));
                    for (int i = 0; i < this->len; ++i)
                        this->phis_z[i][k] /= nrm_phi;
                }
                else
                    nrm_phi = 1;
            }
            else
            {
                std::vector<double> nrms(this->len);

                for (int i = 0; i < this->len; ++i)
                {

                    //   Complete contraction:  zax,zmZ,amnA,xnX->ZAX
                    // --------------------------------------------------------------------------------
                    // scaling        BLAS                current                             remaining
                    // --------------------------------------------------------------------------------
                    // 5           GEMM          xnX,zax->nXza                    zmZ,amnA,nXza->ZAX
                    // 6           TDOT        nXza,amnA->XzmA                         zmZ,XzmA->ZAX
                    // 5           TDOT          XzmA,zmZ->ZAX                              ZAX->ZAX

                    at::Tensor phi;

                    auto tmp = at::tensordot(this->x_cores[i][k], this->phis_z[i][k], {0}, {2});
                    tmp = at::tensordot(tmp, this->A_cores[k], {0, 3}, {2, 0});
                    phi = at::tensordot(z, tmp, {0, 1}, {1, 2}).permute({0, 2, 1}); // zmZ, XzmA ->ZXA and permute

                    if (return_norm)
                    {
                        nrms[i] = torch::norm(phi).item<double>();
                    }
                    if (norm != 1.0)
                        phi = phi / norm;
                    this->phis_z[i][k + 1] = phi.clone();
                }
                if (return_norm)
                {
                    nrm_phi = *std::max_element(std::begin(nrms), std::end(nrms));
                    for (int i = 0; i < this->len; ++i)
                        this->phis_z[i][k + 1] /= nrm_phi;
                }
                else
                    nrm_phi = 1;
            }

            return nrm_phi;
        }

        virtual double update_phi_y(at::Tensor &y, uint32_t k, const char *mode, double norm, bool return_norm)
        {

            double nrm_phi = 1;

            if (mode[0] == 'r' && mode[1] == 'l')
            {
                std::vector<double> nrms(this->len);

                for (int i = 0; i < this->len; ++i)
                {
                    //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                    //  --------------------------------------------------------------------------------
                    //  scaling        BLAS                current                             remaining
                    //  --------------------------------------------------------------------------------
                    //  5           GEMM          xnX,ZAX->xnZA                    amnA,zmZ,xnZA->zax
                    //  6           TDOT        xnZA,amnA->xZam                         zmZ,xZam->zax
                    //  5           TDOT          xZam,zmZ->zax                              zax->zax
                    at::Tensor phi;

                    if (this->A_cores[k].sizes()[1] * 2 < this->x_cores[i][k].sizes()[0] * this->x_cores[i][k].sizes()[2] * this->A_cores[k].sizes()[3])
                    {
                        auto tmp = at::tensordot(this->x_cores[i][k], this->phis_y[i][k + 1], {2}, {2});
                        tmp = at::tensordot(tmp, this->A_cores[k], {1, 3}, {2, 3});
                        phi = at::tensordot(y, tmp, {1, 2}, {3, 1}).permute({0, 2, 1});
                    }
                    else
                    {
                        auto tmp = at::tensordot(y, this->A_cores[k], {1}, {1});
                        tmp = at::tensordot(tmp, this->x_cores[i][k], {3}, {1});
                        phi = at::tensordot(tmp, this->phis_y[i][k + 1], {1, 3, 5}, {0, 1, 2});
                    }
                    // Alternative
                    //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                    //  --------------------------------------------------------------------------------
                    //  scaling        BLAS                current                             remaining
                    //  --------------------------------------------------------------------------------
                    //  6           TDOT        zmZ,amnA->zZanA                    ZAX,xnX,zZanA->zax
                    //  7           TDOT      zZanA,xnX->zZaAxX                       ZAX,zZaAxX->zax
                    //  6    GEMV/EINSUM        zZaAxX,ZAX->zax                              zax->zax

                    if (return_norm)
                    {
                        nrms[i] = torch::norm(phi).item<double>();
                    }
                    if (norm != 1.0)
                        phi = phi / norm;
                    this->phis_y[i][k] = phi.clone();
                }
                if (return_norm)
                {
                    nrm_phi = *std::max_element(std::begin(nrms), std::end(nrms));
                    for (int i = 0; i < this->len; ++i)
                        this->phis_y[i][k] /= nrm_phi;
                }
                else
                    nrm_phi = 1;
            }
            else
            {

                std::vector<double> nrms(this->len);

                for (int i = 0; i < this->len; ++i)
                {

                    at::Tensor phi;

                    auto tmp = at::tensordot(this->x_cores[i][k], this->phis_y[i][k], {0}, {2});
                    tmp = at::tensordot(tmp, this->A_cores[k], {0, 3}, {2, 0});
                    phi = at::tensordot(y, tmp, {0, 1}, {1, 2}).permute({0, 2, 1}); // zmZ, XzmA ->ZXA and permute

                    if (return_norm)
                    {
                        nrms[i] = torch::norm(phi).item<double>();
                    }
                    if (norm != 1.0)
                        phi = phi / norm;
                    this->phis_y[i][k + 1] = phi.clone();
                }
                if (return_norm)
                {
                    nrm_phi = *std::max_element(std::begin(nrms), std::end(nrms));
                    for (int i = 0; i < this->len; ++i)
                        this->phis_y[i][k + 1] /= nrm_phi;
                }
                else
                    nrm_phi = 1;
            }

            return nrm_phi;
        }
    };
*/
/*
    class ContractorMMMultiple : public IContractor
    {

    private:
        std::vector<std::vector<at::Tensor>> A_cores;
        std::vector<std::vector<at::Tensor>> B_cores;
        std::vector<uint64_t> M;
        std::vector<uint64_t> N;
        int len;
        bool has_z;
        std::vector<std::vector<at::Tensor>> phis_y;
        std::vector<std::vector<at::Tensor>> phis_z;

    public:
        ContractorMvMultiple(std::vector<std::vector<at::Tensor>> &A_cores, std::vector<std::vector<at::Tensor>> &B_cores, std::vector<uint64_t> &M, std::vector<uint64_t> &N, bool has_z)
        {
            this->has_z = has_z;
            this->A_cores = A_cores;
            this->B_cores = B_cores;
            this->M = M;
            this->N = N;
            this->len = x_cores.size();
            auto d = M.size();

            auto options = A_cores[0].options();

            this->phis_y = std::vector<std::vector<at::Tensor>>(this->len);
            this->phis_z = std::vector<std::vector<at::Tensor>>(this->len);
            for (int k = 0; k < this->len; ++k)
            {
                this->phis_y[k] = std::vector<at::Tensor>(d + 1);
                this->phis_z[k] = std::vector<at::Tensor>(d + 1);

                this->phis_y[k][0] = torch::ones({1, 1, 1}, options);
                this->phis_z[k][0] = torch::ones({1, 1, 1}, options);
                this->phis_y[k][d] = torch::ones({1, 1, 1}, options);
                this->phis_z[k][d] = torch::ones({1, 1, 1}, options);
            }
        }

        virtual at::Tensor b_fun(uint32_t k, char first, char second)
        {
            at::Tensor result = torch::zeros({(first == 'y' ? this->phis_y[0][k].sizes()[0] : this->phis_z[0][k].sizes()[0]),  (second == 'y' ? this->phis_y[0][k + 1].sizes()[0] : this->phis_z[0][k + 1].sizes()[0]), this->A_cores[k].sizes()[1]}, this->A_cores[0].options());
            for (int i = 0; i < this->len; ++i)
            {
                auto tmp = at::tensordot(this->x_cores[i][k], second == 'y' ? this->phis_y[i][k + 1] : this->phis_z[i][k + 1], {2}, {2}); // xnX,YAX->xnYA
                tmp = at::tensordot(tmp, this->A_cores[i][k], {1, 3}, {2, 3});                                                               // xnYA,amnA->xYam
                tmp = at::tensordot(first == 'y' ? this->phis_y[i][k] : this->phis_z[i][k], tmp, {1, 2}, {2, 0});                         // yax,xYam->ymY
                result += tmp;
            }
            return result.permute({0, 2, 1});
        }

        virtual double update_phi_z(at::Tensor &z, uint32_t k, const char *mode, double norm, bool return_norm)
        {
            double nrm_phi = 1;

            if (mode[0] == 'r' && mode[1] == 'l')
            {
                std::vector<double> nrms(this->len);

                for (int i = 0; i < this->len; ++i)
                {
                    //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                    //  --------------------------------------------------------------------------------
                    //  scaling        BLAS                current                             remaining
                    //  --------------------------------------------------------------------------------
                    //  5           GEMM          xnX,ZAX->xnZA                    amnA,zmZ,xnZA->zax
                    //  6           TDOT        xnZA,amnA->xZam                         zmZ,xZam->zax
                    //  5           TDOT          xZam,zmZ->zax                              zax->zax
                    at::Tensor phi;

                    if (this->A_cores[k].sizes()[1] * 2 < this->x_cores[i][k].sizes()[0] * this->x_cores[i][k].sizes()[2] * this->A_cores[k].sizes()[3])
                    {
                        auto tmp = at::tensordot(this->x_cores[i][k], this->phis_z[i][k + 1], {2}, {2});
                        tmp = at::tensordot(tmp, this->A_cores[k], {1, 3}, {2, 3});
                        phi = at::tensordot(z, tmp, {1, 2}, {3, 1}).permute({0, 2, 1});
                    }
                    else
                    {
                        auto tmp = at::tensordot(z, this->A_cores[k], {1}, {1});
                        tmp = at::tensordot(tmp, this->x_cores[i][k], {3}, {1});
                        phi = at::tensordot(tmp, this->phis_z[i][k + 1], {1, 3, 5}, {0, 1, 2});
                    }
                    // Alternative
                    //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                    //  --------------------------------------------------------------------------------
                    //  scaling        BLAS                current                             remaining
                    //  --------------------------------------------------------------------------------
                    //  6           TDOT        zmZ,amnA->zZanA                    ZAX,xnX,zZanA->zax
                    //  7           TDOT      zZanA,xnX->zZaAxX                       ZAX,zZaAxX->zax
                    //  6    GEMV/EINSUM        zZaAxX,ZAX->zax                              zax->zax

                    if (return_norm)
                    {
                        nrms[i] = torch::norm(phi).item<double>();
                    }
                    if (norm != 1.0)
                        phi = phi / norm;
                    this->phis_z[i][k] = phi.clone();
                }
                if (return_norm)
                {
                    nrm_phi = *std::max_element(std::begin(nrms), std::end(nrms));
                    for (int i = 0; i < this->len; ++i)
                        this->phis_z[i][k] /= nrm_phi;
                }
                else
                    nrm_phi = 1;
            }
            else
            {
                std::vector<double> nrms(this->len);

                for (int i = 0; i < this->len; ++i)
                {

                    //   Complete contraction:  zax,zmZ,amnA,xnX->ZAX
                    // --------------------------------------------------------------------------------
                    // scaling        BLAS                current                             remaining
                    // --------------------------------------------------------------------------------
                    // 5           GEMM          xnX,zax->nXza                    zmZ,amnA,nXza->ZAX
                    // 6           TDOT        nXza,amnA->XzmA                         zmZ,XzmA->ZAX
                    // 5           TDOT          XzmA,zmZ->ZAX                              ZAX->ZAX

                    at::Tensor phi;

                    auto tmp = at::tensordot(this->x_cores[i][k], this->phis_z[i][k], {0}, {2});
                    tmp = at::tensordot(tmp, this->A_cores[k], {0, 3}, {2, 0});
                    phi = at::tensordot(z, tmp, {0, 1}, {1, 2}).permute({0, 2, 1}); // zmZ, XzmA ->ZXA and permute

                    if (return_norm)
                    {
                        nrms[i] = torch::norm(phi).item<double>();
                    }
                    if (norm != 1.0)
                        phi = phi / norm;
                    this->phis_z[i][k + 1] = phi.clone();
                }
                if (return_norm)
                {
                    nrm_phi = *std::max_element(std::begin(nrms), std::end(nrms));
                    for (int i = 0; i < this->len; ++i)
                        this->phis_z[i][k + 1] /= nrm_phi;
                }
                else
                    nrm_phi = 1;
            }

            return nrm_phi;
        }

        virtual double update_phi_y(at::Tensor &y, uint32_t k, const char *mode, double norm, bool return_norm)
        {

            double nrm_phi = 1;

            if (mode[0] == 'r' && mode[1] == 'l')
            {
                std::vector<double> nrms(this->len);

                for (int i = 0; i < this->len; ++i)
                {
                    //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                    //  --------------------------------------------------------------------------------
                    //  scaling        BLAS                current                             remaining
                    //  --------------------------------------------------------------------------------
                    //  5           GEMM          xnX,ZAX->xnZA                    amnA,zmZ,xnZA->zax
                    //  6           TDOT        xnZA,amnA->xZam                         zmZ,xZam->zax
                    //  5           TDOT          xZam,zmZ->zax                              zax->zax
                    at::Tensor phi;

                    if (this->A_cores[k].sizes()[1] * 2 < this->x_cores[i][k].sizes()[0] * this->x_cores[i][k].sizes()[2] * this->A_cores[k].sizes()[3])
                    {
                        auto tmp = at::tensordot(this->x_cores[i][k], this->phis_y[i][k + 1], {2}, {2});
                        tmp = at::tensordot(tmp, this->A_cores[k], {1, 3}, {2, 3});
                        phi = at::tensordot(y, tmp, {1, 2}, {3, 1}).permute({0, 2, 1});
                    }
                    else
                    {
                        auto tmp = at::tensordot(y, this->A_cores[k], {1}, {1});
                        tmp = at::tensordot(tmp, this->x_cores[i][k], {3}, {1});
                        phi = at::tensordot(tmp, this->phis_y[i][k + 1], {1, 3, 5}, {0, 1, 2});
                    }
                    // Alternative
                    //  Complete contraction:  ZAX,amnA,xnX,zmZ->zax
                    //  --------------------------------------------------------------------------------
                    //  scaling        BLAS                current                             remaining
                    //  --------------------------------------------------------------------------------
                    //  6           TDOT        zmZ,amnA->zZanA                    ZAX,xnX,zZanA->zax
                    //  7           TDOT      zZanA,xnX->zZaAxX                       ZAX,zZaAxX->zax
                    //  6    GEMV/EINSUM        zZaAxX,ZAX->zax                              zax->zax

                    if (return_norm)
                    {
                        nrms[i] = torch::norm(phi).item<double>();
                    }
                    if (norm != 1.0)
                        phi = phi / norm;
                    this->phis_y[i][k] = phi.clone();
                }
                if (return_norm)
                {
                    nrm_phi = *std::max_element(std::begin(nrms), std::end(nrms));
                    for (int i = 0; i < this->len; ++i)
                        this->phis_y[i][k] /= nrm_phi;
                }
                else
                    nrm_phi = 1;
            }
            else
            {

                std::vector<double> nrms(this->len);

                for (int i = 0; i < this->len; ++i)
                {

                    at::Tensor phi;

                    auto tmp = at::tensordot(this->x_cores[i][k], this->phis_y[i][k], {0}, {2});
                    tmp = at::tensordot(tmp, this->A_cores[k], {0, 3}, {2, 0});
                    phi = at::tensordot(y, tmp, {0, 1}, {1, 2}).permute({0, 2, 1}); // zmZ, XzmA ->ZXA and permute

                    if (return_norm)
                    {
                        nrms[i] = torch::norm(phi).item<double>();
                    }
                    if (norm != 1.0)
                        phi = phi / norm;
                    this->phis_y[i][k + 1] = phi.clone();
                }
                if (return_norm)
                {
                    nrm_phi = *std::max_element(std::begin(nrms), std::end(nrms));
                    for (int i = 0; i < this->len; ++i)
                        this->phis_y[i][k + 1] /= nrm_phi;
                }
                else
                    nrm_phi = 1;
            }

            return nrm_phi;
        }
    };
*/
}