
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
            at::Tensor result = torch::zeros({(first == 'y' ? this->phis_y[0][k].sizes()[0] : this->phis_z[0][k].sizes()[0]), (second == 'y' ? this->phis_y[0][k + 1].sizes()[0] : this->phis_z[0][k + 1].sizes()[0]), this->A_cores[k].sizes()[1]}, this->A_cores[0].options());
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

    class ContractorMMMultiple : public IContractor
    {

    private:
        std::vector<std::vector<at::Tensor>> A_cores; // cores or bands
        std::vector<std::vector<at::Tensor>> B_cores; // cores or bands
        std::vector<uint64_t> M;
        std::vector<uint64_t> N;
        int len;
        bool has_z;
        std::vector<std::vector<at::Tensor>> phis_y;
        std::vector<std::vector<at::Tensor>> phis_z;
        std::vector<std::vector<int>> bands_A, bands_B;

    public:
        static at::Tensor local_AB(at::Tensor &phi_left, at::Tensor &phi_right, at::Tensor &coreA, at::Tensor &coreB, int bandA, int bandB)
        {

            if (bandA >= 0 && bandB >= 0)
                if (bandA >= bandB)
                    bandA = -1;
                else
                    bandB = -1;

            at::Tensor core;

            if (bandA < 0)
                if (bandB < 0)
                {
                    auto tmp = at::tensordot(coreB, phi_left, {0}, {2});                         // bknB,rab->knBra
                    auto tmp2 = at::tensordot(phi_right, coreA, {0}, {3});                       // RAB,amkA->RBamk
                    core = at::tensordot(tmp2, tmp, {1, 2, 4}, {2, 4, 0}).permute({3, 1, 2, 0}); // RBamk,knBra->Rmnr and transpose
                }
                else
                {
                    auto tmp = at::tensordot(coreB, phi_left, {1}, {2});   // lbBn,rab->lBnra
                    auto tmp2 = at::tensordot(phi_right, coreA, {1}, {3}); // RAB,amnA->RBamn
                    auto cores = at::einsum("lBnra,RBamn->lrmnR", {tmp, tmp2});

                    core = torch::zeros({cores.sizes()[1], cores.sizes()[2], cores.sizes()[3], cores.sizes()[4]}, phi_left.options());

                    for (int i = -bandB; i < 0; ++i)
                        core += torch::constant_pad_nd(cores.index({i + bandB, torch::indexing::Ellipsis, torch::indexing::Ellipsis, -i, torch::indexing::Ellipsis}), {0, 0, 0, -i});
                    for (int i = 0; i <= bandB; ++i)
                        core += torch::constant_pad_nd(cores.index({i + bandB, torch::indexing::Ellipsis, torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, -i), torch::indexing::Ellipsis}), {0, 0, i, 0});
                }
            else
            {
                auto tmp = at::tensordot(coreA, phi_left, {1}, {1});   // laAm,rab->lAmrb
                auto tmp2 = at::tensordot(phi_right, coreA, {1}, {3}); // RAB,bmnB->RAbmn
                auto cores = at::einsum("lAmrb,RAbmn->lrmnR", {tmp, tmp2});

                core = torch::zeros({cores.sizes()[1], cores.sizes()[2], cores.sizes()[3], cores.sizes()[4]}, phi_left.options());

                for (int i = -bandA; i < 0; ++i)
                    core += torch::constant_pad_nd(cores.index({i + bandA, torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, i), torch::indexing::Ellipsis,torch::indexing::Ellipsis}), {0, 0, -i, 0});
                for (int i = 0; i <= bandA; ++i)
                    core += torch::constant_pad_nd(cores.index({i + bandA, torch::indexing::Ellipsis, torch::indexing::Slice(i,torch::indexing::None), torch::indexing::Ellipsis,torch::indexing::Ellipsis}), {0, 0, 0, i});
            }

            return core;
        }

        static at::Tensor compute_phi_AB(char order, at::Tensor &phi_now, at::Tensor &coreA, at::Tensor &coreB, at::Tensor &core, int bandA, int bandB){

            if(coreA.sizes()[1] != coreA.sizes()[2])
                bandA = -1;
            if(coreB.sizes()[1] != coreB.sizes()[2])
                bandB = -1;
            if(bandA>=0 && bandB>=0)
                if(bandA >= bandB)
                    bandA = -1;
                else
                    bandB = -1;
            
            at::Tensor phi;

            if(bandA < 0)
            {
                if(bandB<0)
                {
                    if(order=='B')
                    {
                        //    Complete contraction:  RAB,amkA,bknB,rmnR->rab
                        //    --------------------------------------------------------------------------------
                        //    scaling        BLAS                current                             remaining
                        //    --------------------------------------------------------------------------------
                        //    6           TDOT        amkA,RAB->amkRB                  bknB,rmnR,amkRB->rab
                        //    7           TDOT      amkRB,rmnR->akBrn                       bknB,akBrn->rab
                        //    6           TDOT        akBrn,bknB->rab                              rab->rab
                        auto tmp1 = at::tensordot(coreA, phi_now, {3}, {1});   // amkA,RAB->amkRB 
                        auto tmp2 = at::tensordot(tmp2, core, {1, 4}, {1, 3}); // amkRB,rmnR->akBrn
                        phi = at::tensordot(tmp2, coreB, {1,2,4}, {1,3,2}).permute({1,0,2}); // akBrn,bknB->arb and permute
                    }
                    else
                    {
                        //  Complete contraction:  rab,amkA,bknB,rmnR->RAB
                        //    --------------------------------------------------------------------------------
                        //    scaling        BLAS                current                             remaining
                        //    --------------------------------------------------------------------------------
                        //    6           GEMM        bknB,rab->knBra                  amkA,rmnR,knBra->RAB
                        //    7           TDOT      knBra,rmnR->kBamR                       amkA,kBamR->RAB
                        //    6           TDOT        kBamR,amkA->RAB                              RAB->RAB
                        auto tmp1 = at::tensordot(coreB, phi, {0}, {2}); // bknB,rab->knBra
                        auto tmp2 = at::tensordot(tmp1, core, {1,3}, {2,0}); // knBra,rmnR->kBamR
                        phi = at::tensordot(coreA, tmp2, {0,1,2}, {2,3,0}); // amkA,kBamR->RAB
                    }
                }
                else
                {

                    if(order=='B')
                    {
                        //  Complete contraction:  RAB,amkA,lbBk,lrmkR->rab
                        //--------------------------------------------------------------------------------
                        //scaling        BLAS                current                             remaining
                        //--------------------------------------------------------------------------------
                        //   6           TDOT        amkA,RAB->amkRB                 lbBk,lrmkR,amkRB->rab
                        //   7              0     amkRB,lrmkR->akBlr                       lbBk,akBlr->rab
                        //   6           TDOT        akBlr,lbBk->rab                              rab->rab

                    }
                    else
                    {
                        //Complete contraction:  rab,amkA,lbBk,lrmkR->RAB
                        //--------------------------------------------------------------------------------
                        //scaling        BLAS                current                             remaining
                        //--------------------------------------------------------------------------------
                        //   6           TDOT        lbBk,rab->lBkra                 amkA,lrmkR,lBkra->RAB
                        //   7              0     lBkra,lrmkR->BkamR                       amkA,BkamR->RAB
                        //   6           TDOT        BkamR,amkA->RAB                              RAB->RAB
                        phi = torch::zeros({core.sizes()[3], coreA.sizes()[3], coreB.sizes()[3]}, coreA.options());
                        for(int i=-bandB;i<0;++i)
                        {
                            auto core_tmp = at::constant_pad_nd(core.index({torch::indexing::Ellipsis, torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, i), torch::indexing::Ellipsis}, {0,0,-i,0})); 
                            
                        }
                    }
                }
            }
            else
            {

            }
        }

        ContractorMMMultiple(std::vector<std::vector<at::Tensor>> &A_cores, std::vector<std::vector<at::Tensor>> &B_cores, std::vector<uint64_t> &M, std::vector<uint64_t> &N, bool has_z, std::vector<std::vector<int>> bands_A, std::vector<std::vector<int>> bands_B)
        {
            this->has_z = has_z;
            auto d = M.size();

            this->bands_A = bands_A;
            this->bands_B = bands_B;

            if (bands_A.size() == 0 && bands_B.size() == 0)
            {
                this->A_cores = A_cores;
                this->B_cores = B_cores;
            }
            else
            {
                this->A_cores = std::vector<std::vector<at::Tensor>>(A_cores.size());
                this->B_cores = std::vector<std::vector<at::Tensor>>(B_cores.size());
                for (int k = 0; k < B_cores.size(); ++k)
                {
                    this->A_cores[k] = std::vector<at::Tensor>(d);
                    this->B_cores[k] = std::vector<at::Tensor>(d);
                    for (int i = 0; i < d; ++i)
                    {
                        if(A_cores[k][i].sizes()[1] != A_cores[k][i].sizes()[2])
                            this->bands_A[k][i] = -1;
                        if(B_cores[k][i].sizes()[1] != B_cores[k][i].sizes()[2])
                            this->bands_B[k][i] = -1;
                        if(this->bands_A[k][i]>=0 && this->bands_B[k][i]>=0)
                            if(this->bands_A[k][i] >= this->bands_B[k][i])
                                this->bands_A[k][i] = -1;
                            else
                                this->bands_B[k][i] = -1;

                        if (this->bands_A[k][i] >= 0)
                        {
                            std::vector<at::Tensor> diags;
                            for (int i = -this->bands_A[k][i]; i < 0; ++i)
                                diags.push_back(torch::constant_pad_nd(at::diagonal(A_cores[k][i], i, 1, 2), {-i, 0}));
                            for (int i = 0; i <= this->bands_A[k][i]; ++i)
                                diags.push_back(torch::constant_pad_nd(at::diagonal(A_cores[k][i], i, 1, 2), {0, i}));
                            this->A_cores[k][i] = at::stack(diags);
                        }
                        else
                        {
                            this->A_cores[k][i] = A_cores[k][i];
                        }
                        if (this->bands_B[k][i] >= 0)
                        {
                            std::vector<at::Tensor> diags;
                            for (int i = -this->bands_B[k][i]; i < 0; ++i)
                                diags.push_back(torch::constant_pad_nd(at::diagonal(B_cores[k][i], i, 1, 2), {-i, 0}));
                            for (int i = 0; i <= this->bands_B[k][i]; ++i)
                                diags.push_back(torch::constant_pad_nd(at::diagonal(B_cores[k][i], i, 1, 2), {0, i}));
                            this->B_cores[k][i] = at::stack(diags);
                        }
                        else
                        {
                            this->B_cores[k][i] = B_cores[k][i];
                        }
                    }
                }
            }
            this->M = M;
            this->N = N;
            this->len = B_cores.size();
            

            auto options = A_cores[0][0].options();

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
        /*
                virtual at::Tensor b_fun(uint32_t k, char first, char second)
                {
                    at::Tensor result = torch::zeros({(first == 'y' ? this->phis_y[0][k].sizes()[0] : this->phis_z[0][k].sizes()[0]), (second == 'y' ? this->phis_y[0][k + 1].sizes()[0] : this->phis_z[0][k + 1].sizes()[0]), this->A_cores[k].sizes()[1]}, this->A_cores[0].options());
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
                }*/
    };

}