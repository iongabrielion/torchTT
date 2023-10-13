
#include "define.h"

namespace AMEn
{

    class IContractor
    {
    public:
        virtual at::Tensor b_fun(uint32_t k, char first, char second) = 0;
        virtual double update_phi_z(at::Tensor &z, uint32_t k, char *mode, double norm, bool return_norm) = 0;
        virtual double update_phi_y(at::Tensor &y, uint32_t k, char *mode, double norm, bool return_norm) = 0;
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
        }

        virtual at::Tensor b_fun(uint32_t k, char first, char second)
        {

            auto tmp = at::tensordot(this->x_cores[k], second == 'y' ? this->phis_y[k + 1] : this->phis_z[k + 1], {2}, {2}); // xnX,YAX->xnYA
            tmp = at::tensordot(tmp, this->A_cores[k], {1, 3}, {2, 3});                                                    // xnYA,amnA->xYam
            tmp = at::tensordot(first == 'y' ? this->phis_y[k] : this->phis_z[k], tmp, {1, 2}, {2, 0});                      // yax,xYam->ymY
            return tmp;
        }
    };
}