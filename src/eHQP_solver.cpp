#include "weighted_hqp/eHQP_solver.hpp"
#include <assert.h>

using namespace std;

namespace hcod{
    eHQP_solver::eHQP_solver(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound)
    : A_(A), b_(b), btype_(btype), aset_init_(aset_init), aset_bound_(aset_bound)
    {
       this->set_problem();
    }
    
    void eHQP_solver::set_problem(){
        hcod_ = new HCod(A_, b_, btype_, aset_init_, aset_bound_);
        ehpq_primal_ = new Ehqp_primal(hcod_->geth(), hcod_->getY());
        
    }


    Eigen::VectorXd eHQP_solver::solve(){
        ehpq_primal_->setProblem(hcod_->geth(), hcod_->getY());
        ehpq_primal_->compute();
        return ehpq_primal_->getx();
    }
}