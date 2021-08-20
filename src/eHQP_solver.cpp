#include "weighted_hqp/eHQP_solver.hpp"
#include <assert.h>

using namespace std;

namespace hcod{
    eHQP_solver::eHQP_solver(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound)
    : A_(A), b_(b), btype_(btype), aset_init_(aset_init), aset_bound_(aset_bound)
    {
       _isweighted = false;        
       this->set_problem();
    }
    
    eHQP_solver::eHQP_solver(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound, const std::vector<Eigen::MatrixXd> &W)
    : A_(A), b_(b), btype_(btype),  aset_init_(aset_init), aset_bound_(aset_bound), W_(W)
    {
        _isweighted = true;
       this->set_problem();
    }

    void eHQP_solver::set_problem(){
       if (!_isweighted){
            hcod_ = new HCod(A_, b_, btype_, aset_init_, aset_bound_);
            ehpq_primal_ = new Ehqp_primal(hcod_->geth(), hcod_->getY()); 
        }
        else{
            hcod_ = new HCod(A_, b_, btype_, aset_init_, aset_bound_, W_);
            // hcod_->print_h_structure(0);
            // hcod_->print_h_structure(1);
            // getchar();
            ehpq_primal_ = new Ehqp_primal(hcod_->geth());
        }
    }


    Eigen::VectorXd eHQP_solver::solve(){
        if (!_isweighted){
            ehpq_primal_->setProblem(hcod_->geth(), hcod_->getY());
            ehpq_primal_->compute();
            return ehpq_primal_->getx();
        }
        else{
            ehpq_primal_->setWProblem(hcod_->geth());
            ehpq_primal_->compute();
            return ehpq_primal_->getx();
        }
    }
}