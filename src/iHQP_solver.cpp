#include "weighted_hqp/iHQP_solver.hpp"

#include <assert.h>
#include <numeric>
//#define DEBUG

using namespace std;

namespace hcod{
    iHQP_solver::iHQP_solver(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound)
    : A_(A), b_(b), btype_(btype), aset_init_(aset_init), aset_bound_(aset_bound)
    {
        _isweighted = false;        
        this->set_problem();
    }
    iHQP_solver::iHQP_solver(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound, const std::vector<Eigen::MatrixXd> &W)
    : A_(A), b_(b), btype_(btype), aset_init_(aset_init), aset_bound_(aset_bound), W_(W)
    {
        _isweighted = true;
        this->set_problem();
    //         hcod_->print_h_structure(0);
    // hcod_->print_h_structure(1);
    // hcod_->print_h_structure(2);
    // getchar();

    }
    
    void iHQP_solver::set_problem(){
        iter_ = 0;
        kcheck_ = 0;
        p_ = A_.size();

        if (!_isweighted){
            hcod_ = new HCod(A_, b_, btype_, aset_init_, aset_bound_);
            Y_ = hcod_->getY();
            h_ = hcod_->geth();
            nh_ =Y_.rows();

            x0_.resize(nh_);
            ehpq_primal_ = new Ehqp_primal(h_, Y_);
            step_length_ = new Step_length();
            up_ = new Up();
        }
        else{
            hcod_ = new HCod(A_, b_, btype_, aset_init_, aset_bound_, W_);
            h_ = hcod_->geth();     
            nh_ = h_[0].A.cols();       
            x0_.resize(nh_);
            ehpq_primal_ = new Ehqp_primal(h_);
            step_length_ = new Step_length();
            up_ = new Up();
        }
    }


    Eigen::VectorXd & iHQP_solver::solve(){
        iter_ = 0;
        kcheck_ = 0;
        
        Y_ = hcod_->getY();
        nh_ =Y_.rows();
        h_ = hcod_->geth();

        while (kcheck_ < p_){
            if (!_isweighted)
                ehpq_primal_->setProblem(h_, Y_);
            else
                ehpq_primal_->setWProblem(h_);

            ehpq_primal_->compute();
            x1_ = ehpq_primal_->getx();

            if (!_isweighted)
                y1_ = ehpq_primal_->gety();

#ifdef DEBUG
cout << "x0_ " <<  x0_.transpose() << endl;
cout << "x1_ " <<  x1_.transpose() << endl;
#endif
            step_length_->setProblem(x0_, x1_, h_, Y_);
            step_length_->compute();

            if (step_length_->isviolation()){

                x0_ = (1 - step_length_->gettau()) * x0_ + step_length_->gettau() * x1_;

#ifdef DEBUG
cout << "x0_ " <<  x0_.transpose() << endl;
#endif
                int kup = step_length_->getcst()(0);
                int cup = step_length_->getcst()(1);
                int bound = step_length_->getcst()(2);               
#ifdef DEBUG
cout << "kup " <<  kup << endl;
cout << "cup " <<  cup << endl;
cout << "bound " <<  bound << endl;
#endif      
                up_->compute(kup, cup, bound, h_, Y_, _isweighted);
                if (!_isweighted)
                    Y_ = up_->getY();
                h_ = up_->geth();
          
            } 
            else{
                x0_ = x1_;
                break; //while
            }       
            iter_++;
#ifdef DEBUG

cout << "iter_  " << iter_  << endl;
#endif
        }

        return x1_;
    }
}