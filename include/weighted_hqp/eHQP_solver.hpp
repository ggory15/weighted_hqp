#ifndef __solver_eHQP_solver_h__
#define __solver_eHQP_solver_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "weighted_hqp/HCod.hpp"
#include "weighted_hqp/ehqp_primal.hpp"

namespace hcod{
    class eHQP_solver{
        public:
            eHQP_solver(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound);
            eHQP_solver(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound, const std::vector<Eigen::MatrixXd> &W);
            ~eHQP_solver(){};
        
        private: 
            void set_problem();

        public:
            Eigen::VectorXd solve();
                        
        private:
            std::vector<Eigen::MatrixXd> A_;
            std::vector<Eigen::MatrixXd> b_;
            std::vector<Eigen::VectorXi> btype_;
            std::vector<Eigen::MatrixXd> W_;
            std::vector<Eigen::VectorXi> aset_init_, aset_bound_;

            HCod* hcod_; 
            Ehqp_primal* ehpq_primal_;

            Eigen::VectorXd x_opt_;
            bool _isweighted;
    };
}

#endif