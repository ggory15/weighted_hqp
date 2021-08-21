#ifndef __solver_iHQP_solver_h__
#define __solver_iHQP_solver_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "weighted_hqp/HCod.hpp"
#include "weighted_hqp/ehqp_primal.hpp"
#include "weighted_hqp/step_length.hpp"
#include "weighted_hqp/Up.hpp"

namespace hcod{
    class iHQP_solver{
        public:
            iHQP_solver(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound);
            iHQP_solver(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound, const std::vector<Eigen::MatrixXd> &W);
            ~iHQP_solver(){};
        
        private: 
            void set_problem();

        public:
            Eigen::VectorXd solve();
            std::vector<H_structure> geth(){
                return h_;
            }
                        
        private:
            std::vector<Eigen::MatrixXd> A_;
            std::vector<Eigen::MatrixXd> b_;
            std::vector<Eigen::VectorXi> btype_;
            std::vector<Eigen::VectorXi> aset_init_, aset_bound_;
            std::vector<Eigen::MatrixXd> W_;

            HCod* hcod_; 
            Ehqp_primal* ehpq_primal_;
            Step_length* step_length_;            
            Up* up_;

            Eigen::VectorXd x_opt_, y1_, x1_, x0_, y0_;
            int iter_, kcheck_, p_, nh_;

            std::vector<h_structure> h_;
            Eigen::MatrixXd Y_;
            bool _isweighted;


    };
}

#endif