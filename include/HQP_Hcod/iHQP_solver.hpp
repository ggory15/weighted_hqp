#ifndef __solver_iHQP_solver_h__
#define __solver_iHQP_solver_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "HQP_Hcod/HCod.hpp"
#include "HQP_Hcod/ehqp_primal.hpp"
#include "HQP_Hcod/step_length.hpp"
#include "HQP_Hcod/givens.hpp"

namespace hcod{
    class iHQP_solver{
        public:
            iHQP_solver(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound);
            ~iHQP_solver(){};
        
        private: 
            void set_problem();

        public:
            Eigen::VectorXd solve();
                        
        private:
            std::vector<Eigen::MatrixXd> A_;
            std::vector<Eigen::MatrixXd> b_;
            std::vector<Eigen::VectorXi> btype_;
            std::vector<Eigen::VectorXi> aset_init_, aset_bound_;

            HCod* hcod_; 
            Ehqp_primal* ehpq_primal_;
            Step_length* step_length_;
            Givens* given_;

            Eigen::VectorXd x_opt_, y1_, x1_, x0_, y0_;
            int iter_, kcheck_, p_, nh_;

            std::vector<h_structure> h_;
            Eigen::MatrixXd Y_, Yup_, Wi_, Yi_;


    };
}

#endif