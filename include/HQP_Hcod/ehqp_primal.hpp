#ifndef __solver_givens_h__
#define __solver_givens_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "HQP_Hcod/HCod.hpp"

namespace hcod{
    class Ehqp_primal{
        public:
            Ehqp_primal(const std::vector<h_structure> &h, const Eigen::MatrixXd & Y);
            ~Ehqp_primal(){};
        
        private: 
            void compute();

        public:
            Eigen::VectorXd getx(){
                return x_;
            }
           
        private:
           std::vector<h_structure> h_;
           Eigen::MatrixXd Y_, L_, M1_, W1_;
           int p_, nh_;
           Eigen::VectorXd y_, b_, e_, x_;

    };
}

#endif