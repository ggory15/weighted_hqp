#ifndef __solver_Up_h__
#define __solver_Up_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "weighted_hqp/HCod.hpp"
#include "weighted_hqp/givens.hpp"

namespace hcod{
    class Up{
        public:
            Up(){
                given_ = new Givens();
            };
            ~Up(){};
        
        private: 
            
        public:
            bool compute(const int kup, const int cup, const int bound, const std::vector<h_structure> & h, const Eigen::MatrixXd & Y, const bool & isweighted = true);
            std::vector<h_structure> geth(){
                return h_;
            };
            Eigen::MatrixXd getY(){
                return Y_;
            };

        private:
            Eigen::MatrixXd Y_, Yup_, Wi_, Yi_;
            std::vector<H_structure> h_;
            bool _isweighted;
            Givens* given_;
            int nh_, p_;

    };
}

#endif