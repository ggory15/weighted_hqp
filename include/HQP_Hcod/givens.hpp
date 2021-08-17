#ifndef __solver_givens_h__
#define __solver_givens_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace hcod{
    class Givens{
        public:
            Givens(const Eigen::VectorXd &x, const unsigned int &i, const unsigned int &j);
            ~Givens(){};
        
        private: 
            void compute_rotation();          

        public:
           Eigen::MatrixXd getR(){
               return R_;
           }
            
        private:
           Eigen::VectorXd x_;
           unsigned int i_, j_, m_;
           Eigen::MatrixXd R_;
    };
}

#endif