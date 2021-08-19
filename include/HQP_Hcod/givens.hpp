#ifndef __solver_givens_h__
#define __solver_givens_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace hcod{
    class Givens{
        public:
            Givens();
            ~Givens(){};
            void compute_rotation(const Eigen::VectorXd &x, const unsigned int &i, const unsigned int &j);        
        
        private: 
              

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