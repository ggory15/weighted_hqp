#ifndef __solver_cod_h__
#define __solver_cod_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace hcod{
    class Cod{
        public:
            Cod(const Eigen::MatrixXd &A, const double &THR);
            ~Cod(){};
        
        private: 
            void calc_decomposition();
            void computation();

        public:
            Eigen::MatrixXd getW(){
                return W_permute_;
            }
            Eigen::MatrixXd getL(){
                return L_permute_;
            }
            Eigen::MatrixXd getQ(){
                return Q_;
            }
            Eigen::MatrixXd getE(){
                return E_;
            }
            int getRank(){
                return rankA_;
            }

           
            
        private:
           Eigen::MatrixXd A_, Q_, R_, E_, L_, W_, L_permute_, W_permute_ ;
           double THR_;
           int rankA_;
           
    };
}

#endif