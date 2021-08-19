#ifndef __solver_Random_h__
#define __solver_Random_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace hcod{
    class RandStackWithWeight{
        public:
            RandStackWithWeight(const unsigned int & nh, const unsigned int &p, const Eigen::VectorXi & m, const Eigen::VectorXi & r, const bool & eq_only = true);
            ~RandStackWithWeight(){};
        
        private: 
            void compute_random();
            Eigen::MatrixXd mrand(int n, int m, Eigen::Vector2d sbound);

        public:
            std::vector<Eigen::MatrixXd> getA(){
                return A_;
            };
            std::vector<Eigen::MatrixXd> getb(){
                return b_;
            };
            std::vector<Eigen::MatrixXd> getW(){
                return W_;
            };
            std::vector<Eigen::VectorXi> getbtype(){
                return btype_;
            };
            Eigen::MatrixXd getAu(){
                return Au_;
            };
            Eigen::MatrixXd getbu(){
                return bu_;
            };

        private:
            Eigen::Vector2d svbound_;
            std::vector<Eigen::MatrixXd> A_;
            std::vector<Eigen::MatrixXd> b_;
            std::vector<Eigen::MatrixXd> W_;
            std::vector<Eigen::VectorXi> btype_;
            Eigen::MatrixXd Au_;
            Eigen::MatrixXd bu_;

            unsigned int nh_;
            unsigned int p_;
            Eigen::VectorXi m_;
            Eigen::VectorXi r_;
            bool eq_only_;
    };
}

#endif