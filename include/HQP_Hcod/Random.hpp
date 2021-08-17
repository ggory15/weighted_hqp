#ifndef __solver_Random_h__
#define __solver_Random_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>

namespace hcod{
    class RandStackWithWeight{
        public:
            RandStackWithWeight(const unsigned int & nh, const unsigned int &p, const Eigen::VectorXi & m, const Eigen::VectorXi & r);
            ~RandStackWithWeight(){};
        
        private: 
            void compute_random();
            Eigen::MatrixXd mrand(int n, int m, Eigen::Vector2d sbound);

        public:
            std::vector<Eigen::MatrixXd> getA(){
                return A_;
            };
            std::vector<Eigen::VectorXd> getb(){
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
            Eigen::VectorXd getbu(){
                return bu_;
            };

        private:
            Eigen::Vector2d svbound_;
            std::vector<Eigen::MatrixXd> A_;
            std::vector<Eigen::VectorXd> b_;
            std::vector<Eigen::MatrixXd> W_;
            std::vector<Eigen::VectorXi> btype_;
            Eigen::MatrixXd Au_;
            Eigen::VectorXd bu_;

            unsigned int nh_;
            unsigned int p_;
            Eigen::VectorXi m_;
            Eigen::VectorXi r_;


           

    };
}

#endif