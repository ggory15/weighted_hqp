#ifndef __solver_steps_h__
#define __solver_steps_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "weighted_hqp/HCod.hpp"
#include "weighted_hqp/InitSet.hpp"
#include <algorithm>    // std::set_difference, std::sort

namespace hcod{
    class Step_length{
        public:
            Step_length();
            ~Step_length(){};
        
        private: 
            void check_bound(const double & Ax1, const Eigen::VectorXd &b, const int& typ, double& violval, int& violtype, const double & THR = 1e-8);

        public:
            void setProblem(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1, const std::vector<h_structure> &h, const Eigen::MatrixXd & Y){
                h_ = h;
                Y_ = Y;
                x0_ = x0;
                x1_ = x1;

                nh_ = Y.cols();
                p_ = h.size();
            }
            void compute();
            Eigen::Vector3i & getcst(){
                return cst_;
            }
            double & gettau(){
                return tau_;
            }
            bool & isviolation(){
                return viol_;
            }
            const Eigen::Vector3i & getcst() const {
                return cst_;
            }
            const double & gettau() const {
                return tau_;
            }
            const bool & isviolation() const {
                return viol_;
            }
           
        private:
           std::vector<h_structure> h_;
           Eigen::MatrixXd Y_;
           Eigen::Vector3i cst_;
           Eigen::VectorXd x0_, x1_;
           double tau_;
           bool viol_;      
           long int nh_, p_;   
           double THR_;

    };


}

#endif