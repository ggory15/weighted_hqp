#ifndef __solver_HCod_h__
#define __solver_HCod_h__

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <weighted_hqp/cod.hpp>


namespace hcod{
    typedef struct H_structure {   
        Eigen::MatrixXd A;
        Eigen::MatrixXd b;
        Eigen::VectorXi btype;

        int mmax;
        int m;
        int r;
        int n;
        int ra;
        int rp;

        Eigen::VectorXi iw;
        Eigen::VectorXi im;
        Eigen::MatrixXd W;
        Eigen::MatrixXd H;
        Eigen::VectorXi fw;
        Eigen::VectorXi fm;

        Eigen::VectorXi active, activeb;
        Eigen::VectorXi bound;

        Eigen::MatrixXd A_act;
    } h_structure;   

    class HCod{
        public:
            HCod(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound);
            ~HCod(){};
        
        private: 
            void set_h_structure(const unsigned int & index);
            void compute_hcod();

        public:
            void print_h_structure(const unsigned int & index);
            std::vector<H_structure> geth(){
                return h_;
            }
            Eigen::MatrixXd getY(){
                return Y_;
            };
            
        private:
            std::vector<Eigen::MatrixXd> A_;
            std::vector<Eigen::MatrixXd> b_;
            std::vector<Eigen::VectorXi> btype_;
            std::vector<Eigen::VectorXi> aset_init_, aset_bound_;

            int p_, nh_;
            std::vector<H_structure> h_;
            Cod* cod_; 
            Eigen::MatrixXd  Y_;
    };
}

#endif