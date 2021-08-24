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

        Eigen::VectorXi active, activeb, idx_nh_vec;
        Eigen::VectorXi bound;

        Eigen::MatrixXd A_act;

        // for weighted function
        Eigen::MatrixXd Wk;
        Eigen::MatrixXd AkWk;
        Eigen::VectorXd sol;

        Eigen::MatrixXd Lj;
        Eigen::MatrixXd Y, Yupj, Ydownj;
        Eigen::MatrixXd Hj_c, Wj_c;

        std::vector<int> mj, nj, rj, rpj, raj;
        std::vector<Eigen::VectorXi> iwj, imj, fwj, fmj;
        std::vector<Eigen::MatrixXd> Aj, Wj, Hj;
        int rupj;

    } h_structure;   

    class HCod{
        public:
            HCod(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound);
            HCod(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound, const std::vector<Eigen::MatrixXd> &W);
            ~HCod(){};
        
        private: 
            void set_h_structure(const unsigned int & index);
            void compute_hcod();
            void clear_submatrix(){
                for (int i=0; i<p_; i++){
                    h_[i].mj.clear();
                    h_[i].imj.clear();
                    h_[i].iwj.clear();
                    h_[i].fwj.clear();
                    h_[i].fmj.clear();
                    h_[i].nj.clear();
                    h_[i].rj.clear();
                    h_[i].rpj.clear();
                    h_[i].raj.clear();
                    h_[i].Aj.clear();
                    h_[i].Hj.clear();
                    h_[i].Wj.clear();
                }
            }

        public:
            void print_h_structure(const unsigned int & index);
            std::vector<H_structure> & geth(){
                return h_;
            }
            Eigen::MatrixXd & getY(){
                return Y_;
            };
            const std::vector<H_structure> & geth() const{
                return h_;
            }
            const Eigen::MatrixXd & getY() const{
                return Y_;
            };
            
        private:
            std::vector<Eigen::MatrixXd> A_;
            std::vector<Eigen::MatrixXd> b_;
            std::vector<Eigen::VectorXi> btype_;
            std::vector<Eigen::MatrixXd> W_;
            std::vector<Eigen::VectorXi> aset_init_, aset_bound_;

            int p_, nh_;
            std::vector<H_structure> h_;
            Cod* cod_; 
            Eigen::MatrixXd  Y_;
            bool _isweighted;
    };
}

#endif