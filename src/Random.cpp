#include "HQP_Hcod/Random.hpp"
#include <iterator>
#include <algorithm>

using namespace std;

namespace hcod{
    RandStackWithWeight::RandStackWithWeight(const unsigned int & nh, const unsigned int &p, const Eigen::VectorXi & m, const Eigen::VectorXi & r, const bool & eq_only)
    : nh_(nh), p_(p), m_(m), r_(r), eq_only_(eq_only)
    {
        svbound_(0) = 0.5;
        svbound_(1) = 1.5;

        btype_.clear();
        b_.clear();
        A_.clear();        
        W_.clear();

        Au_.setZero(m_.sum(), nh_);
        if (eq_only){
            bu_.setZero(m_.sum(), 1);
        }
        else {
            bu_.setZero(m_.sum(), 2);
        }

       this->compute_random();
    }

    void RandStackWithWeight::compute_random(){
        int zk, nk, mk, rk, rak;
        for (unsigned int i=0; i<p_; i++){
            if (i == 0){
                rak = 0;
            }
            else{
                rak = r_.head(i).sum();
            }
            mk = m_(i);
            rk = r_(i);
            zk = nh_ - rak - rk;

            Eigen::MatrixXd v;
            if (!eq_only_){
                v =  Eigen::MatrixXd::Random(mk, 2);
                for(auto row : v.rowwise())
                    std::sort(row.begin(), row.end());
                b_.push_back(v);
                int nEq = std::round(std::abs(v(0, 0)) * mk);

                Eigen::VectorXi type_tmp = Eigen::VectorXi::Ones(mk);
                for (int i=0; i < 1; i++)
                    type_tmp(nEq + i) = std::ceil(std::abs(v(0, 1)) * 3) + 1;

               btype_.push_back(type_tmp);
            }
            else{
                v =  Eigen::MatrixXd::Random(mk, 1);
                b_.push_back(v);
                btype_.push_back(Eigen::VectorXi::Ones(mk));
            }           
            
            Eigen::MatrixXd Sk(rk, rk);
            Sk.setZero();
            Sk.diagonal() = Eigen::VectorXd::Random(rk) + Eigen::VectorXd::Ones(rk) * svbound_(0);
            Eigen::MatrixXd Ak(mk, rak + rk + zk);
            Ak.block(0, 0, mk, rak).setRandom();
            Ak.block(0, rak, mk, rk)=  Sk;// check inequality
            Ak.block(0, rak + rk, mk, zk).setZero(); 

            Eigen::MatrixXd Wk = this->mrand(mk, mk, Eigen::Vector2d(1, 1));            
            
            if (i==0){
                Au_.topRows(mk) = Wk * Ak;
            }
            else{
                Au_.block(m_.head(i).sum(), 0, mk, nh_) = Wk * Ak;
            }
            Eigen::MatrixXd W_tmp(nh_, nh_);
            W_tmp.setIdentity();
            W_tmp.block(i*3, i*3, 3, 3) = Eigen::Matrix3d::Identity() * 0.001;
            W_.push_back(W_tmp);
        }
      
        Au_ = Au_ * this->mrand(nh_, nh_, Eigen::Vector2d(1, 1));
        for (unsigned int i=0; i<p_; i++){
            if (i == 0){
                bu_.topRows(m_(i)) = b_[i];
                A_.push_back(Au_.topRows(m_(i)));
            }
            else{
                bu_.block(m_.head(i).sum(), 0, mk, b_[i].cols()) = b_[i];
                A_.push_back(Au_.block(m_.head(i).sum(), 0, m_(i), nh_));
            }
        }
    }

    Eigen::MatrixXd RandStackWithWeight::mrand(int n, int m, Eigen::Vector2d  sbound){
        double smin = sbound(0);
        double slength = std::abs(sbound(0) - sbound(1));
        int r = std::min(n, m);

        Eigen::MatrixXd  rand_tmp(n, m);
        rand_tmp.setRandom();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(rand_tmp, Eigen::ComputeThinU | Eigen::ComputeThinV);

        Eigen::MatrixXd S_new(n, m);
        S_new.setZero();
       
        S_new.diagonal().head(r) = Eigen::VectorXd::Random(r) * slength + Eigen::VectorXd::Ones(r) * smin;

        return svd.matrixU() * S_new * svd.matrixV().transpose();
    }

}