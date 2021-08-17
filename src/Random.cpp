#include "HQP_Hcod/Random.hpp"

using namespace std;

namespace hcod{
    RandStackWithWeight::RandStackWithWeight(const unsigned int & nh, const unsigned int &p, const Eigen::VectorXi & m, const Eigen::VectorXi & r)
    : nh_(nh), p_(p), m_(m), r_(r)
    {
        svbound_(0) = 0.5;
        svbound_(1) = 1.5;

        btype_.clear();
        b_.clear();
        A_.clear();        
        W_.clear();

        Au_.setZero(m_.sum(), nh_);
        bu_.setRandom(m_.sum());

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
            
            b_.push_back(Eigen::VectorXd::Random(mk));
            btype_.push_back(Eigen::VectorXi::Ones(mk));
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
                bu_.head(m_(i)) = b_[i];
                A_.push_back(Au_.topRows(m_(i)));
            }
            else{
                bu_.segment(m_.head(i).sum(), mk) = b_[i];
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