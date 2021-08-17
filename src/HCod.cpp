#include "HQP_Hcod/HCod.hpp"
#include <assert.h>

using namespace std;

#define EPS 1e-8

namespace hcod{
    HCod::HCod(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::VectorXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound)
    : A_(A), b_(b), btype_(btype), aset_init_(aset_init), aset_bound_(aset_bound)
    {
        p_ = A_.size();
        nh_ = A_[0].cols();

        h_.clear();
        Y_.setIdentity(nh_, nh_);

        for (int i=0; i<p_; i++)
            this->set_h_structure(i);

        this->compute_hcod();
    }
    
    void HCod::set_h_structure(const unsigned int & index){
        H_structure hk;
        
        hk.A = A_[index];
        hk.b = b_[index];
        hk.btype = btype_[index];
        hk.mmax = hk.A.rows();

        hk.active = aset_init_[index];
        hk.activeb = hk.active;
        for (int i=0; i<hk.active.size(); i++)
            if (aset_bound_[index](i) == 2)
                hk.activeb(i) += hk.mmax;

        hk.bound = Eigen::VectorXi::Zero(hk.mmax);
        for (int i=0; i<hk.active.size(); i++)
            hk.bound(hk.active(i)) = aset_bound_[index](i); // check
        //hk.freeze = Eigen::VectorXi::Zero(hk.mmax);
        
        hk.W = Eigen::MatrixXd::Identity(hk.mmax, hk.mmax);
        hk.H = Eigen::MatrixXd::Zero(hk.mmax, nh_);
        hk.m = hk.active.size();

        hk.iw = Eigen::VectorXi::LinSpaced(hk.m, 0, hk.m-1);
        hk.im = hk.iw;
        
        if (hk.mmax - (hk.m) > 0) {// check
            hk.fw = Eigen::VectorXi::LinSpaced(hk.mmax - hk.m, hk.m, hk.mmax-1);
            hk.fm = hk.fw;
        }

        hk.n = 0;
        hk.r = 0;
        hk.rp = 0;
        hk.ra = 0;

        hk.A_act.setZero(hk.active.size(), hk.A.cols());
        for (int i=0; i< hk.active.size(); i++)
            hk.A_act.row(i) =hk.A.row(hk.active(i));

        h_.push_back(hk);
    }


    void HCod::compute_hcod(){
        H_structure hk = h_[0];

        if (hk.m > 0){
            cod_ = new Cod(hk.A_act, EPS);
            hk.W.block(0, 0, hk.m, hk.m) = cod_->getW();
            hk.H.topRows(hk.m) = cod_->getL();
            Y_ = cod_->getQ();
            hk.r = cod_->getRank();
            hk.ra = hk.r;
            hk.n = hk.m - hk.r;
        }
        else{
            hk.ra = 0;
            Y_.setIdentity(nh_, nh_);
        }
        h_[0] = hk;

        for (int i=1; i<p_; i++){
            H_structure hk = h_[i];
            hk.rp = h_[i-1].ra;
            
            if (hk.m>0){
                cod_ = new Cod(hk.A_act * Y_.rightCols(nh_-hk.rp) , EPS); // check 
                hk.W.block(0, 0, hk.m, hk.m) = cod_->getW();
                hk.H.block(0, hk.rp, hk.m, hk.H.cols()- hk.rp) = cod_->getL();
                
                hk.r = cod_->getRank();
                hk.ra = hk.rp + hk.r;
                hk.n = hk.m-hk.r;

                hk.H.topLeftCorner(hk.m, hk.rp) = hk.W.topLeftCorner(hk.m, hk.m).transpose() * hk.A_act * Y_.leftCols(hk.rp);

                
                Y_.rightCols(nh_-hk.rp) = Y_.rightCols(nh_-hk.rp) * cod_->getQ();
            }
            else{
                hk.ra =hk.rp;
            }
            h_[i] = hk;
        }
    }
    void HCod::print_h_structure(const unsigned int & index){
        H_structure hk = h_[index];
        cout << "mmax" << hk.mmax << endl;
        cout << "active" << hk.active.transpose() << endl;
        cout << "bound" << hk.bound.transpose() << endl;
        cout << "iw" << hk.iw.transpose() << endl;
        cout << "fw" << hk.fw.transpose() << endl;

        cout << "H" << hk.H << endl;
        cout << "r" << hk.r << endl;
    }
}