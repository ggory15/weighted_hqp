#include "weighted_hqp/HCod.hpp"
#include <assert.h>

using namespace std;

#define EPS 1e-8

namespace hcod{
    HCod::HCod(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound)
    : A_(A), b_(b), btype_(btype), aset_init_(aset_init), aset_bound_(aset_bound)
    {
        p_ = A_.size();
        nh_ = A_[0].cols();

        h_.clear();
        Y_.setIdentity(nh_, nh_);
        _isweighted = false;

        for (int i=0; i<p_; i++)
            this->set_h_structure(i);
        
        this->compute_hcod();
    }
    HCod::HCod(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound, const std::vector<Eigen::MatrixXd> &W)
    : A_(A), b_(b), btype_(btype), aset_init_(aset_init), aset_bound_(aset_bound), W_(W){
        
        p_ = A_.size();
        nh_ = A_[0].cols();

        h_.clear();
        Y_.setIdentity(nh_, nh_);
        
        _isweighted = true;

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

        hk.idx_nh_vec = Eigen::VectorXi::LinSpaced(nh_ ,0, nh_-1);
        if (!_isweighted){
             hk.A_act =  hk.A(hk.active, hk.idx_nh_vec);
        }
        else{
            
            hk.Wk = W_[index].array().sqrt();
            Eigen::MatrixXd Wk = hk.Wk.completeOrthogonalDecomposition().pseudoInverse();
            hk.Wk = Wk;
            hk.AkWk = A_[index] * hk.Wk;

            hk.Yupj.setIdentity(nh_, nh_);
            hk.Ydownj.setIdentity(nh_, nh_);
            hk.Y.setIdentity(nh_, nh_);
        }       

        h_.push_back(hk);
    }


    void HCod::compute_hcod(){
        
        this->clear_submatrix();
        H_structure hk = h_[0];

        if (hk.m > 0){
            if (!_isweighted){
                cod_ = new Cod(hk.A_act, EPS);
            }
            else{
                cod_ = new Cod(hk.AkWk(hk.active, hk.idx_nh_vec), EPS);
            }

            hk.W(hk.iw, hk.im) = cod_->getW();
            hk.H(hk.im, hk.idx_nh_vec) = cod_->getL();

            if (!_isweighted)
                Y_ = cod_->getQ();
            else
                hk.Y = cod_->getQ();

            hk.r = cod_->getRank();
            hk.ra = hk.r;
            hk.n = hk.m - hk.r;
        }
        else{
            hk.ra = 0;
            if (!_isweighted)
                Y_.setIdentity(nh_, nh_);
            else
                hk.Y.setIdentity(nh_, nh_);
        }

        h_[0] = hk;
        if (!_isweighted){
            for (int i=1; i<p_; i++){
                H_structure hk = h_[i];
                hk.rp = h_[i-1].ra;
                
                if (hk.m>0){
                    
                    cod_ = new Cod(hk.A_act * Y_.rightCols(nh_-hk.rp) , EPS); // check 
                    
                    hk.W(hk.iw, hk.im) = cod_->getW();
                    hk.H(hk.im, Eigen::VectorXi::LinSpaced(hk.H.cols()- hk.rp ,hk.rp, hk.H.cols())) = cod_->getL();
                    
                    hk.r = cod_->getRank();
                    hk.ra = hk.rp + hk.r;
                    hk.n = hk.m-hk.r;

                    hk.H(hk.im, Eigen::VectorXi::LinSpaced(hk.rp ,0, hk.rp- 1)) = cod_->getW().transpose() * hk.A_act * Y_.leftCols(hk.rp);
                    Y_.rightCols(nh_-hk.rp) = Y_.rightCols(nh_-hk.rp) * cod_->getQ();
                }
                else{
                    hk.ra =hk.rp;
                }
                h_[i] = hk;
            }
        }
        else{
            for (int k=1; k<p_; k++){
                for (int j=0; j<=k-1; j++){                    
                    h_[k].mj.push_back(h_[j].m);
                    h_[k].imj.push_back(h_[j].im);
                    h_[k].iwj.push_back(h_[j].iw);
                    h_[k].fwj.push_back(h_[j].fw);
                    h_[k].fmj.push_back(h_[j].fm);
                    h_[k].nj.push_back(h_[j].n);
                    h_[k].rj.push_back(h_[j].r);
                    h_[k].rpj.push_back(h_[j].rp);
                    h_[k].raj.push_back(h_[j].ra);
                    h_[k].Aj.push_back(A_[j]*h_[k].Wk);

                    if (h_[j].m > 0){ // check condition
                        Eigen::MatrixXd AjWk_act = (A_[j] * h_[k].Wk)(h_[j].active, h_[j].idx_nh_vec);
                        cod_ = new Cod(  AjWk_act  * h_[k].Y.rightCols(nh_-h_[j].rp) , EPS);
        
                        h_[k].Wj_c.setIdentity(h_[j].mmax, h_[j].mmax);
                        h_[k].Wj_c(h_[j].iw, h_[j].im) = cod_->getW();
                        h_[k].Hj_c.setZero(h_[j].mmax, nh_);
                        if (j == 0){
                            h_[k].Hj_c(h_[j].im, h_[j].idx_nh_vec) = cod_->getL();
                        }
                        else{
                            h_[k].Hj_c(h_[j].im, Eigen::VectorXi::LinSpaced(h_[k].H.cols()- h_[j].rp ,h_[j].rp, h_[k].H.cols() - 1)) = cod_->getL();
                            h_[k].Hj_c(h_[j].im,  Eigen::VectorXi::LinSpaced(h_[j].rp ,0, h_[j].rp- 1)) = cod_->getW().transpose() * AjWk_act * h_[k].Y.leftCols(h_[j].rp) ;
                        }
                    
                        h_[k].Y.rightCols(h_[k].Y.cols() - h_[j].rp) = h_[k].Y.rightCols(h_[k].Y.cols() - h_[j].rp) * cod_->getQ();
                    
                    }
                    else{
                        h_[k].Hj_c.setZero(0, nh_); // null
                        h_[k].Wj_c.setZero(0, 0); // null
                    }
                    h_[k].Hj.push_back(h_[k].Hj_c);
                    h_[k].Wj.push_back(h_[k].Wj_c);
                }

                H_structure hk = h_[k];
                hk.rp = h_[k-1].ra;

                if (hk.m>0){                    
                    cod_ = new Cod(hk.AkWk(hk.active, hk.idx_nh_vec) * hk.Y.rightCols(nh_-hk.rp) , EPS); // check 

                    hk.W(hk.iw, hk.im) = cod_->getW();
                    hk.H(hk.im, Eigen::VectorXi::LinSpaced(hk.H.cols()- hk.rp ,hk.rp, hk.H.cols() - 1)) = cod_->getL();

                    hk.r = cod_->getRank();
                    hk.ra = hk.rp + hk.r;
                    hk.n = hk.m-hk.r;

                    hk.H(hk.im, Eigen::VectorXi::LinSpaced(hk.rp ,0, hk.rp- 1)) = cod_->getW().transpose() * hk.AkWk(hk.active, hk.idx_nh_vec) * hk.Y.leftCols(hk.rp);
                    Eigen::MatrixXd Y_tmp = hk.Y.rightCols(hk.Y.cols() - hk.rp);
                    hk.Y.rightCols(hk.Y.cols() - hk.rp) =  Y_tmp * cod_->getQ();
                }
                else{
                    hk.ra =hk.rp;
                }
                h_[k] = hk;
                
            }
        }
    }
    void HCod::print_h_structure(const unsigned int & index){
        H_structure hk = h_[index];
        cout << "mmax" << hk.mmax << endl;
        cout << "active" << hk.active.transpose() << endl;
        cout << "acive size" << hk.active.size() << endl;
        cout << "bound" << hk.bound.transpose() << endl;
        cout << "iw" << hk.iw.transpose() << endl;
        cout << "fw" << hk.fw.transpose() << endl;

        cout << "H" << hk.H << endl;
        cout << "r" << hk.r << endl;

        if (_isweighted){
            cout << "Y" << hk.Y << endl;
            cout << "Hj" << hk.Hj_c << endl;
            if (index == 2)
                cout << "Aj" << hk.Aj[1] << endl;
        }
    }
}