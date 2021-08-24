#include "weighted_hqp/ehqp_primal.hpp"
#include <assert.h>
#include <Eigen/QR>
using namespace std;

//#define DEBUG_QP
namespace hcod{
    Ehqp_primal::Ehqp_primal(const std::vector<h_structure> &h, const Eigen::MatrixXd & Y)
    : h_(h), Y_(Y){
        _isweighted = false;
    } 
    Ehqp_primal::Ehqp_primal(const std::vector<h_structure> &h)
    : h_(h){
        _isweighted = true;
    } 

    void Ehqp_primal::compute(){
        if (!_isweighted){
            y_.resize(h_[p_-1].ra);
            int y_idx = 0;
            for (int i=0; i<p_; i++){
                h_structure hk = h_[i];
                                
                if (hk.r > 0)
                {
                    Eigen::VectorXi im1_idx = hk.im(Eigen::VectorXi::LinSpaced(hk.m-hk.n, hk.n, hk.m-1));
                    L_ = hk.H(im1_idx, Eigen::VectorXi::LinSpaced(hk.ra-hk.rp, hk.rp, hk.ra-1));
                    if (hk.rp > 0)
                        M1_ = hk.H(im1_idx, Eigen::VectorXi::LinSpaced(hk.rp, 0, hk.rp-1));
                    W1_ = hk.W(hk.iw, im1_idx);
                    b_ = hk.b(hk.activeb, 0);

                    if (hk.rp > 0){
                        e_ = W1_.transpose() * b_ - M1_ * y_.head(hk.rp);
                    }
                    else
                        e_ = W1_.transpose() * b_;
                    
                    y_.segment(hk.rp, hk.ra-hk.rp) = L_.completeOrthogonalDecomposition().pseudoInverse() * e_;
                    y_idx = hk.ra;
                }
            }
            x_ = Y_.leftCols(h_[p_-1].ra) * y_.head(y_idx);
        }
                else{
            for (int i=0; i<p_; i++){

                h_structure hk = h_[i], hj;
                if (i>0)
                    hj = h_[i-1];
                if (hk.r > 0){
                    y_.resize(hk.ra-hk.rp);

                    Eigen::VectorXi im1_idx = hk.im(Eigen::VectorXi::LinSpaced(hk.m-hk.n, hk.n, hk.m-1));
                    L_ = hk.H(im1_idx, Eigen::VectorXi::LinSpaced(hk.ra-hk.rp, hk.rp, hk.ra-1));
                    if (hk.rp > 0)
                        M1_ = hk.H(im1_idx, Eigen::VectorXi::LinSpaced(hk.rp, 0, hk.rp-1));
                    W1_ = hk.W(hk.iw, im1_idx);
                    b_ = hk.b(hk.activeb, 0); //shuld be check 0

#ifdef DEBUG_QP
    cout << "i " << i << endl;
    cout << "hk.W \n" << hk.W << endl;
    cout << "hk.H \n " << hk.H << endl;
    cout << "hk.Y \n" << hk.Y << endl; 
    cout << "b_ " << b_.transpose() << endl;
    //getchar();
#endif
                    if (i > 0){
                        e_ = W1_.transpose() * b_ - W1_.transpose() * hk.A(hk.active, Eigen::VectorXi::LinSpaced(hk.A.cols(), 0, hk.A.cols()-1)) * hj.sol;
                        // cout << "b_" << b_.transpose() << endl; 
                        // cout << "hk.A_act" << hk.A(hk.active, Eigen::VectorXi::LinSpaced(hk.A.cols(), 0, hk.A.cols()-1)) << endl; 
                        // cout << "hj.sol" << hj.sol.transpose() << endl; 
                        // cout << "e_" << e_.transpose() << endl; 
                    }
                    else{
                        e_ = W1_.transpose() * b_;
                    }
                    y_ = L_.completeOrthogonalDecomposition().pseudoInverse() * e_;
                    // cout << "L" << L_ << endl;
                    // cout << "y" << y_.transpose() << endl; 
#ifdef DEBUG_QP
cout << "im1_idx " << im1_idx.transpose() << "   " << hk.im.transpose() << endl;
cout << "y_ " << y_.transpose() << endl;
cout << "x_e" << e_.transpose() << endl;
cout << "L" << L_ << endl;
cout << "hk.rp \t" << hk.rp << endl;
cout << "hk.ra-hk.rp \t" << hk.ra-hk.rp << endl;
#endif DEBUG_QP
            

                if (i == 0)
                    hk.sol = hk.Wk * hk.Y.block(0, hk.rp, hk.Y.rows(), hk.ra-hk.rp) * y_;
                else
                    hk.sol = hj.sol + hk.Wk * hk.Y.block(0, hk.rp, hk.Y.rows(), hk.ra-hk.rp) * y_;
                // cout << "sol" << hk.sol.transpose() << endl;
                // getchar();
                }
                h_[i] = hk;
            }
            x_ = h_[p_-1].sol;
        }
    }
}