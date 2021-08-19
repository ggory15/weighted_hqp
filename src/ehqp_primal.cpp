#include "HQP_Hcod/ehqp_primal.hpp"
#include <assert.h>
#include <Eigen/QR>
using namespace std;

namespace hcod{
    Ehqp_primal::Ehqp_primal(const std::vector<h_structure> &h, const Eigen::MatrixXd & Y)
    : h_(h), Y_(Y){
        
    } 

    void Ehqp_primal::compute(){
        for (int i=0; i<p_; i++){
            h_structure hk = h_[i];
            
            if (hk.r > 0){
                L_ = hk.H.block(hk.n, hk.rp, hk.m-hk.n, hk.ra-hk.rp); // check. im always starts 0. 
                if (hk.rp > 0)
                    M1_ = hk.H.block(hk.n, 0, hk.m-hk.n, hk.rp);
                W1_ = hk.W.block(0, 0, hk.m, hk.m);
                b_.setZero(hk.activeb.size());

                for (int i=0; i<hk.activeb.size(); i++)
                    b_(i) = hk.b(hk.activeb(i));
                
                if (hk.rp > 0)
                    e_ = W1_.transpose() * b_ - M1_ * y_;
                else
                    e_ = W1_.transpose() * b_;

                // cout << "L " << L_ << endl;
                // cout << "M1 " << M1_ << endl;
                //  cout << "W1 " << W1_ << endl;
                // cout << "b_ " <<  hk.b << endl;
                // cout << "e_ " << e_.transpose() << endl;
                // cout << "y_ " << y_.transpose() << endl;
                // cout << "hk.activeb" << hk.activeb.transpose() << endl;
                // getchar();
                
                
                y_.segment(hk.rp, hk.ra-hk.rp) = L_.completeOrthogonalDecomposition().pseudoInverse() * e_;
                
            }
        }

        x_ = Y_.leftCols(h_[p_-1].ra) * y_;
    }
}