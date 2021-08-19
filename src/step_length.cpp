#include "weighted_hqp/step_length.hpp"
#include <assert.h>

using namespace std;

namespace hcod{
    Step_length::Step_length()
    {
        THR_ = 1e-8;
    } 

    void Step_length::compute(){
       double taumax = 1.;
       for (int i=0; i<p_; i++){
            h_structure hk = h_[i];
            Eigen::VectorXi mmax_vec = Eigen::VectorXi::LinSpaced(hk.mmax, 0, hk.mmax-1);
            Eigen::VectorXi diff_vec = Eigen::VectorXi(mmax_vec.size());

            if (hk.active.size() > 0){
                auto it = std::set_difference(mmax_vec.data(), mmax_vec.data() + mmax_vec.size(), 
                                    hk.active.data(), hk.active.data() + hk.active.size(), 
                                    diff_vec.data());
                diff_vec.conservativeResize(std::distance(diff_vec.data(), it)); 
            }
            else{
                diff_vec = mmax_vec;
            }

            for (int j=0; j<diff_vec.size(); j++){
                int idx = diff_vec[j];
                int typ1;
                double b1, val1;
                val1 = (hk.A.row(idx) * x1_)(0);
                check_bound(val1, hk.b.row(idx), hk.btype(idx), b1, typ1 );

                if (typ1 != Enone){
                    int typ0;
                    double b0, val0;
                    val0 = (hk.A.row(idx) * x0_)(0);
                    check_bound(val0, hk.b.row(idx), hk.btype(idx), b0, typ0);
                    

                    if (typ0 == Enone){
                        if (std::abs(val1 - val0) < THR_)
                            tau_ = 1.;
                        else
                            tau_ = (b1 - val0) / (val1 - val0);
                    }
                    else{
                        tau_ = 1- THR_;
                    }

                    if (tau_ < taumax){
                        taumax = tau_;
                        if (typ1 == Einf)
                            cst_ << i, idx, 1;
                        else
                            cst_ << i, idx, 2;
                    }
                }
            }
       }
       viol_ = taumax < 1.;
    }

    void Step_length::check_bound(const double & Ax1, const Eigen::VectorXd &b, const int& typ, double& violval, int& violtype, const double & THR){
        violtype = Enone;
        violval = 0.0;

        if ( (typ == Einf || typ == Edouble) && (Ax1 < b(0) - THR)){
            violval = b(0);
            violtype = Einf;
        }
        if ( (typ == Esup || typ == Edouble) && (Ax1 > b(1) + THR)){
            violval = b(1);
            violtype = Esup;
        }
    }
}