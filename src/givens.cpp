#include "weighted_hqp/givens.hpp"
#include <assert.h>

using namespace std;

namespace hcod{
    Givens::Givens()
    {
        
    }
    
    void Givens::compute_rotation(const Eigen::VectorXd &x, const unsigned int &i, const unsigned int &j){
        x_ = x;
        i_= i;
        j_ = j;

        m_ = x_.size();
        R_.setIdentity(m_, m_);
        double xi = x_(i_);
        double xj = x_(j_);
        double c, s, theta;

        if (abs(xj) < 1e-9){
            c=1;
            s=0;
        }
        else if (abs(xj) > abs(xi)){
            theta = -xi/xj;
            s = 1 / sqrt(1 + theta * theta);
            c = s * theta;
        }
        else{
            theta = -xj/xi;
            c = 1 / sqrt(1 + theta * theta);
            s = c * theta;
        }
        R_(i_, i_) = c;
        R_(i_, j_) = s;
        R_(j_, j_) = c;
        R_(j_, i_) = -s;
    }
 
}