#include "HQP_Hcod/cod.hpp"

#include <assert.h>
#include <Eigen/QR>

using namespace std;

namespace hcod{
    Cod::Cod(const Eigen::MatrixXd &A, const double &THR)
    : A_(A), THR_(THR)
    {   
        W_.setIdentity(A_.rows(), A_.rows());
        givens_t = new Givens();
        this->computation();
        
    }

    void Cod::computation(){
        this->calc_decomposition();

        rankA_ = 0;
        for (int i=0; i<std::min(A_.rows(), A_.cols()); i++)
            if(std::abs(L_(i, i)) > THR_)
                rankA_ ++;

        if (rankA_ < A_.rows())
            for (int i= rankA_-1; i>=0; i--)
                for (int j=A_.rows()-1; i>=rankA_; i--){
                    
                    givens_t->compute_rotation(L_.col(i), i, j);
                    Eigen::MatrixXd U = givens_t->getR().transpose();
                    L_ = U * L_; 
                    W_ = W_ * U.transpose();
                }
        
        L_permute_ = L_;
        
        if (rankA_ < A_.rows()){
            L_permute_.topRows(L_.rows()- rankA_) = L_.bottomRows(rankA_);
            L_permute_.bottomRows(rankA_) = L_.topRows(L_.rows() - rankA_); // check
            W_permute_.topRows(W_.rows()- rankA_) = W_.bottomRows(rankA_);
            W_permute_.bottomRows(rankA_) = W_.topRows(W_.rows() - rankA_); // check
        }
        W_permute_ = E_ * W_;     
    }
    void Cod::calc_decomposition(){
        //Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> qr(A_); // check
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A_.transpose());
        Q_ = qr.matrixQ();
        R_ = qr.matrixR().topLeftCorner(A_.cols(), A_.rows()).triangularView<Eigen::Upper>();
        E_ = qr.colsPermutation();
        L_ = R_.transpose();
    }   
}