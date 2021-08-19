#include "HQP_Hcod/iHQP_solver.hpp"

#include <assert.h>
#include <numeric>
//#define DEBUG

using namespace std;

namespace hcod{
    iHQP_solver::iHQP_solver(const std::vector<Eigen::MatrixXd> &A, const std::vector<Eigen::MatrixXd> &b, const std::vector<Eigen::VectorXi> &btype, const std::vector<Eigen::VectorXi> &aset_init, const std::vector<Eigen::VectorXi> &aset_bound)
    : A_(A), b_(b), btype_(btype), aset_init_(aset_init), aset_bound_(aset_bound)
    {
        iter_ = 0;
        kcheck_ = 0;
        p_ = A_.size();
        

        this->set_problem();
    }
    
    void iHQP_solver::set_problem(){
        hcod_ = new HCod(A_, b_, btype_, aset_init_, aset_bound_);
        Y_ = hcod_->getY();
        h_ = hcod_->geth();
        nh_ =Y_.rows();

        x0_.resize(nh_);
        ehpq_primal_ = new Ehqp_primal(h_, Y_);
        step_length_ = new Step_length();
        given_ = new Givens();
    }


    Eigen::VectorXd iHQP_solver::solve(){
        iter_ = 0;
        kcheck_ = 0;
        
        Y_ = hcod_->getY();
        nh_ =Y_.rows();
        h_ = hcod_->geth();

        while (kcheck_ < p_){
           

            ehpq_primal_->setProblem(h_, Y_);
            ehpq_primal_->compute();
            x1_ = ehpq_primal_->getx();
            y1_ = ehpq_primal_->gety();

#ifdef DEBUG
cout << "Y_ " <<  Y_ << endl;
cout << "x1_ " <<  x1_.transpose() << endl;
cout << "y1_ " <<  y1_.transpose() << endl;

#endif
            step_length_->setProblem(x0_, x1_, h_, Y_);
            step_length_->compute();

            if (step_length_->isviolation()){

                x0_ = (1 - step_length_->gettau()) * x0_ + step_length_->gettau() * x1_;

#ifdef DEBUG
cout << "x0_ " <<  x0_.transpose() << endl;
#endif
                int kup = step_length_->getcst()(0);
                int cup = step_length_->getcst()(1);
                int bound = step_length_->getcst()(2);               
#ifdef DEBUG
cout << "kup " <<  kup << endl;
cout << "cup " <<  cup << endl;
cout << "bound " <<  bound << endl;
#endif
                h_[kup].iw.conservativeResize(std::distance(h_[kup].iw.begin(), h_[kup].iw.end()+ 1));
                h_[kup].iw(h_[kup].iw.size() - 1) = h_[kup].fw(0);
                h_[kup].fw = h_[kup].fw.tail(h_[kup].fw.size() - 1);             
                h_[kup].im.conservativeResize(std::distance(h_[kup].im.begin(), h_[kup].im.end()+ 1));
                h_[kup].im(h_[kup].im.size() - 1) = h_[kup].fm(0);
                h_[kup].fm = h_[kup].fm.tail(h_[kup].fm.size() - 1);
                int idx_im = h_[kup].im(h_[kup].im.size() - 1);
                h_[kup].H.row(idx_im) = h_[kup].A.row(cup) * Y_;
                int idx_iw = h_[kup].iw(h_[kup].iw.size() - 1);
                h_[kup].W.row(idx_iw).setZero();
                h_[kup].W.col(idx_im).setZero();
                h_[kup].W(idx_iw, idx_im) = 1.;
                h_[kup].m = h_[kup].m + 1;
                h_[kup].active.conservativeResize(std::distance(h_[kup].active.begin(), h_[kup].active.end()+ 1));
                h_[kup].active(h_[kup].active.size() - 1) = cup;
                h_[kup].activeb.conservativeResize(std::distance(h_[kup].activeb.begin(), h_[kup].activeb.end()+ 1));
                h_[kup].activeb(h_[kup].activeb.size() - 1) = cup;

                if (bound == 2){
                    h_[kup].activeb(h_[kup].activeb.size() - 1) += h_[kup].mmax;
                }
#ifdef DEBUG
cout << h_[kup].activeb.transpose() <<endl;
getchar();
#endif
                h_[kup].bound(cup) = bound;

                Eigen::VectorXd flip_vec = h_[kup].H.row(idx_im).array().square();
                flip_vec = flip_vec.reverse();
                Eigen::VectorXd csum = flip_vec;
                std::partial_sum(flip_vec.begin(), flip_vec.end(), csum.begin(), std::plus<double>());
                
                int rup = nh_ - std::distance(csum.begin(), std::find_if(csum.begin(), csum.end(), [](const auto& x) { return x != 0; }));
#ifdef DEBUG
cout << "rup " <<  rup << endl;
#endif
                if (rup <= h_[kup].ra){
                    if (rup > h_[kup].rp){
                        for (int i= rup - h_[kup].rp; i>=0; i--){         
                            given_->compute_rotation(h_[kup].H.col(h_[kup].rp + i)(h_[kup].im), h_[kup].n + i, h_[kup].m);
                            Wi_ = given_->getR().transpose();
#ifdef DEBUG
cout << "Wi_ " <<  Wi_ << endl;
#endif
                           
                            
                            // Eigen::MatrixXd H_tmp(h_[kup].im.size(), h_[kup].H.cols());
                            // for (int j=0; j< h_[kup].im.size(); i++)
                            //     H_tmp.row(j) = h_[kup].H.row(h_[kup].im(j));
                            // H_tmp = Wi * H_tmp;
                            // for (int j=0; j< h_[kup].im.size(); i++)
                            //     h_[kup].H.row(h_[kup].im(j)) = H_tmp.row(j);

                            Eigen::VectorXi idx_H_col_vec = Eigen::VectorXi::LinSpaced(h_[kup].H.cols() ,0, h_[kup].H.cols()-1);
                            h_[kup].H(h_[kup].im, idx_H_col_vec) = Wi_ * h_[kup].H(h_[kup].im, idx_H_col_vec);
                            h_[kup].W(h_[kup].iw, h_[kup].im) =  h_[kup].W(h_[kup].iw, h_[kup].im) * Wi_.transpose();

#ifdef DEBUG
cout << "h_[kup].H(h_[kup].im, idx_H_col_vec) " <<  h_[kup].H(h_[kup].im, idx_H_col_vec) << endl;
cout << "h_[kup].W(h_[kup].iw, h_[kup].im) " <<  h_[kup].W(h_[kup].iw, h_[kup].im) << endl;
cout << "Wi_ " <<  Wi_ << endl;
#endif
                        }                        
                    }
                    Eigen::VectorXi im_tmp = h_[kup].im;
                    h_[kup].im(0) = im_tmp(h_[kup].im.size() - 1);
                    for (int i=1; i<h_[kup].im.size(); i++)
                        h_[kup].im(i) = im_tmp(i-1);
                    h_[kup].n += 1;                    
                }
                else{
                    Yup_.setIdentity(nh_, nh_);
                    for (int i=rup-2; i>= h_[kup].ra; i--){ // check
#ifdef DEBUG
cout << "original " << h_[kup].H.row(h_[kup].im(h_[kup].im.size() -1)) << endl;
#endif
                        given_->compute_rotation(h_[kup].H.row(h_[kup].im(h_[kup].im.size() -1)), i, i+1);
                        Yi_ = given_->getR();     
                        h_[kup].H.row(h_[kup].im(h_[kup].im.size() -1)) = h_[kup].H.row(h_[kup].im(h_[kup].im.size() -1)) * Yi_;
                                                
                        Yup_ = Yup_ * Yi_;

                    }
                    h_[kup].r += 1;
                    h_[kup].ra += 1;
                    
#ifdef DEBUG
cout << "Yi_ " << Yi_ << endl;
cout << "Yup_ " <<  Yup_ << endl;

#endif
                    for (int k=kup +1; k<p_; k++){
                        Eigen::VectorXi idx_H_col_vec = Eigen::VectorXi::LinSpaced(h_[k].H.cols() ,0, h_[k].H.cols()-1);

                        h_[k].H(h_[k].im, idx_H_col_vec) = h_[k].H(h_[k].im, idx_H_col_vec) * Yup_;
                       
#ifdef DEBUG
cout << "h_[0].H(  " << h_[0].H  << endl;
cout << "h_[1].H(  " << h_[1].H  << endl;
#endif
                        if (rup < h_[k].rp + 1){

                        }
                        else if (rup > h_[k].ra) {
                            h_[k].rp += 1;
                            h_[k].ra += 1;
                        }
                        else{
                            int rdef = rup -  h_[k].rp;
                            for (int i = rdef; i>=2; i--){
                                given_->compute_rotation(h_[k].H.col(h_[k].rp + i)(h_[k].im), h_[k].n + i, h_[k].n + rdef);
                                Wi_ = given_->getR().transpose();
                                Eigen::VectorXi idx_H_col_vec = Eigen::VectorXi::LinSpaced(h_[k].H.cols() ,0, h_[k].H.cols()-1);
                                h_[k].H(h_[k].im, idx_H_col_vec) = Wi_ * h_[k].H(h_[k].im, idx_H_col_vec);
                                h_[k].W(h_[k].iw, h_[k].im) =  h_[k].W(h_[k].iw, h_[k].im) * Wi_.transpose();

#ifdef DEBUG
cout << "Wi_  " <<  Wi_ << endl;
cout << "h_[k].W(h_[k].iw, h_[k].im)  " << h_[k].W(h_[k].iw, h_[k].im)  << endl;
#endif
                            }
                            h_[k].rp += 1;
                            h_[k].r -= 1;
                            
                            h_[k].im.resize(-1 + h_[k].m);
                            h_[k].im(0) = h_[k].n + rdef;
                            for (int i =0 ; i<h_[k].n; i++)
                                h_[k].im(i) = i;
                            for (int i =h_[kup].n; i<h_[k].n + rdef - 1; i++)
                                h_[k].im(i) = i;
                            for (int i =h_[k].n + rdef; i<h_[k].m; i++)
                                h_[k].im(i) = i;
                            h_[k].n += 1;                            
                        }
#ifdef DEBUG
cout << "H_  " << h_[k].H  << endl;

#endif 
                    }
                    Y_ = Y_ * Yup_;
#ifdef DEBUG
cout << "Y_  " << Y_  << endl;
#endif                  
                }
            } 
            else{
                x0_ = x1_;
                break; //while
            }       
            iter_++;
#ifdef DEBUG

cout << "iter_  " << iter_  << endl;
#endif
        }

        return x1_;
    }
}