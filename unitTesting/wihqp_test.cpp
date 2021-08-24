#include <iostream>
#include <cstdio>
#include <Eigen/Core>
#include "weighted_hqp/Random.hpp"
#include "weighted_hqp/iHQP_solver.hpp"
#include "weighted_hqp/InitSet.hpp"

// #include <gtest/gtest.h>
#include <chrono>


using namespace hcod;
using namespace Eigen;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

//#define DEBUG
Initset* init_set;
iHQP_solver* solver;

bool evaluation(vector<MatrixXd> A, vector<MatrixXd> b, vector<VectorXi> btype, Eigen::VectorXd sol){
    int task_level = A.size();
    int dof = A[0].rows();

    for (int i=0; i<task_level; i++){
        for (int j=0; j<b[i].rows(); j++){
            if (btype[i](j) == 1){
               if (std::abs( (A[i].row(j) * sol) - b[i](j, 0)) > 0.001)
                    return false;
            }
            if (btype[i](j) == 2){
                if ( (A[i].row(j) * sol)(0, 0)  > b[i](j, 1))
                    return false;
                else if ( (A[i].row(j) * sol)(0, 0) < b[i](j, 0))
                    return false;
            }
            if (btype[i](j) == 3){
                if ( (A[i].row(j) * sol)(0, 0) < b[i](j, 0))
                    return false;
            }
            if (btype[i](j) == 4){
                if ( (A[i].row(j) * sol)(0, 0) > b[i](j, 1))
                    return false;
            }
        }
    }
    return true;
}
	
int main(int argc, char **argv){
	vector<MatrixXd> A;
    vector<MatrixXd> b;
    vector<VectorXi> btype;   
    vector<MatrixXd> W;

     int n_size = 6;
    int task = 2;
    A.push_back(Eigen::MatrixXd::Identity(3, 6));
    A.push_back(Eigen::MatrixXd::Identity(3, 6));
    Eigen::MatrixXd b_tmp(3, 2);
    b_tmp.col(0) << -2, -4, -1;
    b_tmp.col(1) << 1, 1, 1;
    b.push_back(b_tmp);
    b_tmp.col(0) << -3, -3, -3;
    b_tmp.col(1) << 3, 3, 3;
    b.push_back(b_tmp);
    Eigen::VectorXi type_tmp(3);
    type_tmp<< 2, 2, 2;
    btype.push_back(type_tmp);
    type_tmp << 1, 1, 1;
    btype.push_back(type_tmp);
    MatrixXd W1(6, 6), W2(6, 6);
    W1.setIdentity();
    W2.setIdentity();
    W.push_back(W1);
    W.push_back(W2);

    // A.push_back(Eigen::MatrixXd::Identity(7, 7));
    // A.push_back(Eigen::MatrixXd::Identity(7, 7));
    // Eigen::MatrixXd b_tmp(7, 2);
    // b_tmp.col(0).setOnes();
    // b_tmp.col(0) = b_tmp.col(0) * -1.0;
    // b_tmp.col(1).setOnes();
    // b.push_back(b_tmp);
    // b_tmp.col(0).setOnes();
    // b_tmp.col(1).setOnes();
    // b_tmp.col(0) = b_tmp.col(0) * -80.0;
    // b_tmp.col(1) = b_tmp.col(1) * 80.0;
    // b.push_back(b_tmp);
    
    // Eigen::VectorXi type_tmp(7);
    // type_tmp<< 2, 2, 2, 2, 2, 2, 2;
    // btype.push_back(type_tmp);
    // type_tmp << 1, 1, 1, 1, 1, 1, 1;
    // btype.push_back(type_tmp);
    // MatrixXd W1(7, 7), W2(7, 7);
    // W1.setIdentity();
    // W2.setIdentity();
    // W.push_back(W1);
    // W.push_back(W2);


    Initset Init_active(btype); 
    iHQP_solver iHQP_(A, b, btype,  Init_active.getactiveset(), Init_active.getbounds(), W);

    auto t1 = high_resolution_clock::now();
    Eigen::VectorXd x_opt = iHQP_.solve();
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms" << endl;
    cout << " " << endl;

    cout << "HCOD Solution: " << x_opt.transpose() << endl;

    return 0;
}