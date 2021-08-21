#include <iostream>
#include <cstdio>
#include <Eigen/Core>
#include "weighted_hqp/Random.hpp"
#include "weighted_hqp/eHQP_solver.hpp"
#include "weighted_hqp/InitSet.hpp"

#include <gtest/gtest.h>
#include <chrono>


using namespace hcod;
using namespace Eigen;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

//#define DEBUG

Eigen::VectorXd validate_wehqp(vector<MatrixXd> A, vector<MatrixXd> b, vector<MatrixXd> W){
    int task_level = A.size();
    int dof = A[0].cols();
    VectorXd prev_sol, current_sol, total_sol;
    prev_sol.setZero(dof);
    current_sol.setZero(dof);
    total_sol.setZero(dof);

    for (int i=0; i<task_level; i++){
        MatrixXd Ai = A[i], Aj, AWi;
        MatrixXd bi = b[i];
        MatrixXd Wi = W[i];
        Wi = Wi.completeOrthogonalDecomposition().pseudoInverse();
        Wi = Wi.array().sqrt();
        
        MatrixXd Ni;
        Ni.setIdentity(dof, dof);

        for (int j= 0; j<=i-1; j++){
            Aj = A[j] * Wi * Ni;
            Ni = Ni * (MatrixXd::Identity(dof, dof) - Aj.completeOrthogonalDecomposition().pseudoInverse() * Aj);
        }
        AWi = Ai * Wi * Ni;

        current_sol = Wi * AWi.completeOrthogonalDecomposition().pseudoInverse() * (bi - Ai* prev_sol);

        total_sol += current_sol;
        prev_sol = total_sol;
    }

    return total_sol;
}

TEST(TestSuite, testCase1){
   

    vector<MatrixXd> A;
    vector<MatrixXd> b;
    vector<VectorXi> btype;   
    vector<MatrixXd> W;

    MatrixXd A1(1, 4), A2(1, 4), A3(1, 4);
    MatrixXd b1(1, 1), b2(1, 1), b3(1, 1);
    VectorXi btype1(1), btype2(1), btype3(1);
    MatrixXd W1(4, 4), W2(4, 4), W3(4, 4);

    A1.row(0) <<-0.7907, 0.2047, 0.4647, 0.4733;
    A2.row(0) <<-0.0103, 0.6034, -0.2287, 0.7969;
    A3.row(0) <<-0.1424, 1.1300, 0.9909, 0.2378;

    b1.row(0) << 0.9081;
    b2.row(0) << 0.8051;
    b3.row(0) << 0.5323;

    btype1 <<1;
    btype2 <<1;
    btype3 <<1;

    W1.setIdentity();
    W2.setIdentity();
    W3.setIdentity();    
    W1(0, 0) = 0.001;
    W2(1, 1) = 0.001;
    W3(2, 2) = 0.001;

    A.push_back(A1);
    A.push_back(A2);
    A.push_back(A3);
    b.push_back(b1);
    b.push_back(b2);
    b.push_back(b3);
    btype.push_back(btype1);
    btype.push_back(btype2);
    btype.push_back(btype3);
    W.push_back(W1);
    W.push_back(W2);
    W.push_back(W3);

    int n_size = 20;
    int task = 5;
    RandStackWithWeight RandStack(n_size, task, 2* VectorXi::Ones(task), 2*VectorXi::Ones(task));
      
    A = RandStack.getA();
    b = RandStack.getb();
    btype = RandStack.getbtype();
    W = RandStack.getW();

    Initset Init_active(btype); 
    eHQP_solver eHQP_(A, b, btype,  Init_active.getactiveset(), Init_active.getbounds(), W);

    auto t1 = high_resolution_clock::now();
    Eigen::VectorXd x_opt = eHQP_.solve();
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms" << endl;
    cout << " " << endl;

    cout << "HCOD Solution: " << x_opt.transpose() << endl;

    cout << "Pinv Solution: " << validate_wehqp(A, b, W).transpose() << endl;
    
    // MatrixXd J1 = RandStack.getA()[0];
    // MatrixXd J2 = RandStack.getA()[1];
    // MatrixXd x1 = RandStack.getb()[0];
    // MatrixXd x2 = RandStack.getb()[1];
    // MatrixXd J1_inv = J1.completeOrthogonalDecomposition().pseudoInverse();
    // MatrixXd N1 = MatrixXd::Identity(n_size, n_size) - J1_inv * J1;
    // MatrixXd q1_star = J1_inv* x1;
    // MatrixXd J2_inv = (J2 * N1).completeOrthogonalDecomposition().pseudoInverse();
    // MatrixXd q2_star = J2_inv * (x2 - J2 * q1_star);

    // cout << "Pinv Solution: " << (q1_star + q2_star).transpose() << endl;
}

	
int main(int argc, char **argv){
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}