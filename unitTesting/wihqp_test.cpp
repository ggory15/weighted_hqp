#include <iostream>
#include <cstdio>
#include <Eigen/Core>
#include "weighted_hqp/Random.hpp"
#include "weighted_hqp/iHQP_solver.hpp"
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


TEST(TestSuite, testCase1){
   

    vector<MatrixXd> A;
    vector<MatrixXd> b;
    vector<VectorXi> btype;   
    vector<MatrixXd> W;

    MatrixXd A1(1, 6), A2(1, 6), A3(1, 6);
    MatrixXd b1(1, 2), b2(1, 2), b3(1, 2);
    VectorXi btype1(1), btype2(1), btype3(1);
    MatrixXd W1(6, 6), W2(6, 6), W3(6, 6);

    A1.row(0) <<0.0269382958863537,	0.660537878060003,	-0.307310017221991,	0.132267718892827,	0.0845044378118006,	0.186303280211331;
    A2.row(0) <<-0.306447992459610,	0.787682124711883,	-0.239976364605327,	0.521404700947989,	0.653873583990222,	0.165962731217687;
    A3.row(0) <<-0.155185660273914,	0.695766711131059,	0.413916144333198,	-0.150979259753082,	0.261416901661546,	1.02187232786035;

    b1.row(0) <<-0.0536372461177532,	0.872689719026761;
    b2.row(0) << -0.850326698383627,	0.807723629856130;
    b3.row(0) << -0.172290331995457,	-0.142866229600455;

    btype1 <<3;
    btype2 <<1;
    btype3 <<1;

    W1.setIdentity();
    W2.setIdentity();
    W3.setIdentity();    
    W1(0, 0) = 0.001;
    W1(1, 1) = 0.001;
    W2(2, 2) = 0.001;
    W2(3, 3) = 0.001;
    W3(4, 4) = 0.001;
    W3(5, 5) = 0.001;

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

    int n_size = 10;
    int task = 4;
    RandStackWithWeight RandStack(n_size, task, 2* VectorXi::Ones(task), 2*VectorXi::Ones(task));
      
    A = RandStack.getA();
    b = RandStack.getb();
    btype = RandStack.getbtype();
    W = RandStack.getW();

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
    if (evaluation(A, b, btype,  x_opt)){
        cout << "This Solution is Feasible." << endl;
    }
    else{
        cout << "This Solution is non-Feasible." << endl;
        assert(false);
    }
  
   // cout << "Pinv Solution: " << validate_wehqp(A, b, W).transpose() << endl;
    
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