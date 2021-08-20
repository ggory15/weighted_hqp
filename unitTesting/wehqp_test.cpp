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

    // MatrixXd A1(2, 6), A2(2, 6);
    // MatrixXd b1(2, 1), b2(2, 1);
    // VectorXi btype1(2), btype2(2);
    // MatrixXd W1(6, 6), W2(6, 6);
    // A1.row(0) << 0.366742341559392,	-0.302725472735441,	0.277662737567718	,-0.0755104628570935,	-0.274669578152753,	-0.106172347191282;
    // A1.row(1) << 0.278455370616141,	0.597115687251413,	0.0109921178447200,	-0.177534354758469,	0.432419712041074,	0.131616734450673;
    // A2.row(0) << -0.00819021186435769	,-0.255920350099915,0.224534719875352	,0.180124980559449,	0.696149211075684,	-0.618139862183112;
    // A2.row(1) << 0.845901714143818,	0.327997808330170,	1.25215384180023,	0.303913326816249,	0.288298334312046,	-0.388675901304074;
    // b1.row(0) << 0.922331997796276;
    // b1.row(1) << 0.770954220673925;
    // b2.row(0) <<  0.623716412667443;
    // b2.row(1) << 0.236444932640910;
    // btype1 << 1, 1;
    // btype2 << 1, 1;
    // W1.setIdentity();
    // W2.setIdentity();
    // W1.topLeftCorner(3,3) *= 0.001;
    // W2.bottomRightCorner(3,3) *= 0.001;

    // A.push_back(A1);
    // A.push_back(A2);
    // b.push_back(b1);
    // b.push_back(b2);
    // btype.push_back(btype1);
    // btype.push_back(btype2);
    // W.push_back(W1);
    // W.push_back(W2);

    int n_size = 10;
    RandStackWithWeight RandStack(n_size, 3, Vector3i(2, 2, 2), Vector3i(2, 2, 2));
      
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