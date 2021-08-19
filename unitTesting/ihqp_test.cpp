#include <iostream>
#include <cstdio>
#include <Eigen/Core>
#include "weighted_hqp/Random.hpp"
#include "weighted_hqp/iHQP_solver.hpp"
#include "weighted_hqp/InitSet.hpp"

#include <gtest/gtest.h>
#include <chrono>
// #include <qpOASES.hpp>

using namespace hcod;
using namespace Eigen;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

TEST(TestSuite, testCase1)
{   
    vector<MatrixXd> A;
    vector<MatrixXd> b;
    vector<VectorXi> btype;
    // MatrixXd A1(1, 6), A2(2, 6);
    // MatrixXd b1(1, 2), b2(2, 2);
    // VectorXi btype1(1), btype2(2);
    // A1.row(0) << -0.397019015131833,	0.194638123362162,	-0.0918517126877779,	0.152455280455517,	0.294210588222668,	0.367206488786295;
    // A2.row(0) << -0.0669660502760966,	0.391136506048131,	0.484365985651431,	-0.325006635749671,	0.631541484855456,	0.313147098065572;
    // A2.row(1) << -0.0518231356068072,	-0.236362837710798,	0.0479884169362985,	0.716726479423114, 0.793359675776558,	0.491762699912157;
    // b1.row(0) << -0.732378029287748	,0.342925778956061;
    // b2.row(0) <<  -0.934120214500247,	0.816204833013900;
    // b2.row(1) << -0.892274147128888,	0.104350053431669;
    // btype1(0) = 1;
    // btype2 << 1, 1;

    // A.push_back(A1);
    // A.push_back(A2);
    // b.push_back(b1);
    // b.push_back(b2);
    // btype.push_back(btype1);
    // btype.push_back(btype2);
    

    int n_size = 20;
    RandStackWithWeight RandStack(n_size, 3, Vector3i(3, 3, 3), Vector3i(3, 3, 3), false);

    A = RandStack.getA();
    b = RandStack.getb();
    btype = RandStack.getbtype();

    Initset Init_active(btype);
    iHQP_solver iHQP_(A, b, btype, Init_active.getactiveset(), Init_active.getbounds());

    auto t1 = high_resolution_clock::now();
    Eigen::VectorXd x_opt = iHQP_.solve();
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms" << endl;
    cout << " " << endl;

    cout << "HCOD Solution: " << x_opt.transpose() << endl << endl;

    
       
    // MatrixXd J1 = RandStack.getA()[0];
    // MatrixXd J2 = RandStack.getA()[1];
    // VectorXd x1 = RandStack.getb()[0];
    // VectorXd x2 = RandStack.getb()[1];
    // MatrixXd J1_inv = J1.completeOrthogonalDecomposition().pseudoInverse();
    // MatrixXd N1 = MatrixXd::Identity(n_size, n_size) - J1_inv * J1;
    // VectorXd q1_star = J1_inv* x1;
    // MatrixXd J2_inv = (J2 * N1).completeOrthogonalDecomposition().pseudoInverse();
    // VectorXd q2_star = J2_inv * (x2 - J2 * q1_star);

    // cout << "Pinv Solution: " << (q1_star + q2_star).transpose() << endl;
}

	
int main(int argc, char **argv){
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
