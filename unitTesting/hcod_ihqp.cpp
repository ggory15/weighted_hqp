#include <iostream>
#include <cstdio>
#include <Eigen/Core>
#include "HQP_Hcod/Random.hpp"
#include "HQP_Hcod/InitSet.hpp"
#include "HQP_Hcod/HCod.hpp"
#include "HQP_Hcod/ehqp_primal.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>
#include <chrono>


using namespace hcod;
using namespace Eigen;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

//#define DEBUG

BOOST_AUTO_TEST_SUITE ( BOOST_TEST_MODULE )

BOOST_AUTO_TEST_CASE ( test_random )
{   
    int n_size = 20;
    RandStackWithWeight RandStack(n_size, 2, Vector2i(3, 3), Vector2i(3, 3));
    Initset Init_active(RandStack.getbtype());
    auto t1 = high_resolution_clock::now();
    HCod Hcod_t(RandStack.getA(), RandStack.getb(), RandStack.getbtype(),Init_active.getactiveset(), Init_active.getbounds());
    cout << "d" << RandStack.getbtype()[0].transpose() << RandStack.getbtype()[1].transpose() << endl;

#ifdef DEBUG
    vector<MatrixXd> A;
    vector<VectorXd> b;
    vector<VectorXi> btype;
    MatrixXd A1(2, 3), A2(1, 3);
    VectorXd b2(1);
    VectorXi btype2(1);
    A1.row(0) << 1.4013, -0.1297, -0.0557;
    A1.row(1) << 0.1266, 1.3881, 0.4328;
    A2.row(0) << 0.2520, -0.2292, 1.0700;
    A.push_back(A1);
    A.push_back(A2);
    b.push_back(Vector2d(0.8356, 0.2442));
    b2(0) = 0.8630;
    b.push_back(b2);
    btype.push_back(Vector2i(1, 1));
    btype2(0) = 1;
    btype.push_back(btype2);
    
    Initset Init_active(btype);
    HCod Hcod_t(A, b, btype,Init_active.getactiveset(), Init_active.getbounds());
#endif   

    VectorXd y0, x0, y1;
    y0.setZero(n_size);
    x0.setZero(n_size);
    y1.setZero(n_size);
    
    double kcheck = 0;
    int iter = 1;

#ifdef DEBUG
    cout << "Level 0" << endl;
    Hcod_t.print_h_structure(0);
    cout << " " << endl;
    cout << "Level 1" << endl;
    Hcod_t.print_h_structure(1);
    cout << " " << endl;
    cout << " " << endl;
#endif

    Ehqp_primal ehqp_primal(Hcod_t.geth(),Hcod_t.getY());
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms" << endl;
    cout << " " << endl;

    cout << "HCOD Solution: " << ehqp_primal.getx().transpose() << endl;
   
    cout << " " << endl;
    
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

	
BOOST_AUTO_TEST_SUITE_END ()
