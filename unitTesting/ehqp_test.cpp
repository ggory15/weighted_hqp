#include <iostream>
#include <cstdio>
#include <Eigen/Core>
#include "HQP_Hcod/Random.hpp"
#include "HQP_Hcod/eHQP_solver.hpp"
#include "HQP_Hcod/InitSet.hpp"

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
    RandStackWithWeight RandStack(n_size, 2, Vector2i(8, 8), Vector2i(8, 8));
    Initset Init_active(RandStack.getbtype());
    
    eHQP_solver eHQP_(RandStack.getA(), RandStack.getb(), RandStack.getbtype(),Init_active.getactiveset(), Init_active.getbounds());

    // auto t1 = high_resolution_clock::now();
    Eigen::VectorXd x_opt = eHQP_.solve();
    // auto t2 = high_resolution_clock::now();
    // auto ms_int = duration_cast<milliseconds>(t2 - t1);
    // duration<double, std::milli> ms_double = t2 - t1;
    // std::cout << ms_double.count() << "ms" << endl;
    // cout << " " << endl;

    cout << "HCOD Solution: " << x_opt.transpose() << endl;

    MatrixXd J1 = RandStack.getA()[0];
    MatrixXd J2 = RandStack.getA()[1];
    MatrixXd x1 = RandStack.getb()[0];
    MatrixXd x2 = RandStack.getb()[1];
    MatrixXd J1_inv = J1.completeOrthogonalDecomposition().pseudoInverse();
    MatrixXd N1 = MatrixXd::Identity(n_size, n_size) - J1_inv * J1;
    MatrixXd q1_star = J1_inv* x1;
    MatrixXd J2_inv = (J2 * N1).completeOrthogonalDecomposition().pseudoInverse();
    MatrixXd q2_star = J2_inv * (x2 - J2 * q1_star);

    cout << "Pinv Solution: " << (q1_star + q2_star).transpose() << endl;
}

	
BOOST_AUTO_TEST_SUITE_END ()
