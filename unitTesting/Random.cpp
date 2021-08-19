#include <iostream>
#include <cstdio>
#include <Eigen/Core>
#include "HQP_Hcod/Random.hpp"
#include <Eigen/QR>    

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

using namespace hcod;
using namespace Eigen;
using namespace std;

BOOST_AUTO_TEST_SUITE ( BOOST_TEST_MODULE )

BOOST_AUTO_TEST_CASE ( test_random2 )
{
    int n_size = 10;
    RandStackWithWeight RandStack(n_size, 2, Vector2i(3, 3), Vector2i(3, 3), false);
    for (int i=0; i<2; i++)
        cout << "b" << i << ": " <<  RandStack.getb()[i] << endl;

    for (int i=0; i<2; i++)
        cout << "btype" << i << ": " <<  RandStack.getbtype()[i] << endl;
    
    
    for (int i=0; i<2; i++)
        cout << "A" << i << ": " <<  RandStack.getA()[i] << endl;
    
    cout << "Au" << RandStack.getAu() << endl;
    cout << "bu" << RandStack.getbu() << endl;
    
    for (int i=0; i<2; i++)
        cout << "W" << RandStack.getW()[i] << endl;

    // cout << "  " << endl;
    // cout << "Checking for Pinv" << endl;

    // MatrixXd J1 = RandStack.getA()[0];
    // MatrixXd J2 = RandStack.getA()[1];
    // VectorXd x1 = RandStack.getb()[0];
    // VectorXd x2 = RandStack.getb()[1];
    // MatrixXd J1_inv = J1.completeOrthogonalDecomposition().pseudoInverse();
    // MatrixXd N1 = MatrixXd::Identity(10, 10) - J1_inv * J1;
    // VectorXd q1_star = J1_inv* x1;
    // MatrixXd J2_inv = (J2 * N1).completeOrthogonalDecomposition().pseudoInverse();
    // VectorXd q2_star = J2_inv * (x2 - J2 * q1_star);

    // cout << "Solution is " << (q1_star + q2_star).transpose() << endl;


}

	
BOOST_AUTO_TEST_SUITE_END ()
