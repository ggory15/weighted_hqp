#include <iostream>
#include <cstdio>
#include <Eigen/Core>
#include "HQP_Hcod/Random.hpp"
#include "HQP_Hcod/InitSet.hpp"


#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

using namespace hcod;
using namespace Eigen;
using namespace std;

BOOST_AUTO_TEST_SUITE ( BOOST_TEST_MODULE )

BOOST_AUTO_TEST_CASE ( test_random )
{
    RandStackWithWeight RandStack(10, 2, Vector2i(4, 3), Vector2i(4, 3));
    Initset Init_active(RandStack.getbtype());
    
    for (int i=0; i<2; i++)
        cout << "A" << i << ": " << Init_active.getactiveset()[i].transpose() << endl;
    for (int i=0; i<2; i++)
        cout << "b" << i << ": " << Init_active.getbounds()[i].transpose() << endl; 
}

	
BOOST_AUTO_TEST_SUITE_END ()
