#include <iostream>
#include <cstdio>
#include <Eigen/Core>
#include "HQP_Hcod/Random.hpp"
#include "HQP_Hcod/InitSet.hpp"
#include "HQP_Hcod/HCod.hpp"
#include "HQP_Hcod/cod.hpp"

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
    //HCod Hcod_t(RandStack.getA(), RandStack.getb(), RandStack.getbtype(),Init_active.getactiveset(), Init_active.getbounds());

    Cod cod_t(RandStack.getA()[0], 1e-8);
    cout << "A" <<  RandStack.getA()[0] << endl;
    cout << "QR_A" << cod_t.getE() * cod_t.getL() * cod_t.getQ().transpose() << endl;

    cout << "W" << cod_t.getW() << endl;
    cout << "E" << cod_t.getE() << endl;
    

}

	
BOOST_AUTO_TEST_SUITE_END ()
