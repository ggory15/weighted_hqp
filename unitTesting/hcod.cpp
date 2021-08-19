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
#include <iostream>     // std::cout
#include <algorithm>    // std::set_difference, std::sort
#include <vector>       // std::vector
#include <numeric>
#include <list>

using namespace hcod;
using namespace Eigen;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

BOOST_AUTO_TEST_SUITE ( BOOST_TEST_MODULE )

BOOST_AUTO_TEST_CASE ( test_random )
{   
    int n_size = 10;
    RandStackWithWeight RandStack(n_size, 2, Vector2i(3, 3), Vector2i(3, 3), false);
   // Initset Init_active(RandStack.getbtype());
   // HCod Hcod_t(RandStack.getA(), RandStack.getb(), RandStack.getbtype(),Init_active.getactiveset(), Init_active.getbounds());

    vector<MatrixXd> A;
    vector<MatrixXd> b;
    vector<VectorXi> btype;
    MatrixXd A1(1, 3), A2(2, 3);
    MatrixXd b1(1, 2), b2(2, 2);
    VectorXi btype1(1), btype2(2);

    A2.row(0) << 0.902944243066385,0.872419711644596,-0.342627652981386;
    A2.row(1) << .478939420486154,0.632568327934086,0.665381814372249;
    b1.row(0) << -0.6854, -0.6018;
    b2.row(0) << -0.841395509866952,-0.159142597519700;
    b2.row(1) << -0.024672785579257,0.752290695058151;
    btype1(0) = 4;
    btype2 << 1, 3;

    A.push_back(A1);
    A.push_back(A2);
    b.push_back(b1);
    b.push_back(b2);
    btype.push_back(btype1);
    btype.push_back(btype2);

    Initset Init_active(btype);
    HCod Hcod_t(A, b, btype, Init_active.getactiveset(), Init_active.getbounds());


    cout << "Level 0" << endl;
    Hcod_t.print_h_structure(0);
    cout << " " << endl;
    cout << "Level 1" << endl;
    Hcod_t.print_h_structure(1);
    cout << " " << endl;
    cout << " " << endl; 

    // int first[] = {5,10,15,20,25};
    // int second[] = {50,40,30,20,10};

    // Eigen::VectorXi a, bb, c;
    // a = Eigen::VectorXi::LinSpaced(6 ,0, 5);
    // bb= Eigen::VectorXi::LinSpaced(6 ,2, 7);
    // c = Eigen::VectorXi(a.size());
    
    //  auto it = std::set_difference(a.data(), a.data() + a.size(), 
    //                               bb.data(), bb.data() + bb.size(), 
    //                               c.data());
    //    c.conservativeResize(std::distance(c.data(), it)); // resize the result
    // std::cout << c.transpose() << " " << bb.transpose()  << std::endl;       

    Eigen::VectorXi diff_vec(5);
    diff_vec.setOnes();
    cout << diff_vec.transpose() << endl;
    diff_vec.conservativeResize(std::distance(diff_vec.data(), diff_vec.data()+ 10)); 
    
    diff_vec(6) = 0;
cout << diff_vec.transpose() << endl;
    // diff_vec.size() - std::distance(diff_vec.begin(), found) - 1

std::cout << std::distance(diff_vec.begin(), std::find_if(diff_vec.begin(), diff_vec.end(), [](const auto& x) { return x != 0; }))  << '\n';

    for (int i=0; i<3; i++){
        cout << i << endl;
        break;
    }

    Eigen::VectorXi idx_vec(3);
    idx_vec << 1, 2, 4;
    cout << diff_vec(idx_vec.array()) << endl;
    Eigen::MatrixXd test_mat(6, 6);
    test_mat.setRandom();
    test_mat(idx_vec.array(), idx_vec.array());
    cout << test_mat << endl;
    cout << test_mat(idx_vec, idx_vec) << endl;
    
    // Eigen::VectorXd csum(3), flip_vec;
    // flip_vec.setRandom(3);
    // cout << "fi" << flip_vec.transpose() << endl;
    // flip_vec = flip_vec.reverse();
    // cout << "se" << flip_vec.transpose() << endl;
    // std::partial_sum(flip_vec.begin(), flip_vec.end(), csum.begin(), std::plus<double>());
    // cout << "cu" << csum.transpose() << endl;
 
}

	
BOOST_AUTO_TEST_SUITE_END ()
