#include <iostream>
#include <cstdio>
#include <Eigen/Core>
#include "weighted_hqp/Random.hpp"
#include "weighted_hqp/iHQP_solver.hpp"
#include "weighted_hqp/InitSet.hpp"

#include <gtest/gtest.h>
#include <chrono>
#include <qpOASES.hpp>

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

    int n_size = 6;
    RandStackWithWeight RandStack(n_size, 2, Vector3i(2, 2, 2), Vector3i(2, 2, 2), false);

    A = RandStack.getA();
    b = RandStack.getb();
    btype = RandStack.getbtype();

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

    

    std::vector<H_structure> h = iHQP_.geth();
    int r1 = h[0].mmax;
    int eq1 = 0;
    int ineq1 = 0;

    VectorXd eq1_idx, ineq1_up_idx, ineq1_lo_idx;
    eq1_idx.setZero(0);
    ineq1_up_idx.setZero(0);
    ineq1_lo_idx.setZero(0);

    typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> MatrixXd;
    MatrixXd H1;
    H1.setIdentity(n_size, n_size);
    H1 = H1 * 1000.;
    MatrixXd A_qp;
    VectorXd b_up, b_low;
    
    for (int i=0; i<r1; i++){
        if (btype[0](i) == 1){
            eq1++;
            A_qp.conservativeResize(eq1 + ineq1, A[0].cols());
            A_qp.row(eq1 + ineq1 - 1) = A[0].row(i);
            b_up.conservativeResize(eq1 + ineq1);
            b_low.conservativeResize(eq1 + ineq1);
            b_up(eq1 + ineq1 - 1) = b[0](i,0);
            b_low(eq1 + ineq1 - 1) = b[0](i,0);

        }
        else if (btype[0](i) == 3){
            ineq1++;
            A_qp.conservativeResize(eq1 + ineq1, A[0].cols());
            A_qp.row(eq1 + ineq1 - 1) = -A[0].row(i);
            b_up.conservativeResize(eq1 + ineq1);
            b_low.conservativeResize(eq1 + ineq1);
            b_up(eq1 + ineq1 - 1) = -b[0](i,0);
            b_low(eq1 + ineq1 - 1) =-10000.0;
        }
        else if (btype[0](i) == 4){
            ineq1++;
            A_qp.conservativeResize(eq1 + ineq1, A[0].cols());
            A_qp.row(eq1 + ineq1 - 1) = A[0].row(i);
            b_up.conservativeResize(eq1 + ineq1);
            b_low.conservativeResize(eq1 + ineq1);
            b_up(eq1 + ineq1 - 1) = b[0](i,1);
            b_low(eq1 + ineq1 - 1) =-10000.0;
        }
        else{
            ineq1 += 1;
            A_qp.conservativeResize(eq1 + ineq1, A[0].cols());
            A_qp.row(eq1 + ineq1 - 1) = A[0].row(i);
            b_up.conservativeResize(eq1 + ineq1);
            b_low.conservativeResize(eq1 + ineq1);
            b_up(eq1 + ineq1 - 1) = b[0](i,1);
            b_low(eq1 + ineq1 - 1) = b[0](i,0);

        }
    }
    VectorXd f1(n_size), f2(n_size);
    f1.setZero();
    f2.setZero();

    using namespace qpOASES;
    SQProblem m_solver, m_solver2;
    Options m_options;
    returnValue m_status, m_status2;
    m_solver = SQProblem(n_size, eq1 + ineq1);
    m_options.setToDefault();
    m_options.initialStatusBounds = ST_INACTIVE;
    m_options.printLevel          = PL_HIGH; //PL_LOW
    m_options.enableRegularisation = BT_TRUE;
    m_options.enableEqualities = BT_TRUE;
    m_solver.setOptions(m_options);
    int iter = 1000;

    m_status = m_solver.init(H1.data(), f1.data(), A_qp.data(), 0, 0, b_low.data(), b_up.data(), iter);
    VectorXd x1_opt(n_size);
    m_solver.getPrimalSolution(x1_opt.data());
    cout << "x1_opt" << x1_opt.transpose() << endl;

    MatrixXd H2;
    H2.setIdentity(n_size, n_size);
    H2 = H2 * 1000.;
    MatrixXd A2_qp;
    VectorXd b2_up, b2_low;
    int r2 = h[1].mmax;
    int eq2 = 0, ineq2=0;

    for (int i=0; i<r2; i++){
        if (btype[1](i) == 1){
            eq2++;
            A2_qp.conservativeResize(eq2 + ineq2, A[1].cols());
            A2_qp.row(eq2 + ineq2 - 1) = A[1].row(i);
            b2_up.conservativeResize(eq2 + ineq2);
            b2_low.conservativeResize(eq2 + ineq2);
            b2_up(eq2 + ineq2 - 1) = b[1](i,0) - (A[1].row(i) * x1_opt)(0,0);
            b2_low(eq2 + ineq2 - 1) = b[1](i,0)- (A[1].row(i) * x1_opt)(0,0);

        }
        else if (btype[1](i) == 3){
            ineq2++;
            A2_qp.conservativeResize(eq2 + ineq2, A[1].cols());
            A2_qp.row(eq2 + ineq2 - 1) = -A[1].row(i);
            b2_up.conservativeResize(eq2 + ineq2);
            b2_low.conservativeResize(eq2 + ineq2);
            b2_up(eq2 + ineq2 - 1) = -b[1](i,0) + (A[1].row(i) * x1_opt)(0,0);
            b2_low(eq2 + ineq2 - 1) =-10000.0;
        }
        else if (btype[1](i) == 4){
            ineq2++;
            A2_qp.conservativeResize(eq2 + ineq2, A[1].cols());
            A2_qp.row(eq2 + ineq2 - 1) = A[1].row(i);
            b2_up.conservativeResize(eq2 + ineq2);
            b2_low.conservativeResize(eq2 + ineq2);
            b2_up(eq2 + ineq2 - 1) = b[1](i,1) - (A[1].row(i) * x1_opt)(0,0);
            b2_low(eq2 + ineq2 - 1) =-10000.0;
        }
        else{
            ineq2 += 1;
            A2_qp.conservativeResize(eq2 + ineq2, A[1].cols());
            A2_qp.row(eq2 + ineq2 - 1) = A[1].row(i);
            b2_up.conservativeResize(eq2 + ineq2);
            b2_low.conservativeResize(eq2 + ineq2);
            b2_up(eq2 + ineq2 - 1) = b[1](i,1) - (A[1].row(i) * x1_opt)(0,0);
            b2_low(eq2 + ineq2 - 1) = b[1](i,0) - (A[1].row(i) * x1_opt)(0,0);
        }
    }
    eq1 = eq2;
    ineq1 = ineq2;
    for (int i=0; i<r1; i++){
        if (btype[0](i) == 1){
            eq1++;
            A2_qp.conservativeResize(eq1 + ineq1, A[0].cols());
            A2_qp.row(eq1 + ineq1 - 1) = A[0].row(i);
            b2_up.conservativeResize(eq1 + ineq1);
            b2_low.conservativeResize(eq1 + ineq1);
            b2_up(eq1 + ineq1 - 1) = 0.0;
            b2_low(eq1 + ineq1 - 1) = 0.0;
        }
        else if (btype[0](i) == 3){
            ineq1++;
            A2_qp.conservativeResize(eq1 + ineq1, A[0].cols());
            A2_qp.row(eq1 + ineq1 - 1) = -A[0].row(i);
            b2_up.conservativeResize(eq1 + ineq1);
            b2_low.conservativeResize(eq1 + ineq1);
            b2_up(eq1 + ineq1 - 1) = -b[0](i,0) +(A[1].row(i) * x1_opt)(0,0) ;
            b2_low(eq1 + ineq1 - 1) =-10000.0;
        }
        else if (btype[0](i) == 4){
            ineq1++;
            A2_qp.conservativeResize(eq1 + ineq1, A[0].cols());
            A2_qp.row(eq1 + ineq1 - 1) = A[0].row(i);
            b2_up.conservativeResize(eq1 + ineq1);
            b2_low.conservativeResize(eq1 + ineq1);
            b2_up(eq1 + ineq1 - 1) = b[0](i,1)- (A[1].row(i) * x1_opt)(0,0);
            b2_low(eq1 + ineq1 - 1) =-10000.0;
        }
        else{
            ineq1 += 1;
            A2_qp.conservativeResize(eq1 + ineq1, A[0].cols());
            A2_qp.row(eq1 + ineq1 - 1) = A[0].row(i);
            b2_up.conservativeResize(eq1 + ineq1);
            b2_low.conservativeResize(eq1 + ineq1);
            b2_up(eq1 + ineq1 - 1) = b[0](i,1)+ (A[1].row(i) * x1_opt)(0,0);
            b2_low(eq1 + ineq1 - 1) = b[0](i,0)- (A[1].row(i) * x1_opt)(0,0);
        }
    }

    m_solver2 = SQProblem(n_size, eq1 + ineq1);
    m_solver2.setOptions(m_options);
    int iter2 = 1000;
    m_status2 = m_solver2.init(H2.data(), f2.data(), A2_qp.data(), 0, 0, b2_low.data(), b2_up.data(), iter2);
    VectorXd x2_opt(n_size);
    m_solver2.getPrimalSolution(x2_opt.data());

    cout << "x1_opt" << x2_opt.transpose() << endl;

    cout << " " << endl;
    cout << "QP Solution: " << (x1_opt + x2_opt).transpose() << endl << endl;



}

	
int main(int argc, char **argv){
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
