#include "gauss_newton/gauss_newton.h"
#include <iostream>
#include <random>
int main()
{
    // true funtion : x ^ 2 + 2 * x + 3;
    Eigen::VectorXd int_x(4);
    int_x << 1.1, 2.2, 3.3, 4.4;

    solver::GaussNewton guass_newton_solver(int_x);

    const double mean = 0;      //均值
    const double stddev = 0.01; //标准差
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, stddev);

    for (int i = 0; i < 100; ++i)
    {
        double x = rand() % 100;
        double y = x * x * x + 2 * x * x + 3 * x + 4 + dist(generator);
        guass_newton_solver.AddData(x, y);
    }

    guass_newton_solver.SolverProblem();
    return 0;
}