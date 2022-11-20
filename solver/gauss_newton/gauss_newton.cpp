#include <eigen3/Eigen/Core>
#include "gauss_newton.h"
#include <eigen3/Eigen/Cholesky>
#include <iostream>
constexpr int kMaxIterNum = 10;
constexpr int kMinDeltaValue = 1e-10;

namespace solver
{

    GaussNewton::GaussNewton(Eigen::VectorXd init_x) : x_(init_x)
    {
    }

    Eigen::MatrixXd GaussNewton::CalculateJacob()
    {
        jacob_matrix_.resize(data_pair_.size(), x_.size());
        cost_.resize(data_pair_.size());

        for (int i = 0; i < data_pair_.size(); ++i)
        {

            const auto &data = data_pair_.at(i);

            cost_(i) = data.second;

            for (int j = 0; j < x_.size(); ++j)
            {

                jacob_matrix_(i, j) = -std::pow(data.first, x_.size() - 1 - j);

                cost_(i) -= x_(j) * std::pow(data.first, x_.size() - 1 - j);
            }
        }
        return jacob_matrix_;
    }

    void GaussNewton::CalculateHessian()
    {
        hessian_matrix_ = jacob_matrix_.transpose() * jacob_matrix_;
    }

    void GaussNewton::UpdateX()
    {
        delta_x_ = hessian_matrix_.ldlt().solve(-jacob_matrix_.transpose() * cost_);
        std::cout << "delta_x_:" << delta_x_.transpose() << std::endl;
        x_ += delta_x_;
    }

    double GaussNewton::GetCost()
    {
        return cost_.transpose() * cost_;
    }

    bool GaussNewton::SolverProblem()
    {
        for (int i = 0; i < kMaxIterNum; ++i)
        {
            CalculateJacob();

            CalculateHessian();

            UpdateX();
            if (delta_x_.norm() < kMinDeltaValue)
            {
                return true;
            }
        }
        std::cout << x_.transpose() << " " << delta_x_.transpose() << " " << GaussNewton::GetCost() << std::endl;

        return false;
    }
}