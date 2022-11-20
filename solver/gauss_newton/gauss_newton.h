#include <eigen3/Eigen/Core>
#include <vector>
#include <iostream>
namespace solver
{
    class GaussNewton
    {
    public:
        GaussNewton(Eigen::VectorXd init_x);
        void AddData(const double &x, const double &y)
        {
            data_pair_.push_back(std::make_pair(x, y));
        };

        Eigen::MatrixXd CalculateJacob();
        void CalculateHessian();
        void UpdateX();
        double GetCost();
        bool SolverProblem();

    private:
        Eigen::MatrixXd jacob_matrix_;
        Eigen::MatrixXd hessian_matrix_;
        Eigen::VectorXd x_;

        Eigen::VectorXd delta_x_;
        Eigen::VectorXd cost_;

        std::vector<std::pair<double, double>> data_pair_;
    };
}