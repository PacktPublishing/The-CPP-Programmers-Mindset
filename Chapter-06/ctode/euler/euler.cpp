#include "euler.h"

void ct::ode::euler_solver(std::vector<double> &solution, std::vector<double> &step_params, const ODEFunction &func,
                           std::span<const double> initial_condition, double initial_param, double final_param,
                           double step_size) {
}