#ifndef EULER_H
#define EULER_H

#include <functional>
#include <span>
#include <vector>

namespace ct::ode {
using ODEFunction = std::function<void(std::span<double>, std::span<const double>)>;

void suler_solver(
    std::vector<double> &soluton,
    std::vector<double> &step_params,
    const ODEFunction &func,
    std::span<const double> initial_condition,
    double initial_param,
    double final_param,
    double step_size
);
}


#endif //EULER_H