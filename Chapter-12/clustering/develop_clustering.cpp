#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

using namespace duckies;

static constexpr Coord3 polar_data[] = {
    {0.0f, 0.0f, 1.0f},
    {0.0f, 0.0f, -1.0f},
    {0.0f, 0.099833417f, 0.995004165},
    {0.099833417f, 0.0f, 0.995004165},
    {-0.099833417f, 0.0f, 0.995004165},
    {0.0f, -0.099833417f, 0.995004165},
    {0.0f, 0.099833417f, -0.995004165},
    {0.099833417f, 0.0f, -0.995004165},
    {-0.099833417f, 0.0f, -0.995004165},
    {0.0f, -0.099833417f, -0.995004165},
    {1.0f, 0.0f, 0.0f},
    {0.995004165f, 0.099833417f, 0.0f},
    {0.995004165f, -0.099833417f, 0.0f},
};

int main() {
    spdlog::cfg::load_env_levels();

    // std::span<const Coord3> data(polar_data, 13);

    RubberDuckData data;
    data.reserve_additional(20);
    constexpr float mile100 = 0.89833485f;

    data.insert(90.f, 0.0f, std::chrono::year_month_day{}, "");
    data.insert(-90.f, 0.0f, std::chrono::year_month_day{}, "");
    data.insert(90.f - mile100, 0.0f, std::chrono::year_month_day{}, "");
    data.insert(90.f - mile100, 90.0f, std::chrono::year_month_day{}, "");
    data.insert(90.f - mile100, 180.0f, std::chrono::year_month_day{}, "");
    data.insert(90.f - mile100, -90.0f, std::chrono::year_month_day{}, "");
    data.insert(-90.f + mile100, 0.0f, std::chrono::year_month_day{}, "");
    data.insert(-90.f + mile100, 90.0f, std::chrono::year_month_day{}, "");
    data.insert(-90.f + mile100, 180.0f, std::chrono::year_month_day{}, "");
    data.insert(-90.f + mile100, -90.0f, std::chrono::year_month_day{}, "");

    auto clusters = compute_clusters(data, 2, 10, 5, 100);

    std::cout << "clusters: " << clusters.size() << '\n';
    for (auto &cluster: clusters) {
        std::cout << cluster.position.latitude << ' '
                << cluster.position.longitude << ' ' << cluster.num_points
                << '\n';
    }
    return 0;
}