//
// Created by sammorley on 04/01/25.
//

#ifndef CT_MIN_DISTANCE_OPENMP_H
#define CT_MIN_DISTANCE_OPENMP_H

#include <cmath>
#include <span>

namespace ct {

struct Point2D {
    float x;
    float y;
};

inline float distance(const Point2D &p1, const Point2D &p2) {
    return std::hypot(p1.x - p2.x, p1.y - p2.y);
}

float nearest_neighbor_distance_seq(std::span<const Point2D> points);
float nearest_neighbor_distance_omp(std::span<const Point2D> points);


}



#endif //CT_MIN_DISTANCE_OPENMP_H
