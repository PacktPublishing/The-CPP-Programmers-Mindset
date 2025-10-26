
#include <sycl/sycl.hpp>

#include "sycl_saxy.hpp"



void ct::sycl_saxpy(float a, const float *x, float *y, int N) {
    sycl::queue queue;
    sycl::range<1> vec_range(N);

    sycl::buffer x_buf(x, vec_range);
    sycl::buffer y_buf(y, vec_range);

    auto event = queue.submit([&](sycl::handler & handle) {
        sycl::accessor x_access(x_buf, handle, sycl::read_only);
        sycl::accessor y_access(y_buf, handle, sycl::read_write);

        handle.parallel_for(N, [=](auto i) {
            y_access[i] = a * x_access[i] + y_access[i];
        });
    });

    event.wait_and_throw();
}
