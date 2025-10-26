#ifndef CHAPTER_14_SAXPY_MT_HPP
#define CHAPTER_14_SAXPY_MT_HPP

namespace ct {

void cpu_saxpy(float a, const float* x, float* y, int N);

void cpu_mt_saxpy(float a, const float* x, float* y, int N);

void openmp_saxpy(float a, const float* x, float* y, int N);



}


#endif //CHAPTER_14_SAXPY_MT_HPP