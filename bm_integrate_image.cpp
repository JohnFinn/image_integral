#include <benchmark/benchmark.h>
#include "integrate_image.hpp"

static void BM_integrate(benchmark::State& state)
{
    cv::Mat3d m(1000, 1000);
    cv::randu(m, -1000.0, 1000.0);
    for (auto _ : state) {
        integrate(m);
    }
}

template<unsigned Threads>
static void BM_integrate_inplace(benchmark::State& state)
{
    cv::Mat3d m(1000, 1000);
    for (auto _ : state) {
        state.PauseTiming();
        cv::randu(m, -1000, 1000);
        state.ResumeTiming();
        integrate_inplace(m, Threads);
    }
}

static void BM_opencv_integrate(benchmark::State& state)
{
    cv::Mat3d m(1000, 1000);
    cv::randu(m, -1000.0, 1000.0);
    for (auto _ : state) {
        cv::Mat res;
        cv::integral(m, res);
    }
}

template<unsigned Threads>
static void BM_row_partial_sums(benchmark::State& state)
{
    cv::Mat3d m(1000, 1000);
    cv::randu(m, -1000.0, 1000.0);
    for (auto _ : state) {
        cv::Mat res;
        row_partial_sums(m, Threads);
    }
}

template<unsigned Threads>
static void BM_col_partial_sums(benchmark::State& state)
{
    cv::Mat3d m(1000, 1000);
    cv::randu(m, -1000.0, 1000.0);
    for (auto _ : state) {
        cv::Mat res;
        col_partial_sums(m, Threads);
    }
}


BENCHMARK_TEMPLATE1(BM_integrate_inplace, 1);
BENCHMARK_TEMPLATE1(BM_integrate_inplace, 2);
BENCHMARK_TEMPLATE1(BM_integrate_inplace, 3);
BENCHMARK(BM_integrate);
BENCHMARK(BM_opencv_integrate);
BENCHMARK_TEMPLATE1(BM_row_partial_sums, 1);
BENCHMARK_TEMPLATE1(BM_row_partial_sums, 2);
BENCHMARK_TEMPLATE1(BM_row_partial_sums, 3);
BENCHMARK_TEMPLATE1(BM_col_partial_sums, 1);
BENCHMARK_TEMPLATE1(BM_col_partial_sums, 2);
BENCHMARK_TEMPLATE1(BM_col_partial_sums, 3);

BENCHMARK_MAIN();