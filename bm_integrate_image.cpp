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

static void BM_integrate_inplace(benchmark::State& state)
{
    cv::Mat3d m(1000, 1000);
    cv::randu(m, -1000.0, 1000.0);
    std::vector<cv::Mat3d> vec;
    for (size_t i = 0; i < 200; ++i)
        vec.push_back(m.clone());

    size_t i = 0;
    for (auto _ : state) {
        integrate_inplace(vec[i]);
        ++i;
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

static void BM_row_partial_sums(benchmark::State& state)
{
    cv::Mat3d m(1000, 1000);
    cv::randu(m, -1000.0, 1000.0);
    for (auto _ : state) {
        cv::Mat res;
        row_partial_sums(m);
    }
}

static void BM_col_partial_sums(benchmark::State& state)
{
    cv::Mat3d m(1000, 1000);
    cv::randu(m, -1000.0, 1000.0);
    for (auto _ : state) {
        cv::Mat res;
        row_partial_sums(m);
    }
}

BENCHMARK(BM_integrate);
BENCHMARK(BM_opencv_integrate);
BENCHMARK(BM_row_partial_sums);
BENCHMARK(BM_col_partial_sums);
BENCHMARK(BM_integrate_inplace);

BENCHMARK_MAIN();