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
    for (auto _ : state) {
        state.PauseTiming();
        cv::randu(m, -1000, 1000);
        state.ResumeTiming();
        integrate_inplace(m, state.range(0));
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
        row_partial_sums(m, state.range(0));
    }
}

static void BM_col_partial_sums(benchmark::State& state)
{
    cv::Mat3d m(1000, 1000);
    cv::randu(m, -1000.0, 1000.0);
    for (auto _ : state) {
        cv::Mat res;
        col_partial_sums(m, state.range(0));
    }
}

static void num_threads(benchmark::internal::Benchmark* b) {
    for (size_t i = 0; i < std::thread::hardware_concurrency(); ++i) {
        b->Arg(i+1);
    }
}

BENCHMARK(BM_integrate_inplace)->Apply(num_threads);
BENCHMARK(BM_integrate);
BENCHMARK(BM_opencv_integrate);
BENCHMARK(BM_row_partial_sums)->Apply(num_threads);
BENCHMARK(BM_col_partial_sums)->Apply(num_threads);

BENCHMARK_MAIN();
