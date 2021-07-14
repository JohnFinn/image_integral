#pragma once
/** @file
    @brief image integration functional
 */

#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/asio.hpp>
#include <opencv2/opencv.hpp>

/**
   @brief integrates image with unknown type

   Internally calls typed version if integrate()
   If type is not supported, throws TypeNotSupportedError
 */
cv::Mat integrate(const cv::Mat&);


class TypeNotSupportedError : std::exception {
    std::string message;
public:
    TypeNotSupportedError(int);

    virtual const char* what() const noexcept override;
};


namespace detail {

    template<class T>
    struct channel_double : public std::enable_if<std::is_arithmetic_v<T>, double> {};

    template<class T, int Channels>
    struct channel_double<cv::Vec<T, Channels>> { using type = cv::Vec<double, Channels>; };

    /**
       given channel type returns channel type of a same structure but with doubles
     */
    template<class T>
    using channel_double_t = typename channel_double<T>::type;
}

/** @brief integrates image with known type
 */
template<class T>
inline cv::Mat_<detail::channel_double_t<T>> integrate(const cv::Mat_<T>& m)
{
    using VecCd = detail::channel_double_t<T>;
    cv::Mat_<VecCd> res = cv::Mat_<VecCd>::zeros(m.size());
    auto cvt2vecCd = [](const T& vec) { return VecCd(vec); };
    auto row0 = m.row(0);
    std::partial_sum(
        boost::make_transform_iterator(row0.begin(), cvt2vecCd),
        boost::make_transform_iterator(row0.end(), cvt2vecCd),
        res.row(0).begin()
    );
    /**
       Here we calculating sums based on previously calculated sums.
       When summing `res(r, c-1)` and `res(r-1, c)` we get too much since they intersect
       and that intersection is doubled.
       But we know the sum of intersection! It's just `res(r-1, c-1)` so we can just subtruct it once.
       It is similar to inclusion exclusion principle by the way.
     */
    for (size_t r = 1; r < m.rows; ++r) {
        res(r, 0) = VecCd(m(r, 0)) + res(r-1, 0);
        for (size_t c = 1; c < m.cols; ++c) {
            res(r, c) = VecCd(m(r, c)) + res(r, c-1) - res(r-1, c-1) + res(r-1, c);
        }
    }
    return res;
}

template<int Channels>
struct get_channel_type { using type = cv::Vec<double, Channels>; };

template<>
struct get_channel_type<1> { using type = double; };

template<int Channels>
using get_channel_type_t = typename get_channel_type<Channels>::type;

template<class T>
inline void row_partial_sums(cv::Mat_<T>& m, unsigned threads = 1)
{
    if (threads == 0) {
        throw std::logic_error("zero threads won't do any good");
    }
    if (threads == 1) {
        for (size_t r = 0; r < m.rows; ++r) {
            auto row = m.row(r);
            std::partial_sum(
                row.begin(),
                row.end(),
                row.begin()
            );
        }
    } else {
        boost::asio::thread_pool pool(threads);
        for (size_t r = 0; r < m.rows; ++r) {
            boost::asio::post(pool, [&m,r]() {
                auto row = m.row(r);
                std::partial_sum(
                    row.begin(),
                    row.end(),
                    row.begin()
                );
            });
        }
        pool.join();
    }
}

template<class T>
inline void col_partial_sums(cv::Mat_<T>& m)
{
    for (size_t r = 1; r < m.rows; ++r) {
        m.row(r) += m.row(r-1);
    }
}

template<class T>
inline void col_partial_sums(cv::Mat_<T>& m, unsigned threads)
{
    if (threads == 0) {
        throw std::logic_error("zero threads won't do any good");
    }
    boost::asio::thread_pool pool(threads);
    size_t step = m.cols / threads;
    cv::Range interval(0, step + m.cols % threads);
    cv::Range first = interval;
    for (unsigned t = 1; t < threads; ++t) {
        interval = cv::Range(interval.end, interval.end + step);
        boost::asio::post(pool, [&m,interval]() {
            cv::Mat_<T> columns(m.colRange(interval));
            col_partial_sums(columns);
        });
    }
    cv::Mat_<T> columns(m.colRange(first));
    col_partial_sums(columns);

    pool.join();
}

template<class T>
inline void integrate_inplace(cv::Mat_<T>& m, unsigned threads = 1)
{
    row_partial_sums(m, threads);
    col_partial_sums(m, threads);
}

void integrate_inplace(cv::Mat&);
