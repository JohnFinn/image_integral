#pragma once
/** @file
    @brief image integration functional
 */

#include <numeric>
#include <type_traits>
#include <boost/iterator/transform_iterator.hpp>
#include <opencv2/opencv.hpp>

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

/** @brief integrates image
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
    auto col0 = m.col(0);
    std::partial_sum(
        boost::make_transform_iterator(col0.begin(), cvt2vecCd),
        boost::make_transform_iterator(col0.end(), cvt2vecCd),
        res.col(0).begin()
    );
    /**
       Here we calculating sums based on previously calculated sums.
       When summing `res(r, c-1)` and `res(r-1, c)` we get too much since they intersect
       and that intersection is doubled.
       But we know the sum of intersection! It's just `res(r-1, c-1)` so we can just subtruct it once.
       It is similar to inclusion exclusion principle by the way.
     */
    for (size_t r = 1; r < m.rows; ++r) {
        for (size_t c = 1; c < m.cols; ++c) {
            res(r, c) = VecCd(m(r, c)) + res(r, c-1) - res(r-1, c-1) + res(r-1, c);
        }
    }
    return res;
}
