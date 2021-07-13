#pragma once

#include <numeric>
#include <boost/iterator/transform_iterator.hpp>
#include <opencv2/opencv.hpp>


/** \brief integrates image
 */
template<class T, int Channels>
cv::Mat_<cv::Vec<double, Channels>> integrate(const cv::Mat_<cv::Vec<T, Channels>>& m)
{
    using VecCd = cv::Vec<double, Channels>;
    cv::Mat_<VecCd> res = cv::Mat_<VecCd>::zeros(m.size());
    auto cvt2vecCd = [](const cv::Vec<T, Channels>& vec) { return VecCd(vec); };
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
       When summing res(r, c-1) and res(r-1, c) we get too much since they intersect
       and that intersection is doubled.
       But we know the sum of intersection! It's just res(r-1, c-1) so we can just subtruct it once.
       It is similar to inclusion exclusion principle by the way.
     */
    for (size_t r = 1; r < m.rows; ++r) {
        for (size_t c = 1; c < m.cols; ++c) {
            res(r, c) = VecCd(m(r, c)) + res(r, c-1) - res(r-1, c-1) + res(r-1, c);
        }
    }
    return res;
}
