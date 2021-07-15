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

   Internally calls typed version of integrate()
   If type is not supported, throws TypeNotSupportedError
 */
cv::Mat integrate(const cv::Mat&);

/**
   @brief integrates image inplace

   Internally calls typed version of integrate_inplace()
   If type is not supported, throws TypeNotSupportedError
 */
void integrate_inplace(cv::Mat&, unsigned threads = 1);


class TypeNotSupportedError : std::exception {
    std::string message;
public:
    TypeNotSupportedError(int);

    virtual const char* what() const noexcept override;
};


/**
   @brief integrates image with known type

   integrate_inplace() turned out to be more effecient and easier in parralelization
   the reason I left this one is to compare performance
 */
template<class VecCd>
inline cv::Mat_<VecCd> integrate(const cv::Mat_<VecCd>& m)
{
    cv::Mat_<VecCd> res = cv::Mat_<VecCd>::zeros(m.size());
    auto cvt2vecCd = [](const auto& vec) { return VecCd(vec); };
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

/**
   @brief calculates partial sums for each row
 */
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
        /*
           Note that iterations in the loop above are independent from each other.
           This makes is easy to parallelize them.
        */
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

/**
   @brief calculates partial sums for each column

   The reason this function is implemented not the same way as
   row_partial_sums (except changing row to col of course)
   is because that benchmark showed that this is more efficent.

   Probably this has something to do with the fact that cache is faster
   when working with continuous memory.
 */
template<class T>
inline void col_partial_sums(cv::Mat_<T>& m)
{
    for (size_t r = 1; r < m.rows; ++r) {
        m.row(r) += m.row(r-1);
    }
}

/**
   @brief multithreaded version of col_partial_sums()
 */
template<class T>
inline void col_partial_sums(cv::Mat_<T>& m, unsigned threads)
{
    if (threads == 0) {
        throw std::logic_error("zero threads won't do any good");
    }
    /*
        each thread gets its range of columns, like in picture below

        +----+----+----+
        | T0 | T1 | T2 |
        |    |    |    |
              ....

        this makes calculations independent from each other, so there is
        no need for syncronization
    */
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

/**
   @brief integrates image inplace
 */
template<class T>
inline void integrate_inplace(cv::Mat_<T>& m, unsigned threads = 1)
{
    /*
        row_partial_sums gives us:

        m(R, C) =  Sum[  m(r, C) ]
                  r <= R

        then col_partial_sums gives us:

        m(R, C) = Sum[    Sum[  m(r, c) ] ]
                c <= C  r <= R

        When opening brackets we get exactly integral. This is why it works.
    */
    row_partial_sums(m, threads);
    col_partial_sums(m, threads);
}