#include <algorithm>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "integrate_image.hpp"

/**
   @brief Wraps cv::Mat_ giving it operator==() and PrintTo() so gtest can use it.

   cv::Mat_<T>::operator==() does not check equality of objects.
   So in order to use EXPECT_EQ and have it printing objects to stdout better
   we need some kind of a wrapper.
 */
template<class T>
class MatWrapper {
    const cv::Mat_<T>& _mat;
public:
    explicit MatWrapper(const cv::Mat_<T>& mat) : _mat(mat) {}

    bool operator==(const MatWrapper& other) const
    {
        return _mat.size() == other._mat.size() && std::equal(_mat.begin(), _mat.end(), other._mat.begin());
    }

    friend void PrintTo(const MatWrapper& mwrapper, std::ostream* ostream)
    {
        *ostream << std::endl << cv::format(mwrapper._mat, cv::Formatter::FMT_NUMPY);
    }
};

TEST(integration, _3x4_2channels)
{
    cv::Mat2b mat(3, 4);
    mat << cv::Vec2b(1,1), cv::Vec2b(0,0), cv::Vec2b(1,0), cv::Vec2b(3,0),
           cv::Vec2b(0,0), cv::Vec2b(4,0), cv::Vec2b(4,0), cv::Vec2b(1,0),
           cv::Vec2b(6,0), cv::Vec2b(8,0), cv::Vec2b(4,0), cv::Vec2b(0,0);
    cv::Mat2d expected(3, 4);
    expected << cv::Vec2d(1,1), cv::Vec2d( 1,1), cv::Vec2d( 2,1), cv::Vec2d( 5,1),
                cv::Vec2d(1,1), cv::Vec2d( 5,1), cv::Vec2d(10,1), cv::Vec2d(14,1),
                cv::Vec2d(7,1), cv::Vec2d(19,1), cv::Vec2d(28,1), cv::Vec2d(32,1);
    cv::Mat2d res = integrate(mat);
    EXPECT_EQ(MatWrapper(res), MatWrapper(expected));
}

TEST(integration, _3x2_1channel)
{
    cv::Mat1b mat(3,2);
    mat << 0, 1,
           2, 3,
           4, 5;
    cv::Mat1d expected(3,2);
    expected << 0, 1,
                2, 6,
                6, 15;
    cv::Mat1d res = integrate(mat);
    EXPECT_EQ(MatWrapper(res), MatWrapper(expected));
}

TEST(inplace_integration, _3x4_2channels)
{
    cv::Mat mat = (cv::Mat2b(3, 4) <<
        cv::Vec2b(1,1), cv::Vec2b(0,0), cv::Vec2b(1,0), cv::Vec2b(3,0),
        cv::Vec2b(0,0), cv::Vec2b(4,0), cv::Vec2b(4,0), cv::Vec2b(1,0),
        cv::Vec2b(6,0), cv::Vec2b(8,0), cv::Vec2b(4,0), cv::Vec2b(0,0));
    cv::Mat2d expected(3, 4);
    expected << cv::Vec2d(1,1), cv::Vec2d( 1,1), cv::Vec2d( 2,1), cv::Vec2d( 5,1),
                cv::Vec2d(1,1), cv::Vec2d( 5,1), cv::Vec2d(10,1), cv::Vec2d(14,1),
                cv::Vec2d(7,1), cv::Vec2d(19,1), cv::Vec2d(28,1), cv::Vec2d(32,1);
    integrate_inplace(mat);
    EXPECT_EQ(MatWrapper(cv::Mat_<cv::Vec2d>(mat)), MatWrapper(expected));
}

TEST(inplace_integration, _3x2_1channel)
{
    cv::Mat mat = (cv::Mat1b(3, 2) <<
        0, 1,
        2, 3,
        4, 5);
    cv::Mat1d expected(3, 2);
    expected << 0,  1,
                2,  6,
                6, 15;
    integrate_inplace(mat);
    EXPECT_EQ(MatWrapper(cv::Mat_<double>(mat)), MatWrapper(expected));
}

TEST(inplace_integration, _big_ones_threads)
{
    cv::Mat1d mat = cv::Mat1d::ones(1000, 1000);
    cv::Mat1d expected(1000, 1000);
    for (size_t r = 0; r < expected.rows; ++r) {
        for (size_t c = 0; c < expected.cols; ++c) {
            expected(r, c) = (r + 1) * (c + 1);
        }
    }
    integrate_inplace(mat, 9);
    EXPECT_EQ(MatWrapper(cv::Mat_<double>(mat)), MatWrapper(expected));
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}