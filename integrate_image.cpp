#include "integrate_image.hpp"

TypeNotSupportedError::TypeNotSupportedError(int cv_type)
: message(std::to_string(cv_type) + " type not supported")
{}

const char* TypeNotSupportedError::what() const noexcept
{
    return message.c_str();
}


cv::Mat integrate(const cv::Mat& m)
{
    switch (m.type())
    {
    case CV_8U:  return integrate(cv::Mat1b(m));
    case CV_16U: return integrate(cv::Mat1w(m));
    case CV_16S: return integrate(cv::Mat1s(m));
    case CV_32S: return integrate(cv::Mat1i(m));
    case CV_32F: return integrate(cv::Mat1f(m));
    case CV_64F: return integrate(cv::Mat1d(m));

    case CV_8UC2: return integrate(cv::Mat2b(m));
    case CV_8UC3: return integrate(cv::Mat3b(m));
    case CV_8UC4: return integrate(cv::Mat4b(m));

    case CV_16UC2: return integrate(cv::Mat2w(m));
    case CV_16UC3: return integrate(cv::Mat3w(m));
    case CV_16UC4: return integrate(cv::Mat3w(m));

    case CV_32SC2: return integrate(cv::Mat2i(m));
    case CV_32SC3: return integrate(cv::Mat3i(m));
    case CV_32SC4: return integrate(cv::Mat4i(m));

    case CV_32FC2: return integrate(cv::Mat2f(m));
    case CV_32FC3: return integrate(cv::Mat3f(m));
    case CV_32FC4: return integrate(cv::Mat4f(m));

    case CV_64FC2: return integrate(cv::Mat2d(m));
    case CV_64FC3: return integrate(cv::Mat3d(m));
    case CV_64FC4: return integrate(cv::Mat4d(m));
    default: throw TypeNotSupportedError(m.type());
    }
}

void integrate_inplace(cv::Mat& m)
{
    m.convertTo(m, CV_64FC(m.channels()));
    switch (m.channels())
    {
    case 1: { auto a = cv::Mat1d(m); return integrate_inplace(a); }
    case 2: { auto a = cv::Mat2d(m); return integrate_inplace(a); }
    case 3: { auto a = cv::Mat3d(m); return integrate_inplace(a); }
    case 4: { auto a = cv::Mat4d(m); return integrate_inplace(a); }
    default: throw TypeNotSupportedError(m.type());
    }
}
