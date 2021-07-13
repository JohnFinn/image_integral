/** @file
 *  @brief image integrating CLI
 *
 *  @b Usage
 *  @code
 *  ./integral_image [-i|--image] <path_to_image1> [[-i|--image] <path_to_image2> […]] [[-t|--threads] <threads number>]
 *  @endcode
 */

#include <thread>
#include <iostream>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include "integrate_image.hpp"

/**
 * @brief parsed conmmand line arguments
 */
struct Config {
    int num_threads = 0;
    std::vector<std::string> filenames;
};

/**
 * @brief higher lever entry function
 *        when command line arguments parsing is handled
 */
void make_integral_images(Config&);


/**
 * @brief parses command line arguments into Config
 *        and calls make_integral_images()
 */
int main(int argc, char** argv)
{
    namespace po = boost::program_options;
    Config conf;
    po::options_description desc("my description");
    desc.add_options()
        ("help,h", "produce help message")
        ("image,i", po::value<std::vector<std::string>>(&conf.filenames), "input image")
        ("threads,t", po::value<int>(&conf.num_threads)->default_value(0), "number of threads")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    make_integral_images(conf);
}

void make_integral_images(Config& conf)
{
    if (conf.num_threads == 0)
        conf.num_threads = std::thread::hardware_concurrency();

    std::cout << conf.num_threads << " number of threads" << std::endl
              << "making integral images from files:" << std::endl;
    for (const std::string& s : conf.filenames) {
        std::cout << s << std::endl;
    }
    cv::Mat image = cv::imread(conf.filenames[0]);
    //cv::Mat image = cv::Mat::ones(cv::Size(600, 600), CV_8U) * 2;
    cv::Mat z = cv::Mat::zeros(image.size(), CV_32F);
    cv::integral(image, z);
    cv::Mat a;
    z.convertTo(a, CV_64F, 0.0000001, 0);
    cv::imshow("foo", a);
    cv::waitKey(0);
    cv::destroyWindow("foo");

    std::cout << cv::format(image(cv::Range(0, 5), cv::Range(0, 5)), cv::Formatter::FMT_NUMPY) << std::endl;
    std::cout << cv::format(    z(cv::Range(0, 5), cv::Range(0, 5)), cv::Formatter::FMT_NUMPY) << std::endl;
    std::cout << cv::format(image.row(0)(cv::Range::all(), cv::Range(0, 5)), cv::Formatter::FMT_NUMPY) << std::endl;
    cv::Mat row0 = image.row(0);
    cv::Mat integrated = cv::Mat::zeros(image.size(), CV_32SC3);
    std::partial_sum(
        row0.begin<cv::Vec3b>(),
        row0.end<cv::Vec3b>(),
        integrated.row(0).begin<cv::Vec3i>());
    std::cout << cv::format(integrated(cv::Rect(0,0, 5,5)), cv::Formatter::FMT_NUMPY) << std::endl;
    std::cout << cv::format(cv::Mat3f::zeros(cv::Size(4,4)), cv::Formatter::FMT_NUMPY) << std::endl;
    cv::Mat3b img(image);
    cv::Mat3f m3f = integrate(img);
    std::cout << cv::format(m3f(cv::Rect(0,0, 5,5)), cv::Formatter::FMT_NUMPY) << std::endl;
}
