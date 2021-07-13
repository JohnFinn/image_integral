/** @file
    @brief image integrating CLI

    @b Usage
    @code
    ./integral_image [-i|--image] <path_to_image1> [[-i|--image] <path_to_image2> […]] [[-t|--threads] <threads number>]
    @endcode
 */

#include <thread>
#include <iostream>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include "integrate_image.hpp"

/**
   @brief parsed conmmand line arguments
 */
struct Config {
    int num_threads = 0;
    std::vector<std::string> filenames;
};

/**
   @brief higher lever entry function
          when command line arguments parsing is handled
 */
int make_integral_images(Config&);


/**
   @brief parses command line arguments into Config
          and calls make_integral_images()
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

    return make_integral_images(conf);
}

int make_integral_images(Config& conf)
{
    if (conf.num_threads < 0) {
        std::cerr << "number of threads must be >= 0" << std::endl;
        return 1;
    }
    if (conf.num_threads == 0) {
        conf.num_threads = std::thread::hardware_concurrency();
    }

    for (const std::string& filename : conf.filenames) {
        cv::Mat image = cv::imread(filename);
        cv::Mat integrated = integrate(image);
    }
    //cv::Mat image = cv::Mat::ones(cv::Size(600, 600), CV_8U) * 2;
    //cv::Mat z = cv::Mat::zeros(image.size(), CV_32F);
    //cv::integral(image, z);
    //cv::Mat a;
    //z.convertTo(a, CV_64F, 0.0000001, 0);
    //cv::imshow("foo", a);
    //cv::waitKey(0);
    //cv::destroyWindow("foo");

    //std::cout << cv::format(image(cv::Range(0, 5), cv::Range(0, 5)), cv::Formatter::FMT_NUMPY) << std::endl;
    //std::cout << cv::format(    z(cv::Range(0, 5), cv::Range(0, 5)), cv::Formatter::FMT_NUMPY) << std::endl;
    //cv::Mat m3f = integrate(image);
    //std::cout << cv::format(m3f(cv::Rect(0,0, 5,5)), cv::Formatter::FMT_NUMPY) << std::endl;
    //m3f.convertTo(m3f, CV_64F, 0.0000001, 0);
    //cv::imshow("foo", m3f);
    //cv::waitKey(0);
    //cv::destroyWindow("foo");
}
