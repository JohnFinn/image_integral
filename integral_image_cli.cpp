#include <boost/program_options.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <numeric>
#include <iostream>

namespace po = boost::program_options;


struct Config {
    int num_threads = 0;
    std::vector<std::string> filenames;
};

/**
* Like main, but with arguments already parsed
*/
void make_integral_images(Config&);

/**
* Usage:
* ./integral_image [-i|--image] <path_to_image1> [[-i|--image] <path_to_image2> […]] [[-t|--threads] <threads number>]
*     аргумент -t может быть равен 0, в этом случае необходимо автоматически выбрать количество потоков, исходя из возможностей процессора;
*     аргумент -t может отстутствовать, в этом случае его считать равным 0;
*     при указании некорректного количества потоков приложение должно ничего не сделать, вывести сообщение об ошибке и корректно завершиться;
*     при указании некорректного пути, например, path_to_image2, оно не должно обрабатываться, должно быть выведено сообщение об ошибке с этим изображением, при этом результат должен быть посчитан для всех изображений с корректными путями;
*     интегральное изображение для изображения path_to_image2 стоит записать в текстовый файл path_to_image2.integral в следующем формате: интегральное изображение для первого канала, пустая строка, интегральное изображение для второго канала, если оно есть и пустая строка и т.д. (для всех каналов);
*/
int main(int argc, char** argv)
{
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
    for (size_t r = 1; r < m.rows; ++r) {
        for (size_t c = 1; c < m.cols; ++c) {
            res(r, c) = res(r, c-1) - res(r-1, c-1) + res(r-1, c);
        }
    }
    return res;
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
