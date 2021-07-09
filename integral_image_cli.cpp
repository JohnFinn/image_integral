#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
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
    cv::imshow("abcd", res);
    cv::waitKey(0);
}
