#include <boost/program_options.hpp>
#include <thread>
#include <iostream>

namespace po = boost::program_options;

struct Config {
    int num_threads = 0;
    std::vector<std::string> filenames;
};

void make_integral_images(Config&);

int main(int argc, char** argv)
{
    Config conf;
    std::vector<std::string> filenames;
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

void make_integral_images(Config& conf) {
    if (conf.num_threads == 0)
        conf.num_threads = std::thread::hardware_concurrency();

    std::cout << conf.num_threads << " number of threads" << std::endl
              << "making integral images from files:" << std::endl;
    for (const std::string& s : conf.filenames) {
        std::cout << s << std::endl;
    }
}
