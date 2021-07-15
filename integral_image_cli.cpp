/** @file
    @brief image integrating CLI

    @b Usage
    @code
    ./integral_image [-i|--image] <path_to_image1> [[-i|--image] <path_to_image2> […]] [[-t|--threads] <threads number>]
    @endcode
 */

#include <thread>
#include <fstream>
#include <filesystem>
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
   one channel version of write_channel_by_channe()
 */
template<class T>
void write_channel_by_channel(const cv::Mat_<T>&, std::ostream&);

/**
   outputs matrices by channels
   First goes table for channel 1, then for channel 2 and so on
 */
template<class T, int Channels>
void write_channel_by_channel(const cv::Mat_<cv::Vec<T, Channels>>&, std::ostream&);

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
        cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            std::cerr << "incorrect image: " << filename << ", skipping" << std::endl;
            continue;
        }
        std::string integral_fname = filename + ".integral";
        if (std::filesystem::exists(integral_fname)) {
            std::cerr << integral_fname << " already exists, skipping" << std::endl;
            continue;
        }
        try {
            integrate_inplace(image, conf.num_threads);
        } catch (TypeNotSupportedError ex) {
            std::cerr << "image " << filename << " has unsupported type" << std::endl;
        }

        std::ofstream fout(integral_fname);
        switch (image.channels())
        {
        case 1: write_channel_by_channel(cv::Mat1d(image), fout); break;
        case 2: write_channel_by_channel(cv::Mat2d(image), fout); break;
        case 3: write_channel_by_channel(cv::Mat3d(image), fout); break;
        case 4: write_channel_by_channel(cv::Mat4d(image), fout); break;
        default: break;
        }
    }
    return 0;
}

template<class T>
void write_channel_by_channel(const cv::Mat_<T>& m, std::ostream& out)
{
    for (size_t r = 0; r < m.rows; ++r) {
        auto row = m.row(r);
        auto last = std::prev(row.end());
        std::copy(
            row.begin(),
            last,
            std::ostream_iterator<T>(out, " ")
        );
        out << *last;
        if (r < m.rows - 1) {
            out << '\n';
        }
    }
}

template<class T, int Channels>
void write_channel_by_channel(const cv::Mat_<cv::Vec<T, Channels>>& m, std::ostream& out)
{
    for (int channel = 0; channel < Channels; ++channel) {
        for (size_t r = 0; r < m.rows; ++r) {
            auto row = m.row(r);
            auto last = std::prev(row.end());
            std::transform(
                row.begin(),
                last,
                std::ostream_iterator<T>(out, " "),
                [channel](const cv::Vec<T, Channels>& vec) { return vec[channel]; }
            );
            out << (*last)[channel];
            if (r < m.rows - 1) {
                out << '\n';
            }
        }
        if (channel < Channels - 1) {
            out << "\n\n";
        }
    }
}