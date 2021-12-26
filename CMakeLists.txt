cmake_minimum_required(VERSION 3.20)

project(integral_image)

find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(Threads REQUIRED)
find_package(benchmark REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(integrate_image integrate_image.cpp)

add_executable(integral_image integral_image_cli.cpp)
target_link_libraries(integral_image integrate_image Boost::program_options Threads::Threads ${OpenCV_LIBS})

add_executable(test_integral_image test_integral_image.cpp)
target_link_libraries(test_integral_image integrate_image GTest::GTest Threads::Threads ${OpenCV_LIBS})

add_executable(benchmark_integrate_image bm_integrate_image.cpp)
target_link_libraries(benchmark_integrate_image integrate_image Threads::Threads benchmark::benchmark ${OpenCV_LIBS})
