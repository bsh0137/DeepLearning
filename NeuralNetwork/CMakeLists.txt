cmake_minimum_required(VERSION 2.8)
project(train)
include_directories( include )
find_package(OpenCV REQUIRED)
add_executable(image_test src/train.c src/image.cpp)
target_link_libraries(image_test ${OpenCV_LIBS})

