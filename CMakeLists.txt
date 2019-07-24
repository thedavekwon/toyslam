cmake_minimum_required(VERSION 3.14)
project(toyslam)

set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(SDL2_INCLUDE_DIR /usr/include/SDL2)
set(SDL2_LIBRARY /usr/lib/libSDL2.so)

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(G2O REQUIRED)

add_executable(toyslam main.cpp)
#add_executable(toyslam main.cpp bundleAdjustment.cpp bundleAdjustment.h)

target_link_libraries(toyslam ${SDL2_LIBRARIES} ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${G2O_LIBRARIES})