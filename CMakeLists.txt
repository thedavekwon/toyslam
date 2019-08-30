cmake_minimum_required(VERSION 3.14)
project(toyslam)

set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Eigen REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(G2O REQUIRED)
find_package(SDL2 REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(Cholmod REQUIRED)
find_package(CSparse REQUIRED)
find_package(PCL REQUIRED)

LIST(APPEND G2O_LIBRARIES
        cxsparse
        cholmod
        g2o_cli g2o_ext_freeglut_minimal g2o_simulator
        g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
        g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only
        g2o_types_sba g2o_types_slam3d g2o_csparse_extension
        g2o_opengl_helper g2o_solver_dense g2o_stuff
        g2o_types_sclam2d g2o_parser g2o_solver_pcg
        g2o_types_data g2o_types_sim3
        )

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)

add_library(${PROJECT_NAME} SHARED
        src/bundleAdjustment.cpp
        src/featureExtractionandMatching.cpp
        src/motionEstimation.cpp
        src/cameraParamters.cpp
        src/loadData.cpp
        src/sequences.cpp
        src/drawMap.cpp)

add_executable(main src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES} ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${G2O_LIBRARIES}
        ${SuiteSparse_LIBRARIES} ${Cholmod_LIBRARIES} ${Eigen_LIBRARIES} ${CSparse_LIBRARIES} ${PCL_LIBRARIES} )

target_link_libraries(main ${PROJECT_NAME})