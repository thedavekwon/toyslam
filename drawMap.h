//
// Created by Do Hyung Kwon on 8/14/19.
//

#ifndef TOYSLAM_DRAWMAP_H
#define TOYSLAM_DRAWMAP_H

#include <fstream>
#include <iostream>
#include <string>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "loadData.h"

void draw2D(const cv::Mat &poseT, cv::Mat &trajectory, const std::vector<std::string> &cur);

void draw3D(const cv::Mat &poseT);

#endif //TOYSLAM_DRAWMAP_H
