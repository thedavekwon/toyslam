//
// Created by Do Hyung Kwon on 8/14/19.
//

#ifndef TOYSLAM_DRAWMAP_H
#define TOYSLAM_DRAWMAP_H

#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>

#include "loadData.h"

void draw2D(const cv::Mat &poseT, cv::Mat &trajectory, const std::vector<std::string> &cur);

void draw3D(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud, const cv::Mat &poseR, const cv::Mat &poseT,
            const cv::Mat &color, const cv::Mat &depth, const cv::Mat &K);

void draw2Don3D(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud, const cv::Mat &poseR, const cv::Mat &poseT);

void voxelFiltering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud,
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_pointcloud);

#endif //TOYSLAM_DRAWMAP_H
