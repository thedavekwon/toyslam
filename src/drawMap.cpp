//
// Created by Do Hyung Kwon on 8/14/19.
//

#include "drawMap.h"

void draw2D(const cv::Mat &poseT, cv::Mat &trajectory,
            const std::vector<std::string> &cur) {
    auto truePose = loadTruePose(cur[1]);
    int x = int(poseT.at<double>(0)) + 300;
    int y = int(poseT.at<double>(2)) + 100;
    cv::circle(trajectory, cv::Point2f(x, y), 1, CV_RGB(255, 0, 0), 2);
    cv::circle(trajectory, truePose, 1, CV_RGB(0, 0, 255), 2);
    imshow("Trajectory", trajectory);
}

void draw3D(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud, const cv::Mat &poseR, const cv::Mat &poseT,
            const cv::Mat &color, const cv::Mat &depth, const cv::Mat &K) {
    Eigen::Matrix3d R_mat;
    R_mat <<
          poseR.at<double>(0, 0), poseR.at<double>(0, 1), poseR.at<double>(0, 2),
          poseR.at<double>(1, 0), poseR.at<double>(1, 1), poseR.at<double>(1, 2),
          poseR.at<double>(2, 0), poseR.at<double>(2, 1), poseR.at<double>(2, 2);
    Eigen::Vector3d T_vec;
    T_vec << poseT.at<double>(0), poseT.at<double>(1), poseT.at<double>(2);

    Eigen::Quaterniond q(R_mat);
    Eigen::Isometry3d T(q);
    T.pretranslate(T_vec);

    double depthScale = 500.0;

    for (int i = 0; i < color.rows; i++) {
        for (int j = 0; j < color.cols; j++) {
            unsigned int d = depth.at<unsigned short>(i, j);
            if (d == 0) continue;
            Eigen::Vector3d point;
            point[2] = double(d)/depthScale;
            point[0] = (i-K.at<double>(0, 2))*point[2]/K.at<double>(0, 0);
            point[1] = (j-K.at<double>(1, 2))*point[2]/K.at<double>(1, 1);
            Eigen::Vector3d worldPoint = T*point;

            pcl::PointXYZRGB p;
            p.x = worldPoint[0];
            p.y = worldPoint[1];
            p.z = worldPoint[2];
            p.b = color.data[j*color.step+i*color.channels()];
            p.g = color.data[j*color.step+i*color.channels()+1];
            p.r = color.data[j*color.step+i*color.channels()+2];
            pointcloud->points.push_back(p);
        }
    }
}

void draw2Don3D(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud, const cv::Mat &poseR, const cv::Mat &poseT) {
    Eigen::Matrix3d R_mat;
    R_mat <<
          poseR.at<double>(0, 0), poseR.at<double>(0, 1), poseR.at<double>(0, 2),
            poseR.at<double>(1, 0), poseR.at<double>(1, 1), poseR.at<double>(1, 2),
            poseR.at<double>(2, 0), poseR.at<double>(2, 1), poseR.at<double>(2, 2);
    Eigen::Vector3d T_vec;
    T_vec << poseT.at<double>(0), poseT.at<double>(1), poseT.at<double>(2);

    Eigen::Quaterniond q(R_mat);
    Eigen::Isometry3d T(q);
    T.pretranslate(T_vec);

    pcl::PointXYZRGB p;
    p.x = T_vec[0];
    p.y = 0;
    p.z = T_vec[2];
    p.b = 255;
    p.g = 255;
    p.r = 255;
    pointcloud->points.push_back(p);
}

void voxelFiltering(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud,
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_pointcloud) {
    std::cout << "Before Filtering Point Cloud Size: " << pointcloud->size() << std::endl;
    pcl::VoxelGrid<pcl::PointXYZRGB> filter;
    filter.setInputCloud(pointcloud);
    filter.setLeafSize(1, 1, 1);
    filter.filter(*filtered_pointcloud);
    std::cout << "After Filtering Point Cloud Size: " << filtered_pointcloud->size() << std::endl;
}