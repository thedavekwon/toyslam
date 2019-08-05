//
// Created by dodo on 7/29/19.
//

#ifndef TOYSLAM_MOTIONESTIMATION_H
#define TOYSLAM_MOTIONESTIMATION_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "cameraParamters.h"
#include "bundleAdjustment.h"

const bool DEBUG = false;

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

void poseEstimation2D2D(const std::vector<cv::KeyPoint> &kps1,
                        const std::vector<cv::KeyPoint> &kps2,
                        const std::vector<cv::DMatch> &matches,
                        cv::Mat &R,
                        cv::Mat &t);

void poseEstimation3D2D(const std::vector<cv::KeyPoint> &kps1,
                        const std::vector<cv::KeyPoint> &kps2,
                        const std::vector<cv::DMatch> &matches,
                        const cv::Mat &K,
                        cv::Mat &R,
                        cv::Mat &t);

void triangulation(const std::vector<cv::KeyPoint> &kps1,
                   const std::vector<cv::KeyPoint> &kps2,
                   const std::vector<cv::DMatch> &matches,
                   const cv::Mat &K,
                   const cv::Mat &R,
                   const cv::Mat &t,
                   std::vector<cv::Point3f> &points_3d,
                   std::vector<cv::Point3f> &points_2d);

void poseEstimation3D2DwithTriangulation(const std::vector<cv::KeyPoint> &kps1,
                                         const std::vector<cv::KeyPoint> &kps2,
                                         const std::vector<cv::DMatch> &matches,
                                         cv::Mat &R,
                                         cv::Mat &t);

#endif //TOYSLAM_MOTIONESTIMATION_H
