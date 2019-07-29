//
// Created by Do Hyung Kwon on 7/29/19.
//

#include "motionEstimation.h"

// K: Calibration Matrix
cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
    return cv::Point2d(
            (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
            (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}

void poseEstimation2D2D(std::vector<cv::KeyPoint> kps1,
                          std::vector<cv::KeyPoint> kps2,
                          std::vector<cv::DMatch> matches,
                          cv::Mat &R,
                          cv::Mat &t) {
    cv::Mat K = loadCalibrationMatrix(0);

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (auto &match : matches) {
        points1.push_back(kps1[match.queryIdx].pt);
        points2.push_back(kps2[match.trainIdx].pt);
    }

    cv::Mat fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC);
    if (DEBUG) std::cout << "fundamental matrix is " << std::endl << fundamental_matrix << std::endl;

    cv::Point2d principal_point = loadPrincipalPoint(0);
    double focal_length = loadFocalLength(0);

    cv::Mat essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    if (DEBUG) std::cout << "essential_matrix is " << std::endl << essential_matrix << std::endl;

    cv::Mat homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    if (DEBUG) std::cout << "homography_matrix is " << std::endl << homography_matrix << std::endl;

    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    if (DEBUG) std::cout << "R is " << std::endl << R << std::endl;
    if (DEBUG) std::cout << "t is " << std::endl << t << std::endl;
}