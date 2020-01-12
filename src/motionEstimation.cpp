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

void poseEstimationOpticalFlow(const std::vector<cv::Point2f> &points1,
                               const std::vector<cv::Point2f> &points2,
                               cv::Mat &mask,
                               cv::Mat &R,
                               cv::Mat &t) {
    cv::Point2d principal_point = loadPrincipalPoint(1);
    double focal_length = loadFocalLength(1);
    cv::Mat essential_matrix = cv::findEssentialMat(points2, points1, focal_length, principal_point,
                                                    cv::RANSAC, 0.999, 1.0, mask);
    if (DEBUG) std::cout << "essential_matrix is " << std::endl << essential_matrix << std::endl;
    if (DEBUG) std::cout << points2.size() << " " << points1.size() << std::endl;
    cv::recoverPose(essential_matrix, points2, points1, R, t, focal_length, principal_point, mask);
    if (DEBUG) std::cout << "R is " << std::endl << R << std::endl;
    if (DEBUG) std::cout << "t is " << std::endl << t << std::endl;
}

void poseEstimation2D2D(const std::vector<cv::KeyPoint> &kps1,
                        const std::vector<cv::KeyPoint> &kps2,
                        const std::vector<cv::DMatch> &matches,
                        cv::Mat &mask,
                        cv::Mat &R,
                        cv::Mat &t) {
//    cv::Mat K = loadCalibrationMatrixKitti();

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (auto &match : matches) {
        points1.push_back(kps1[match.trainIdx].pt);
        points2.push_back(kps2[match.queryIdx].pt);
    }

//    cv::Mat fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC);
//    if (DEBUG) std::cout << "fundamental matrix is " << std::endl << fundamental_matrix << std::endl;

    cv::Point2d principal_point = loadPrincipalPoint(1);
    double focal_length = loadFocalLength(1);

    cv::Mat essential_matrix = cv::findEssentialMat(points2, points1, focal_length, principal_point,
                                                    cv::RANSAC, 0.999, 1.0, mask);
    if (DEBUG) std::cout << "essential_matrix is " << std::endl << essential_matrix << std::endl;
    if (DEBUG) std::cout << "Mask: " << std::endl;
    if (DEBUG) std::cout << mask << std::endl;

//    cv::Mat homography_matrix = cv::findHomography(points2, points1, cv::RANSAC, 3);
//    if (DEBUG) std::cout << "homography_matrix is " << std::endl << homography_matrix << std::endl;
//
//    if (DEBUG) std::cout << "Point1 size: " << points1.size() << "\t" << "Point2 size " << points2.size() << std::endl;
    int inlier_num = cv::recoverPose(essential_matrix, points2, points1, R, t, focal_length, principal_point);

    std::cout << "Inlier Count: " << inlier_num << std::endl;
    if (DEBUG) std::cout << "R is " << std::endl << R << std::endl;
    if (DEBUG) std::cout << "t is " << std::endl << t << std::endl;
}

void poseEstimation3D2D(const std::vector<cv::KeyPoint> &kps1,
                        const std::vector<cv::KeyPoint> &kps2,
                        const std::vector<cv::DMatch> &matches,
                        const cv::Mat &K,
                        cv::Mat &R,
                        cv::Mat &t,
                        std::vector<cv::Point3f> &points_3d,
                        std::vector<cv::Point2f> &points_2d) {
    cv::Mat r;
    cv::solvePnP(points_3d, points_2d, K, cv::Mat(), r, t, false);
    cv::Rodrigues(r, R);
    if (DEBUG) std::cout << "R is " << std::endl << R << std::endl;
    if (DEBUG) std::cout << "t is " << std::endl << t << std::endl;
    bundleAdjustment3d2d(points_3d, points_2d, K, R, t);
}

void triangulation(const std::vector<cv::KeyPoint> &kps1,
                   const std::vector<cv::KeyPoint> &kps2,
                   const std::vector<cv::DMatch> &matches,
                   const cv::Mat &K,
                   const cv::Mat &R,
                   const cv::Mat &t,
                   std::vector<cv::Point3f> &points_3d,
                   std::vector<cv::Point2f> &points_2d) {
    points_3d.clear();
    points_2d.clear();

    cv::Mat T1 = (cv::Mat_<float>(3, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0);
    cv::Mat T2 = (cv::Mat_<float>(3, 4) <<
            R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    std::vector<cv::Point2f> points1, points2;
    for (auto &match : matches) {
        points1.push_back(pixel2cam(kps1[match.queryIdx].pt, K));
        points2.push_back(pixel2cam(kps2[match.trainIdx].pt, K));
        points_2d.push_back(kps1[match.queryIdx].pt);
        //points_2d.push_back(pixel2cam(kps1[match.queryIdx].pt, K));
    }

    cv::Mat triangulated;
    cv::triangulatePoints(T1, T2, points1, points2, triangulated);

    for (int i = 0; i < triangulated.cols; i++) {
        cv::Mat x = triangulated.col(i);
        x /= x.at<float>(3, 0);
        cv::Point3d p(
                x.at<float>(0, 0),
                x.at<float>(1, 0),
                x.at<float>(2, 0)
        );
        points_3d.push_back(p);
    }
}

void poseEstimation3D2DwithTriangulation(const std::vector<cv::KeyPoint> &kps1,
                                         const std::vector<cv::KeyPoint> &kps2,
                                         const std::vector<cv::DMatch> &matches,
                                         cv::Mat &mask,
                                         cv::Mat &R,
                                         cv::Mat &t) {
    cv::Mat K = loadCalibrationMatrixKitti();
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;
    std::cout << kps1.size() << " " << kps2.size() << " " << matches.size() << std::endl;
    poseEstimation2D2D(kps1, kps2, matches, mask, R, t);
    std::cout << "after 2D2D" << std::endl;
    std::cout << R << std::endl;
    std::cout << t << std::endl;
    triangulation(kps1, kps2, matches, K, R, t, points_3d, points_2d);
    poseEstimation3D2D(kps1, kps2, matches, K, R, t, points_3d, points_2d);
    std::cout << "after 3D2D" << std::endl;
    std::cout << R << std::endl;
    std::cout << t << std::endl;
}