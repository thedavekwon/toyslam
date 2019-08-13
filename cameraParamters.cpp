//
// Created by Do Hyung Kwon on 7/29/19.
//

#include "cameraParamters.h"

cv::Mat loadCalibrationMatrix(int type) {
    if (type == 0) return cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;
    else return (cv::Mat_<double>(3, 3) <<
                7.188560000000e+02, 0, 6.071928000000e+02,
                0, 7.188560000000e+02, 1.852157000000e+02,
                0, 0, 1);
}

cv::Point2d loadPrincipalPoint(int type) {
    if (type == 0) return cv::Point2d(
                360 / 2,
                480 / 2);
    else return cv::Point2d(607.1928, 185.2157);
}

double loadFocalLength(int type) {
    if (type == 0) return 100.0;
    else return 718.8560;
}

double loadScale(const std::string &prevPose, const std::string &curPose) {
    auto prevP = loadPoseXYZ(prevPose);
    auto curP = loadPoseXYZ(curPose);

    double scale = sqrt(pow(curP.x-prevP.x, 2) + pow(curP.y-prevP.y, 2) + pow(curP.z-prevP.z, 2)) ;
    std::cout << scale << std::endl;
    return scale;
}

cv::Point3f loadPoseXYZ(const std::string &pose) {
    float x, y, z;
    std::istringstream parser(pose);
    for (int j = 0; j < 12; j++) {
        parser >> z;
        if (j == 7) y = z;
        if (j == 3) x = z;
    }
    return cv::Point3f(x, y, z);
}

cv::Point2f loadTruePose(const std::string &pose) {
    float x, y;
    std::istringstream parser(pose);
    for (int j = 0; j < 12; j++) {
        parser >> y;
        if (j == 3) x = y;
    }
    return cv::Point2f(x+300, y+100);
}