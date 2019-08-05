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
    else return cv::Point2d(
                1241 / 2,
                 376 / 2);
}

double loadFocalLength(int type) {
    if (type == 0) return 100.0;
    else return 718.8560;
}