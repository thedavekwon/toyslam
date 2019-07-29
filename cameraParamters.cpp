//
// Created by Do Hyung Kwon on 7/29/19.
//

#include "cameraParamters.h"

cv::Mat loadCalibrationMatrix(int type) {
    return cv::Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;
}

cv::Point2d loadPrincipalPoint(int type) {
    return cv::Point2d(
            360/2,
            480/2
            );
}

double loadFocalLength(int type) {
    return 100.0;
}