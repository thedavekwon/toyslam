//
// Created by dodo on 7/29/19.
//

#ifndef TOYSLAM_CAMERAPARAMTERS_H
#define TOYSLAM_CAMERAPARAMTERS_H

#include <opencv2/core.hpp>

cv::Mat loadCalibrationMatrix(int type);
cv::Point2d loadPrincipalPoint(int type);
double loadFocalLength(int type);


#endif //TOYSLAM_CAMERAPARAMTERS_H
