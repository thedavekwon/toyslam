//
// Created by dodo on 7/29/19.
//

#ifndef TOYSLAM_CAMERAPARAMTERS_H
#define TOYSLAM_CAMERAPARAMTERS_H

#include <cmath>
#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>

cv::Mat loadCalibrationMatrix(int type);

cv::Point2d loadPrincipalPoint(int type);

double loadFocalLength(int type);

double loadScale(int frameId, int type);

#endif //TOYSLAM_CAMERAPARAMTERS_H
