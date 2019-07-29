//
// Created by Do Hyung Kwon on 7/26/19.
//

#ifndef TOYSLAM_FEATUREEXTRACTIONANDMATCHING_H
#define TOYSLAM_FEATUREEXTRACTIONANDMATCHING_H

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "motionEstimation.h"

const int MAX_FEATURES = 500;
const float RATIO_THRESH = 0.4f;
const bool SHOW = true;

int mainLoop();
cv::VideoCapture getCapture(std::string s);
void extractFeature(cv::Mat &inputFrame, cv::Mat &outputFrame, std::vector<cv::KeyPoint> &kps, cv::Mat &des,
                    int featureType);

#endif //TOYSLAM_FEATUREEXTRACTIONANDMATCHING_H
