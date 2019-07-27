//
// Created by Do Hyung Kwon on 7/26/19.
//

#ifndef TOYSLAM_FEATUREEXTRACTIONANDMATCHING_H
#define TOYSLAM_FEATUREEXTRACTIONANDMATCHING_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

const int MAX_FEATURES = 500;
const float RATIO_THRESH = 0.9f;

int mainLoop();
void extractFeature(cv::Mat &inputFrame, cv::Mat &outputFrame, std::vector<cv::KeyPoint> &kps, cv::Mat &des, int featureType);

#endif //TOYSLAM_FEATUREEXTRACTIONANDMATCHING_H
