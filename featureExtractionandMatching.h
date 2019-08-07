//
// Created by Do Hyung Kwon on 7/26/19.
//

#ifndef TOYSLAM_FEATUREEXTRACTIONANDMATCHING_H
#define TOYSLAM_FEATUREEXTRACTIONANDMATCHING_H

#include <iostream>
#include <limits>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "motionEstimation.h"
#include "loadData.h"

const int MAX_FEATURES = 500;
const float RATIO_THRESH = 0.7f;

void sequenceFromKitti();

void sequenceFromVideo();

cv::VideoCapture getVideoCapture(std::string s);

void extractFeature(cv::Mat &inputFrame, cv::Mat &outputFrame, std::vector<cv::KeyPoint> &kps, cv::Mat &des,
                    int featureType);

std::vector<cv::DMatch> get_matches(const cv::Mat &prevDes, const cv::Mat &des);

std::vector<cv::DMatch> get_matches_ORB(const cv::Mat &prevDes, const cv::Mat &des);

#endif //TOYSLAM_FEATUREEXTRACTIONANDMATCHING_H
