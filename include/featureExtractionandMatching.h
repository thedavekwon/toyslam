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
#include <opencv2/xfeatures2d.hpp>

#include "motionEstimation.h"
#include "loadData.h"
#include "constants.h"

cv::VideoCapture getVideoCapture(std::string s);

void extractFeature(cv::Mat &frame, cv::Mat &kFrame, std::vector<cv::KeyPoint> &kps, std::vector<cv::Point2f> &points,
                    cv::Mat &des, const int featureType);

std::vector<cv::DMatch> get_matches(const cv::Mat &prevDes, const cv::Mat &des, const int featureType);

std::vector<cv::DMatch> get_matches_ORB(const cv::Mat &prevDes, const cv::Mat &des);

void
featureTrackingWithOpticalFlow(const cv::Mat &prevFrame, const cv::Mat &frame, std::vector<cv::Point2f> &prevKeypoints,
                               std::vector<cv::Point2f> &keyPoints, std::vector<uchar> &status);

std::vector<cv::DMatch> filterMatches(const cv::Mat &des, std::vector<cv::DMatch> &matches);

#endif //TOYSLAM_FEATUREEXTRACTIONANDMATCHING_H
