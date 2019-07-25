#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "bundleAdjustment.h"

const int MAX_FEATURES = 500;

void extractFeature(cv::Mat &inputFrame, cv::Mat &outputFrame, std::vector<cv::KeyPoint> &kps, int featureType);

int mainLoop();

int main() {
//    return mainLoop();
    BA();
}

int mainLoop() {
    cv::VideoCapture cap("video.mp4");

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream!" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame, kFrame;
        std::vector<cv::KeyPoint> kps;
        cap >> frame;
        if (frame.empty()) break;

        extractFeature(frame, kFrame, kps, 1);

        cv::imshow("Frame", kFrame);
        cv::waitKey(16.66);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

void extractFeature(cv::Mat &frame, cv::Mat &kFrame, std::vector<cv::KeyPoint> &kps, int featureType) {
    cv::Mat des;
    cv::Ptr<cv::Feature2D> fet;
    if (featureType == 1) {
        fet = cv::ORB::create(MAX_FEATURES);
    }
    fet->detectAndCompute(frame, cv::noArray(), kps, des);
    cv::drawKeypoints(frame, kps, kFrame);
}