//
// Created by Do Hyung Kwon on 8/14/19.
//

#include "drawMap.h"

void draw2D(const cv::Mat &poseT, cv::Mat &trajectory,
            const std::vector<std::string> &cur) {
    auto truePose = loadTruePose(cur[1]);
    int x = int(poseT.at<double>(0)) + 300;
    int y = int(poseT.at<double>(2)) + 100;
    cv::circle(trajectory, cv::Point2f(x, y), 1, CV_RGB(255, 0, 0), 2);
    cv::circle(trajectory, truePose, 1, CV_RGB(0, 0, 255), 2);
    imshow("Trajectory", trajectory);
}


