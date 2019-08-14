//
// Created by Do Hyung Kwon on 7/29/19.
//

#include "loadData.h"

void loadKitti(const std::pair<std::string, std::string> &cur, cv::Mat &out) {
    cv::Mat left = cv::imread(cur.first, cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(cur.second, cv::IMREAD_GRAYSCALE);
//    left.convertTo(left, CV_8UC1);
//    right.convertTo(right, CV_8UC1);

    if (left.empty() || right.empty()) {
        std::cerr << "Left or Right is empty" << std::endl;
    }

    auto stereo = cv::StereoBM::create(0, 21);
    stereo->compute(left, right, out);
    if (SHOW) cv::imshow("left", left);
    if (SHOW) cv::imshow("right", right);
    if (SHOW) cv::imshow("disparity map", out);
    if (SHOW) cv::waitKey(1);
}

void loadKittiMono(const std::pair<std::string, std::string> &cur, cv::Mat &out, int type) {
    if (type == 0) {
        out = cv::imread(cur.first);
    } else {
        out = cv::imread(cur.second);
    }
    if (SHOW) cv::imshow("mono", out);
    if (SHOW) cv::waitKey(1);
}

cv::Point3f loadPoseXYZ(const std::string &pose) {
    float x, y, z;
    std::istringstream parser(pose);
    for (int j = 0; j < 12; j++) {
        parser >> z;
        if (j == 7) y = z;
        if (j == 3) x = z;
    }
    return cv::Point3f(x, y, z);
}

cv::Point2f loadTruePose(const std::string &pose) {
    float x, y;
    std::istringstream parser(pose);
    for (int j = 0; j < 12; j++) {
        parser >> y;
        if (j == 3) x = y;
    }
    return cv::Point2f(x+300, y+100);
}