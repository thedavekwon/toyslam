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

//int main() {
//    for (auto &cur : kitti_range(10)) {
//        std::cout << cur.first << " " << cur.second << std::endl;
//        cv::Mat out;
//        //loadKitti(cur, out);
//        loadKittiMono(cur, out, 0);
//    }
//}
