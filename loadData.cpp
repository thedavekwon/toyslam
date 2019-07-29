//
// Created by Do Hyung Kwon on 7/29/19.
//

#include "loadData.h"

void loadKitti(std::string path) {
    char filename1[100];
    char filename2[100];
    sprintf(filename1, "/data/kitti/image_2/%06d.png", 0);
    sprintf(filename2, "/data/kitti/image_3/%06d.png", 0);

    cv::Mat left = cv::imread(filename1);
    cv::Mat right = cv::imread(filename2);
    cv::Mat out;

    if (left.empty() || right.empty()) {
        std::cerr << "Left or Right is empty" << std::endl;
    }

    auto stereo = cv::StereoBM::create(16, 15);
    stereo->compute(left, right, out);
    cv::imshow("disparity map", out);
    cv::waitKey(0);
}
