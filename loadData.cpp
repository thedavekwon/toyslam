//
// Created by Do Hyung Kwon on 7/29/19.
//

#include "loadData.h"

void loadKitti(std::string path) {
    cv::dataset
    cv::Mat left = cv::imread(, IMCOLOR_READ);
    cv::Mat right = cv::imread(, IMCOLOR_READ);

    if (left.empty() || right.empty()) {
        std::cerr << "Left or Right is empty" << std::endl;
    }

    cv::Mat left_for_matcher, right_for_matcher, left_disp, right_disp, filtered_disp, solved_disp, solved_filter_disp;
    cv::Mat conf_map = cv::Mat(left.rows, left.cols, CV_8U);
    conf_map = cv::Scalar(255);
}