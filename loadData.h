//
// Created by Do Hyung Kwon on 7/29/19.
//

#ifndef TOYSLAM_LOADDATA_H
#define TOYSLAM_LOADDATA_H

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "constants.h"

class kittiLoader {
private:
    std::string filenameLeft;
    std::string filenameRight;
    std::string pose;
    std::string prevPose;
    std::ifstream poses;

    char tmp[100]{};
    int idx = 0;

    void storeString() {
        sprintf(tmp, "./data/dataset/sequences/%02d/image_0/%06d.png", SEQUENCE_NUM, idx);
        filenameLeft = tmp;
        sprintf(tmp, "./data/dataset/sequences/%02d/image_1/%06d.png", SEQUENCE_NUM, idx);
        filenameRight = tmp;
    }

public:
    explicit kittiLoader(int idx_) : idx{idx_} {
        sprintf(tmp, "./data/dataset/poses/%02d.txt", SEQUENCE_NUM);
        poses = std::ifstream(tmp);
        if (poses.is_open()) std::getline(poses, pose);
        storeString();
    };

    std::vector<std::string> operator*() const { return {prevPose, pose, filenameLeft, filenameRight}; }

    kittiLoader &operator++() {
        idx++;
        prevPose = pose;
        if (poses.is_open()) std::getline(poses, pose);
        if (DEBUG) std::cout << pose << std::endl;
        storeString();
    }

    bool operator!=(const kittiLoader &other) {
        return other.idx != idx;
    }

    std::string getLeft() {
        return filenameLeft;
    }

    std::string getRight() {
        return filenameRight;
    }
};

//class kitti_range {
//private:
//    kittiLoader begin_it;
//    int end_n;
//
//public:
//    explicit kitti_range(std::string poses_filepath, int end_n_, int begin_n_ = 0)
//            : begin_it{kittiLoader(poses_filepath, begin_n_)}, end_n{end_n_ + 1} {
//    };
//
//    kittiLoader begin() {
//        return begin_it;
//    }
//
//    kittiLoader end() {
//        return kittiLoader(end_n);
//    }
//};

void loadKitti(const std::pair<std::string, std::string> &cur, cv::Mat &out);

void loadKittiMono(const std::pair<std::string, std::string> &cur, cv::Mat &out, int type);

#endif //TOYSLAM_LOADDATA_H
