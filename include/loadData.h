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
        if (!TEST) {
            sprintf(tmp, "../data/dataset/sequences/%02d/image_0/%06d.png", SEQUENCE_NUM, idx);
            filenameLeft = tmp;
            sprintf(tmp, "../data/dataset/sequences/%02d/image_1/%06d.png", SEQUENCE_NUM, idx);
            filenameRight = tmp;
        } else {
            sprintf(tmp, "../tmp/images/%02d/image_0/%06d.png", SEQUENCE_NUM, idx);
            filenameLeft = tmp;
            sprintf(tmp, "../tmp/images/%02d/image_1/%06d.png", SEQUENCE_NUM, idx);
            filenameRight = tmp;
        }
    }

public:
    explicit kittiLoader(int idx_) : idx{idx_} {
        if (!TEST) sprintf(tmp, "../data/dataset/poses/%02d.txt", SEQUENCE_NUM);
        else sprintf(tmp, "../tmp/images/poses/%02d.txt", SEQUENCE_NUM);
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

void loadKitti(const std::pair<std::string, std::string> &cur, cv::Mat &out);

void loadKittiMono(const std::pair<std::string, std::string> &cur, cv::Mat &out, int type);

cv::Point2f loadTruePose(const std::string &pose);

cv::Point3f loadPoseXYZ(const std::string &pose);

#endif //TOYSLAM_LOADDATA_H
