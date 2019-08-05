//
// Created by Do Hyung Kwon on 7/29/19.
//

#ifndef TOYSLAM_LOADDATA_H
#define TOYSLAM_LOADDATA_H

#include <iostream>
#include <iterator>
#include <string>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

class kittiLoader {
private:
    std::string filenameLeft;
    std::string filenameRight;
    char tmp[100]{};
    int idx = 0;

    void storeString() {
        sprintf(tmp, "./data/tmp/images/00/image_0/%06d.png", idx);
        filenameLeft = tmp;
        sprintf(tmp, "./data/tmp/images/00/image_1/%06d.png", idx);
        filenameRight = tmp;
    }

public:
    explicit kittiLoader(int idx_) : idx{idx_} {
        storeString();
    };

    const std::pair<std::string, std::string> operator*() const { return {filenameLeft, filenameRight}; }

    kittiLoader &operator++() {
        idx++;
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

class kitti_range {
private:
    kittiLoader begin_it;
    int end_n;

public:
    explicit kitti_range(int end_n_, int begin_n_ = 0)
            : begin_it{kittiLoader(begin_n_)}, end_n{end_n_ + 1} {};

    kittiLoader begin() {
        return begin_it;
    }

    kittiLoader end() {
        return kittiLoader(end_n);
    }
};

const bool SHOW = true;

void loadKitti(const std::pair<std::string, std::string> &cur, cv::Mat &out);

void loadKittiMono(const std::pair<std::string, std::string> &cur, cv::Mat &out, int type);

#endif //TOYSLAM_LOADDATA_H
