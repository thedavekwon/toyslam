//
// Created by Do Hyung Kwon on 7/29/19.
//

#include "loadData.h"

class kittiLoader {
private:
    std::string filenameLeft;
    std::string filenameRight;
    char tmp[100]{};
    int idx = 0;

    void storeString() {
        sprintf(tmp, "./data/kitti/training/image_2/%06d.png", idx);
        filenameLeft = tmp;
        sprintf(tmp, "./data/kitti/training/image_3/%06d.png", idx);
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
            : begin_it{kittiLoader(begin_n_)}, end_n{end_n_} {};

    kittiLoader begin() {
        return begin_it;
    }

    kittiLoader end() {
        return kittiLoader(end_n);
    }
};

void loadKitti(const std::pair<std::string, std::string> &cur) {
    cv::Mat left = cv::imread(cur.first);
    cv::Mat right = cv::imread(cur.second);
    cv::Mat out;

    left.convertTo(left, CV_8UC1);
    right.convertTo(right, CV_8UC1);

    if (left.empty() || right.empty()) {
        std::cerr << "Left or Right is empty" << std::endl;
    }

    auto stereo = cv::StereoBM::create(16, 15);
    stereo->compute(left, right, out);
    cv::imshow("disparity map", out);
    cv::waitKey(0);
}

int main() {
    for (auto &cur : kitti_range(10)) {
        std::cout << cur.first << " " << cur.second <<  std::endl;
        loadKitti(cur);
    }
}
