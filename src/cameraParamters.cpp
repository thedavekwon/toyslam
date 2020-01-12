//
// Created by Do Hyung Kwon on 7/29/19.
//

#include "cameraParamters.h"

cv::Mat loadCalibrationMatrix(int type) {
    if (type == 0) return cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;
    else
        return (cv::Mat_<double>(3, 3) <<
                                       7.188560000000e+02, 0, 6.071928000000e+02,
                                       0, 7.188560000000e+02, 1.852157000000e+02,
                                       0, 0, 1);
}

cv::Mat loadCalibrationMatrixKitti() {
    char tmp[100]{};
    if (!TEST) sprintf(tmp, "../data/dataset/sequences/%02d/calib.txt", SEQUENCE_NUM);
    else sprintf(tmp, "../tmp/images/%02d/calib.txt", SEQUENCE_NUM);

    std::ifstream calib_file(tmp);
    if (calib_file.is_open()) {
        std::string line;
        std::string camera = "P0: ";

        while (std::getline(calib_file, line)) {
            if (line.compare(0, camera.length(), camera) == 0) break;
        }

        double t, fx, fy, cx, cy;
        std::istringstream ss(line.substr(4));

        for (int i = 0; i < 12; i++) {
            ss >> t;
            if (i == 0) fx = t;
            else if (i == 2) cx = t;
            else if (i == 5) fy = t;
            else if (i == 6) cy = t;
        }
        return (cv::Mat_<double>(3, 3) <<
                                       fx, 0, cx,
                                       0, fy, cy,
                                       0, 0, 1);
    } else {
        std::cerr << "Failed to open Calibration File!" << std::endl;
        exit(EXIT_FAILURE);
    }
}

cv::Point2d loadPrincipalPoint(int type) {
    if (type == 0)
        return cv::Point2d(
                360 / 2,
                480 / 2);
    else return cv::Point2d(607.1928, 185.2157);
}

double loadFocalLength(int type) {
    if (type == 0) return 100.0;
    else return 718.8560;
}

double loadScale(const std::string &prevPose, const std::string &curPose) {
    auto prevP = loadPoseXYZ(prevPose);
    auto curP = loadPoseXYZ(curPose);

    double scale = sqrt(pow(curP.x - prevP.x, 2) + pow(curP.y - prevP.y, 2) + pow(curP.z - prevP.z, 2));
    std::cout << scale << std::endl;
    return scale;
}