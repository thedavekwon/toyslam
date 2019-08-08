//
// Created by Do Hyung Kwon on 7/29/19.
//

#include "cameraParamters.h"

cv::Mat loadCalibrationMatrix(int type) {
    if (type == 0) return cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1;
    else return (cv::Mat_<double>(3, 3) <<
                7.188560000000e+02, 0, 6.071928000000e+02,
                0, 7.188560000000e+02, 1.852157000000e+02,
                0, 0, 1);
}

cv::Point2d loadPrincipalPoint(int type) {
    if (type == 0) return cv::Point2d(
                360 / 2,
                480 / 2);
    else return cv::Point2d(607.1928, 185.2157);
}

double loadFocalLength(int type) {
    if (type == 0) return 100.0;
    else return 718.8560;
}

double loadScale(int frameId, int type) {
//    std::ifstream poses("./tmp/images/poses/00.txt");
    std::ifstream poses("./data/dataset/poses/00.txt");
    std::string info;
    double x, y, z;
    double prevX, prevY, prevZ;
    while (frameId--) {
        std::getline(poses, info);
        prevX = x;
        prevY = y;
        prevZ = z;

        std::istringstream parser(info);
        for (int j = 0; j < 12; j++) {
            parser >> z;
            if (j == 7) y = z;
            if (j == 3) x = z;
        }
    }

    double scale = sqrt(pow(x-prevX, 2) + pow(y-prevY, 2) + pow(z-prevZ, 2)) ;
    std::cout << scale << std::endl;
    return scale;
}

cv::Point2f loadTruePose(int frameId) {
    std::ifstream poses("./data/dataset/poses/00.txt");
    std::string info;
    float x, y;
    while (frameId--) std::getline(poses, info);

    std::istringstream parser(info);
    for (int j = 0; j < 12; j++) {
        parser >> y;
        if (j == 3) x = y;
    }
    return cv::Point2f(x+300, y+100);
}