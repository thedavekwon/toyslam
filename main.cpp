#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap("video.mp4");

    std::cout << cap << std::endl;

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream!" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        cv::imshow("Frame", frame);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}