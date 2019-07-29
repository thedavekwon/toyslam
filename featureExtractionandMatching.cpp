//
// Created by Do Hyung Kwon on 7/26/19.
//

#include "featureExtractionandMatching.h"

int mainLoop() {
    int frameCnt = 0;
    cv::VideoCapture cap("video.mp4");

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream!" << std::endl;
        return -1;
    }
    cv::Mat prevFrame, prevkFrame, prevDes, prevR, prevT;
    std::vector<cv::KeyPoint> prevKps;

    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);
    cv::rectangle(trajectory, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);

    while (true) {
        cv::Mat frame, kFrame, des;
        std::vector<cv::KeyPoint> kps;
        cap >> frame;
        if (frame.empty()) break;

        extractFeature(frame, kFrame, kps, des, 1);

        cv::imshow("Frame", kFrame);
        cv::waitKey(1);

        if (frameCnt) {
            cv::Ptr<cv::FlannBasedMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(
                    cv::makePtr<cv::flann::LshIndexParams>(5, 10, 2));
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher->knnMatch(des, prevDes, knn_matches, 3);
            std::vector<cv::DMatch> good_matches;
            for (size_t i = 0; i < knn_matches.size(); i++) {
                if (knn_matches[i][0].distance < RATIO_THRESH * knn_matches[i][1].distance) {
                    good_matches.push_back(knn_matches[i][0]);
                }
            }

            cv::Mat R, t;
            poseEstimation2D2D(prevKps, kps, good_matches, R, t);

            cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

            if (frameCnt > 2) {
                prevT = prevT + 4 * prevR * t;
                prevR = prevR * R;

                int x = int(prevT.at<double>(0)) + 300;
                int y = int(prevT.at<double>(1)) + 300;
                cv::circle(trajectory, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 4);
                imshow("Trajectory", trajectory);
            }
            if (SHOW) {
                cv::Mat img_matches;
                cv::drawMatches(prevFrame, prevKps, frame, kps, good_matches, img_matches, cv::Scalar::all(-1),
                                cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                cv::imshow("Good Matches", img_matches);
                cv::waitKey();
            }

            prevR = R;
            prevT = t;
        }

        prevFrame = frame;
        prevkFrame = kFrame;
        prevDes = des;
        prevKps = kps;
        frameCnt++;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

void extractFeature(cv::Mat &frame, cv::Mat &kFrame, std::vector<cv::KeyPoint> &kps, cv::Mat &des, int featureType) {
    cv::Ptr<cv::Feature2D> fet;
    if (featureType == 1) {
        fet = cv::ORB::create(MAX_FEATURES);
    }
    fet->detectAndCompute(frame, cv::noArray(), kps, des);
    cv::drawKeypoints(frame, kps, kFrame);
}