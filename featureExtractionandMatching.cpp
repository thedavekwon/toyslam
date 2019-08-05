//
// Created by Do Hyung Kwon on 7/26/19.
//

#include "featureExtractionandMatching.h"

cv::VideoCapture getCapture();

void sequenceFromKitti() {
    int frameCnt = 0;
    cv::Mat prevFrame, prevkFrame, prevDes, prevR, prevT;
    std::vector<cv::KeyPoint> prevKps;

    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);
    cv::rectangle(trajectory, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);

    for (auto &cur : kitti_range(100)) {
        cv::Mat frame, kFrame, des;
        std::vector<cv::KeyPoint> kps;
        std::cout << cur.first << " " << cur.second << std::endl;
        loadKittiMono(cur, frame, 0);
        if (frame.empty()) break;

        extractFeature(frame, kFrame, kps, des, 1);

        cv::imshow("Frame", kFrame);
        cv::waitKey(1);

        if (frameCnt) {
            std::vector<cv::DMatch> good_matches = get_matches(prevDes, des);

            cv::Mat R, t;
            poseEstimation2D2D(prevKps, kps, good_matches, R, t);
             //poseEstimation3D2DwithTriangulation(prevKps, kps, good_matches, R, t);

            cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

            if (frameCnt > 2) {
                prevT = prevT + prevR * t;
                prevR = prevR * R;

                int x = int(prevT.at<double>(0)) + 300;
                int y = int(prevT.at<double>(2)) + 100;
                std::cout << x << " " << y << std::endl;
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

    cv::destroyAllWindows();
}

std::vector<cv::DMatch> get_matches(const cv::Mat &prevDes, const cv::Mat &des) {
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(
            cv::makePtr<cv::flann::LshIndexParams>(20, 10, 2)
    );
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(des, prevDes, knn_matches, 3);
    std::vector<cv::DMatch> good_matches;
    for (auto &knn_match : knn_matches) {
        if (knn_match[0].distance < RATIO_THRESH * knn_match[1].distance) {
            good_matches.push_back(knn_match[0]);
        }
    }
    return good_matches;
}

void sequenceFromVideo() {
    int frameCnt = 0;
    cv::VideoCapture cap = getCapture("data/video.mp4");

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream!" << std::endl;
        return;
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
                    cv::makePtr<cv::flann::LshIndexParams>(16, 32, 2)
            );
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher->knnMatch(des, prevDes, knn_matches, 3);
            std::vector<cv::DMatch> good_matches;
            for (auto &knn_match : knn_matches) {
                if (knn_match[0].distance < RATIO_THRESH * knn_match[1].distance) {
                    good_matches.push_back(knn_match[0]);
                }
            }

            cv::Mat R, t;
            // poseEstimation2D2D(prevKps, kps, good_matches, R, t);
            poseEstimation3D2DwithTriangulation(prevKps, kps, good_matches, R, t);

            cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

            if (frameCnt > 2) {
                prevT = prevT + prevR * t;
                prevR = prevR * R;

                int x = int(prevT.at<double>(0)) + 300;
                int y = int(prevT.at<double>(2)) + 100;
                //std::cout << x << " " << y << std::endl;
                cv::circle(trajectory, cv::Point(x, y), 1, CV_RGB(255, 0, 0), 3);
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
}

cv::VideoCapture getCapture(std::string s) {
    cv::VideoCapture cap(s);
    return cap;
}

void extractFeature(cv::Mat &frame, cv::Mat &kFrame, std::vector<cv::KeyPoint> &kps, cv::Mat &des, int featureType) {
    cv::Ptr<cv::Feature2D> fet;
    fet = cv::ORB::create(MAX_FEATURES);
    fet->detectAndCompute(frame, cv::noArray(), kps, des);
    cv::drawKeypoints(frame, kps, kFrame);
}