//
// Created by Do Hyung Kwon on 7/26/19.
//

#include "featureExtractionandMatching.h"

cv::VideoCapture getCapture();

void sequenceFromKitti() {
    int frameCnt = 0;
    cv::Mat prevFrame, prevkFrame, prevDes, poseR, poseT;
    cv::Point2f truePose;
    std::vector<cv::KeyPoint> prevKps;

    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);
    //cv::rectangle(trajectory, cv::Point(10, 10), cv::Point(550, 550), CV_RGB(0, 0, 0), cv::FILLED);
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

    for (auto &cur : kitti_range(4540)) {
        cv::Mat frame, kFrame, des;
        std::vector<cv::KeyPoint> kps;
        loadKittiMono(cur, frame, 0);
        if (frame.empty()) break;

        std::vector<cv::Point2f> keyPoints;
        extractFeature(frame, kFrame, kps, keyPoints, des, FEATURE_TYPE);

        cv::imshow("Frame", kFrame);
        cv::waitKey(1);

        if (frameCnt) {
            std::vector<cv::DMatch> good_matches = get_matches(prevDes, des, FEATURE_TYPE);

            cv::Mat R, t, mask;
            // poseEstimation2D2D(prevKps, kps, good_matches, mask,R, t);
             poseEstimation3D2DwithTriangulation(prevKps, kps, good_matches, mask, R, t);
            if (frameCnt == 1) {
                poseR = R.clone();
                poseT = t.clone();
            }
            if (frameCnt > 2) {
                auto scale = loadScale(frameCnt, 0);

                if (scale>0.1) {
                    poseT = poseT + scale * (poseR * t);
                    poseR = R * poseR;
                }

                truePose = loadTruePose(frameCnt);
                if (DEBUG) std::cout << "current position: " << std::endl;
                if (DEBUG) std::cout << poseT << std::endl;
                int x = int(poseT.at<double>(0)) + 300;
                int y = int(poseT.at<double>(2)) + 100;
                //std::cout << frameCnt << " " << x << " " << y << std::endl;
                cv::circle(trajectory, cv::Point2f(x, y), 1, CV_RGB(255, 0, 0), 2);
                cv::circle(trajectory, truePose, 1, CV_RGB(0, 0, 255), 2);
                imshow("Trajectory", trajectory);
            }
            if (SHOW) {
                cv::Mat img_matches;
                cv::drawMatches(prevFrame, prevKps, frame, kps, good_matches, img_matches, cv::Scalar::all(-1),
                                cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                cv::imshow("Good Matches", img_matches);
                cv::waitKey(1);
            }
        }

        prevFrame = frame.clone();
        prevkFrame = kFrame.clone();
        prevDes = des.clone();
        prevKps = kps;
        frameCnt++;
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void sequenceFromKittiOpticalFlow() {
    int frameCnt = 0;
    cv::Mat prevFrame, prevkFrame, prevDes, poseR, poseT;
    cv::Point2f truePose;
    std::vector<cv::KeyPoint> prevKps;
    std::vector<cv::Point2f> prevKeyPoints;
    std::vector<cv::Point2f> keyPoints;
    std::vector<cv::KeyPoint> kps;

    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);
//    cv::rectangle(trajectory, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

    for (auto &cur : kitti_range(4540)) {
        std::cout << "frameId: " << frameCnt << std::endl;
        cv::Mat frame, kFrame, des;
        kps.clear();

        loadKittiMono(cur, frame, 0);
        if (frame.empty()) break;

        // cv::imshow("Frame", kFrame);
        cv::waitKey(1);

        if (frameCnt == 0) {
            extractFeature(frame, kFrame, kps, keyPoints, des, FEATURE_TYPE);
        }

        if (frameCnt) {
            cv::Mat R, t, mask;
            if (frameCnt == 1) {
                std::vector<uchar> status;
                featureTrackingWithOpticalFlow(prevFrame, frame, prevKeyPoints, keyPoints, status);
                poseEstimationOpticalFlow(prevKeyPoints, keyPoints, mask, R, t);
                poseR = R.clone();
                poseT = t.clone();
            }
            if (frameCnt > 2) {
                std::vector<uchar> status;
                featureTrackingWithOpticalFlow(prevFrame, frame, prevKeyPoints, keyPoints, status);
                poseEstimationOpticalFlow(prevKeyPoints, keyPoints, mask, R, t);

                auto scale = loadScale(frameCnt, 0);

                if (scale>0.1) {
                    poseT = poseT + scale * (poseR * t);
                    poseR = R * poseR;
                }
                if (keyPoints.size() < MIN_FEATURE_THRESHOLD) {
                    if (FEAUTRE_DEBUG) std::cout << "before: " << prevKeyPoints.size() << " " << keyPoints.size() << " ";
                    extractFeature(prevFrame, prevkFrame, prevKps, prevKeyPoints, prevDes, FEATURE_TYPE);
                    if (FEAUTRE_DEBUG) std::cout << "after: " << prevKeyPoints.size() << " " << keyPoints.size() << " ";
                    featureTrackingWithOpticalFlow(prevFrame, frame, prevKeyPoints, keyPoints, status);
                    if (FEAUTRE_DEBUG) std::cout << "aafter: " << prevKeyPoints.size() << " " << keyPoints.size() << std::endl;
                }

                truePose = loadTruePose(frameCnt);
                int x = int(poseT.at<double>(0)) + 300;
                int y = int(poseT.at<double>(2)) + 100;
                //std::cout << frameCnt << " " << x << " " << y << std::endl;
                cv::circle(trajectory, cv::Point2f(x, y), 1, CV_RGB(255, 0, 0), 2);
                cv::circle(trajectory, truePose, 1, CV_RGB(0, 0, 255), 2);
                imshow("Trajectory", trajectory);
            }
            if (SHOW) {
                cv::Mat img_matches;
                // cv::imshow("Good Matches", img_matches);
                cv::waitKey(1);
            }
        }

        prevFrame = frame.clone();
        prevkFrame = kFrame.clone();
        prevDes = des.clone();
        if (DEBUG) std::cout << "before copying: " << prevKeyPoints.size() << " " << keyPoints.size() << " ";
        prevKeyPoints = keyPoints;
        if (DEBUG) std::cout << "after copying: " << prevKeyPoints.size() << " " << keyPoints.size() << std::endl;
        prevKps = kps;
        frameCnt++;
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void sequenceFromVideo() {
    int frameCnt = 0;
    cv::VideoCapture cap = getVideoCapture("data/video.mp4");

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream!" << std::endl;
        return;
    }
    cv::Mat prevFrame, prevkFrame, prevDes, poseR, poseT;
    cv::Point2f truePose;
    std::vector<cv::KeyPoint> prevKps;

    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);
    cv::rectangle(trajectory, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0, 0, 0), cv::FILLED);
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

    for (auto &cur : kitti_range(4540)) {
        cv::Mat frame, kFrame, des;
        std::vector<cv::KeyPoint> kps;
        loadKittiMono(cur, frame, 0);
        if (frame.empty()) break;

        std::vector<cv::Point2f> keyPoints;
        extractFeature(frame, kFrame, kps, keyPoints, des, FEATURE_TYPE);

        cv::imshow("Frame", kFrame);
        cv::waitKey(1);

        if (frameCnt) {
            std::vector<cv::DMatch> good_matches = get_matches(prevDes, des, FEATURE_TYPE);

            cv::Mat R, t, mask;
            //poseEstimation2D2D(prevKps, kps, good_matches, mask,R, t);
            poseEstimation3D2DwithTriangulation(prevKps, kps, good_matches, mask, R, t);

            if (frameCnt == 1) {
                poseR = R.clone();
                poseT = t.clone();
            }
            if (frameCnt > 2) {
                poseT = poseT + loadScale(frameCnt, 0) * (poseR * t);
                poseR = R * poseR;
                truePose = loadTruePose(frameCnt);

                int x = int(poseT.at<double>(0)) + 300;
                int y = int(poseT.at<double>(2)) + 100;
                std::cout << frameCnt << " " << x << " " << y << std::endl;
                cv::circle(trajectory, cv::Point2f(x, y), 1, CV_RGB(255, 0, 0), 2);
                cv::circle(trajectory, truePose, 1, CV_RGB(0, 0, 255), 2);
                imshow("Trajectory", trajectory);
            }
            if (SHOW) {
                cv::Mat img_matches;
                cv::drawMatches(prevFrame, prevKps, frame, kps, good_matches, img_matches, cv::Scalar::all(-1),
                                cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                cv::imshow("Good Matches", img_matches);
                cv::waitKey(1);
            }
        }

        prevFrame = frame;
        prevkFrame = kFrame;
        prevDes = des;
        prevKps = kps;
        frameCnt++;
    }
    cv::waitKey(0);
    cap.release();
    cv::destroyAllWindows();
}

cv::VideoCapture getVideoCapture(std::string s) {
    cv::VideoCapture cap(s);
    return cap;
}

void extractFeature(cv::Mat &frame, cv::Mat &kFrame, std::vector<cv::KeyPoint> &kps, std::vector<cv::Point2f> &points,
        cv::Mat &des, const int featureType) {
    if (featureType == 1) {
        cv::Ptr<cv::Feature2D> fet;
        fet = cv::ORB::create(MAX_FEATURES_THRESHOLD);
        fet->detectAndCompute(frame, cv::noArray(),kps, des);
        cv::KeyPoint::convert(kps, points, std::vector<int>());
        cv::drawKeypoints(frame, kps, kFrame);
    } else if (featureType == 2) {
        cv::FAST(frame, kps, 5, true);
        cv::KeyPoint::convert(kps, points, std::vector<int>());
        cv::drawKeypoints(frame, kps, kFrame);
    }
}


std::vector<cv::DMatch> filterMatches(std::vector<cv::DMatch> &matches) {
    std::vector<cv::DMatch> filterd_matches;
    float min_dist = std::numeric_limits<float>::max(), max_dist = 0;

    for (auto &match: matches) {
        min_dist = std::min(min_dist, match.distance);
        max_dist = std::max(max_dist, match.distance);
    }

    for (auto &match: matches) {
        if (match.distance <= std::max(2*min_dist, 30.0f)) {
            filterd_matches.push_back(match);
        }
    }
    return filterd_matches;
}

std::vector<cv::DMatch> get_matches(const cv::Mat &prevDes, const cv::Mat &des, const int featureType) {
    if (featureType == 1) {
        cv::Ptr<cv::BFMatcher> matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_HAMMING2, true);
        std::vector<cv::DMatch> matches;
        matcher->match(des, prevDes, matches);
        return filterMatches(matches);;
    } else if (featureType == 2) {
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
        return filterMatches(good_matches);;
    }
}

void featureTrackingWithOpticalFlow(const cv::Mat &prevFrame, const cv::Mat &frame, std::vector<cv::Point2f> &prevKeypoints,
                                    std::vector<cv::Point2f> &keyPoints, std::vector<uchar>& status) {
    std::vector<float> error;
    // default window size for the calcOpticalFlowPyrLK
    auto winSize = cv::Size(5, 5);
    // termination criteria for iterative algorithms
    // default criterion for the calcOpticalFlowPyrLK
    auto criterion = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 100, 0.01);
    // Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
    if (DEBUG) std::cout << "before tracking: " << prevKeypoints.size() << " " << keyPoints.size() << std::endl;
    cv::calcOpticalFlowPyrLK(prevFrame, frame, prevKeypoints, keyPoints, status, error, winSize, 5, criterion, 0, 0.001);
    if (DEBUG) std::cout << "after tracking: " << prevKeypoints.size() << " " << keyPoints.size() << std::endl;

    // filter out failed at Tracking
    int indexCorrection = 0;
    for (size_t i = 0; i < status.size(); i++) {
        cv::Point2f pt = keyPoints.at(i - indexCorrection);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
            if((pt.x<0)||(pt.y<0))	{
                status.at(i) = 0;
            }
            prevKeypoints.erase (prevKeypoints.begin() + (i - indexCorrection));
            keyPoints.erase (keyPoints.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}