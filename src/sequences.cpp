//
// Created by Do Hyung Kwon on 8/14/19.
//
#include "sequences.h"

void sequenceFromKitti3D2D() {
    int frameCnt = 0;
    cv::Mat prev2Frame, prev2KFrame, prev2Des;
    cv::Mat prevFrame, prevKFrame, prevDes;
    cv::Mat poseR, poseT;
    cv::Point2f truePose;
    cv::Mat K = loadCalibrationMatrixKitti();

    std::vector<cv::KeyPoint> prev2Kps, prevKps, kps;
    std::vector<cv::Point2f> keyPoints;

    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

    auto it = kittiLoader(0);
    const auto end = kittiLoader(4540);

    while (it != end) {
        cv::Mat frame, kFrame, des;
        kps.clear();
        auto cur = *it;
        loadKittiMono({cur[2], cur[3]}, frame, 0);
        if (frame.empty()) break;

        if (frameCnt == 0) {
            extractFeature(frame, kFrame, kps, keyPoints, des, FEATURE_TYPE);
            prev2Kps = kps;
            prev2Frame = frame;
            prev2KFrame = kFrame;
            prev2Des = des;
        } else {
            cv::Mat R, t, mask;
            if (frameCnt == 1) {
                extractFeature(frame, kFrame, kps, keyPoints, des, FEATURE_TYPE);
                prevKps = kps;
                prevFrame = frame;
                prevKFrame = kFrame;
                prevDes = des;
            }
            if (frameCnt >= 2) {
                std::vector<cv::Point3f> points_3d;
                std::vector<cv::Point2f> points_2d;
                std::vector<cv::DMatch> good_matches = get_matches(prev2Des, des, FEATURE_TYPE);
//                triangulation(prev2Kps, prevKps, good_matches, K, R, t, points_3d, points_2d);
                if (frameCnt == 2) {
                    poseR = R.clone();
                    poseT = t.clone();
                }
            }
        }
        frameCnt++;
        ++it;
    }
}

void sequenceFromKitti2D2D() {
    int frameCnt = 0;
    cv::Mat prevFrame, prevKFrame, prevDes, poseR, poseT;
    cv::Point2f truePose;
    std::vector <cv::KeyPoint> prevKps, kps;
    std::vector <cv::Point2f> prevKeyPoints, keyPoints;

    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

    auto it = kittiLoader(0);
    const auto end = kittiLoader(4540);

    while (it != end) {
        std::cout << "frameId: " << frameCnt << std::endl;
        cv::Mat frame, KFrame, des;
        kps.clear();
        keyPoints.clear();

        auto cur = *it;
        loadKittiMono({cur[2], cur[3]}, frame, 0);
        if (frame.empty()) break;

        extractFeature(frame, KFrame, kps, keyPoints, des, FEATURE_TYPE);

        if (SHOW) {
            cv::imshow("Frame", KFrame);
            cv::waitKey(1);
        }

        if (frameCnt) {
            auto good_matches = get_matches(prevDes, des, FEATURE_TYPE);

            cv::Mat R, t, mask;
            poseEstimation2D2D(prevKps, kps, good_matches, mask, R, t);
            if (frameCnt == 1) {
                poseR = R.clone();
                poseT = t.clone();
            } else {
                auto scale = loadScale(cur[0], cur[1]);

                if (scale > SCALE_THRESHOLD) {
                    poseT = poseT + scale * (poseR * t);
                    poseR = R * poseR;
                }

                draw2D(poseT, trajectory, cur);
                if (SHOW) {
                    cv::Mat img_matches;
                    std::cout << "KeyPoint1 Size: " << prevKps.size() << " KeyPoint2 Size: " << kps.size() << std::endl;
                    std::cout << "Matches Size: " << good_matches.size() << std::endl;
                    cv::drawMatches(prevFrame, prevKps, frame, kps, good_matches, img_matches, cv::Scalar::all(-1),
                                    cv::Scalar::all(-1), std::vector<char>(),
                                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                    cv::imshow("Good Matches", img_matches);
                    cv::waitKey(1);
                }
            }
        }

        prevFrame = frame.clone();
        prevKFrame = KFrame.clone();
        prevDes = des.clone();
        prevKeyPoints = keyPoints;
        prevKps = kps;
        frameCnt++;
        ++it;
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void sequenceFromKittiOpticalFlow() {
    // for visualization
    cv::RNG rng;

    int frameCnt = 0;
    cv::Mat prevFrame, prevkFrame, prevDes, poseR, poseT;
    cv::Point2f truePose;
    std::vector<cv::KeyPoint> prevKps, kps;
    std::vector<cv::Point2f> prevKeyPoints, keyPoints;
    std::vector<uchar> status;

    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);

    auto it = kittiLoader(0);
    const auto end = kittiLoader(4540);

    while (it != end) {
        std::cout << "frameId: " << frameCnt << std::endl;
        cv::Mat frame, kFrame, des;
        kps.clear();

        auto cur = *it;
        loadKittiMono({cur[2], cur[3]}, frame, 0);
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
                featureTrackingWithOpticalFlow(prevFrame, frame, prevKeyPoints, keyPoints, status);
                poseEstimationOpticalFlow(prevKeyPoints, keyPoints, mask, R, t);

                auto scale = loadScale(cur[0], cur[1]);

                if (scale > SCALE_THRESHOLD) {
                    poseT = poseT + scale * (poseR * t);
                    poseR = R * poseR;
                }
                if (keyPoints.size() < MIN_FEATURE_THRESHOLD) {
                    if (FEATURE_DEBUG)
                        std::cout << "before: " << prevKeyPoints.size() << " " << keyPoints.size() << " ";
                    extractFeature(prevFrame, prevkFrame, prevKps, prevKeyPoints, prevDes, FEATURE_TYPE);
                    if (FEATURE_DEBUG) std::cout << "after: " << prevKeyPoints.size() << " " << keyPoints.size() << " ";
                    featureTrackingWithOpticalFlow(prevFrame, frame, prevKeyPoints, keyPoints, status);
                    if (FEATURE_DEBUG)
                        std::cout << "aafter: " << prevKeyPoints.size() << " " << keyPoints.size() << std::endl;
                }
                draw2D(poseT, trajectory, cur);
            }
            if (SHOW && frameCnt > 2) {
                cv::Mat img_matches = cv::Mat::zeros(frame.size(), frame.type());
                int cnt = 0;
                for (int i = 0; i < prevKeyPoints.size(); i++) {
                    if (status[i] == 1) {
                        if (cnt % 10 != 0) {
                            cnt++;
                            continue;
                        }
                        auto color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                        cv::arrowedLine(img_matches, keyPoints[i], prevKeyPoints[i], color, 2);
                        cv::circle(frame, keyPoints[i], 5, color, -1);
                        cnt++;
                    }
                }
                cv::Mat img;
                add(frame, img_matches, img);
                cv::imshow("Good Matches", img);
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
        ++it;
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void sequenceFromKittiOpticalFlow3D() {
    int frameCnt = 0;
    cv::Mat prevFrame, prevkFrame, prevDes, poseR, poseT;
    cv::Point2f truePose;
    std::vector<cv::KeyPoint> prevKps;
    std::vector<cv::Point2f> prevKeyPoints;
    std::vector<cv::Point2f> keyPoints;
    std::vector<cv::KeyPoint> kps;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    auto K = loadCalibrationMatrix(1);

    auto it = kittiLoader(0);
    const auto end = kittiLoader(4500);

    while (it != end) {
        std::cout << "frameId: " << frameCnt << std::endl;
        cv::Mat frame, kFrame, des;
        kps.clear();

        auto cur = *it;
        loadKittiMono({cur[2], cur[3]}, frame, 0);
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

                auto scale = loadScale(cur[0], cur[1]);

                if (scale > 0.1) {
                    poseT = poseT + scale * (poseR * t);
                    poseR = R * poseR;
                }
                if (keyPoints.size() < MIN_FEATURE_THRESHOLD) {
                    if (FEATURE_DEBUG)
                        std::cout << "before: " << prevKeyPoints.size() << " " << keyPoints.size() << " ";
                    extractFeature(prevFrame, prevkFrame, prevKps, prevKeyPoints, prevDes, FEATURE_TYPE);
                    if (FEATURE_DEBUG) std::cout << "after: " << prevKeyPoints.size() << " " << keyPoints.size() << " ";
                    featureTrackingWithOpticalFlow(prevFrame, frame, prevKeyPoints, keyPoints, status);
                    if (FEATURE_DEBUG)
                        std::cout << "aafter: " << prevKeyPoints.size() << " " << keyPoints.size() << std::endl;
                }
                cv::Mat color = cv::Mat::zeros(frame.size(), CV_8UC3);
                //draw3D(pointcloud, poseR, poseT, color, frame, K);
                draw2Don3D(pointcloud, poseR, poseT);
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
        ++it;
    }
    cv::waitKey(0);
    cv::destroyAllWindows();

    voxelFiltering(pointcloud, filtered_pointcloud);
    filtered_pointcloud->is_dense = false;
    pcl::io::savePCDFileBinary("map.pcd", *filtered_pointcloud);
}

void sequenceFromKitti3D3D() {
    int frameCnt;
    cv::Mat prevFrame, prevKFrame, prevDes, poseR, poseT;
    cv::Point3f truePose;
    std::vector<cv::KeyPoint> prevKps;
}