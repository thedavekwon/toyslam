//
// Created by Do Hyung Kwon on 7/26/19.
//

#include "featureExtractionandMatching.h"

cv::VideoCapture getVideoCapture(std::string s) {
    cv::VideoCapture cap(s);
    return cap;
}

void extractFeature(cv::Mat &frame, cv::Mat &kFrame, std::vector<cv::KeyPoint> &kps, std::vector<cv::Point2f> &points,
                    cv::Mat &des, const int featureType) {
    if (featureType == 1) {
        cv::Ptr<cv::Feature2D> fet;
        fet = cv::ORB::create(MAX_FEATURES);
        fet->detectAndCompute(frame, cv::noArray(), kps, des);
        cv::KeyPoint::convert(kps, points, std::vector<int>());
        cv::drawKeypoints(frame, kps, kFrame);
    } else if (featureType == 2) {
        cv::FAST(frame, kps, 5, true);
        cv::KeyPoint::convert(kps, points, std::vector<int>());
        cv::drawKeypoints(frame, kps, kFrame);
    } else if (featureType == 3) {
        cv::Ptr<cv::Feature2D> fet;
        fet = cv::xfeatures2d::SURF::create();
        fet->detectAndCompute(frame, cv::noArray(), kps, des);
        cv::KeyPoint::convert(kps, points, std::vector<int>());
        cv::drawKeypoints(frame, kps, kFrame);
    } else if (featureType == 4) {
        cv::Ptr<cv::Feature2D> fet;
        fet = cv::xfeatures2d::SIFT::create();
        fet->detectAndCompute(frame, cv::noArray(), kps, des);
        cv::KeyPoint::convert(kps, points, std::vector<int>());
        cv::drawKeypoints(frame, kps, kFrame);
    }
}


std::vector<cv::DMatch> filterMatches(const cv::Mat &des, std::vector<cv::DMatch> &matches) {
    std::vector<cv::DMatch> filtered_matches;
    float min_dist = std::numeric_limits<float>::max(), max_dist = 0;
    if (DEBUG) std::cout << "Descriptor Size: " << des.rows << std::endl;
    for (auto &match : matches) {
        min_dist = std::min(min_dist, match.distance);
        max_dist = std::max(max_dist, match.distance);
    }
    if (DEBUG) std::cout << "Max Distance: " << max_dist << std::endl;
    if (DEBUG) std::cout << "Min Distance: " << min_dist << std::endl;

    if (!DEBUG) std::cout << "Before Match Filtering: " << matches.size() << std::endl;
    for (auto &match : matches) {
        if (match.distance <= std::max(2 * min_dist, 40.0f)) {
            filtered_matches.push_back(match);
        }
    }
    if (!DEBUG) std::cout << "After Match Filtering: " << filtered_matches.size() << std::endl;
    return filtered_matches;
}

std::vector<cv::DMatch> get_matches(const cv::Mat &prevDes, const cv::Mat &des, const int featureType) {
    if (featureType == 1) {
        cv::Ptr<cv::BFMatcher> matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_HAMMING2, true);
//        cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming");
        std::vector<cv::DMatch> matches;
        matcher->match(des, prevDes, matches);
        if (MINMAX_FILTER) return filterMatches(des, matches);;
        return matches;
    } else if (featureType == 2) {
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        std::vector<cv::DMatch> matches;
        matcher->match(des, prevDes, matches);
        if (MINMAX_FILTER) return filterMatches(des, matches);;
        return matches;
    } else if (featureType == 3 || featureType == 4) {
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(des, prevDes, knn_matches, 5);
        std::vector<cv::DMatch> good_matches;
        for (auto &knn_match : knn_matches) {
            if (knn_match[0].distance < RATIO_THRESH * knn_match[1].distance) {
                good_matches.push_back(knn_match[0]);
            }
        }
        if (MINMAX_FILTER) return filterMatches(des, good_matches);;
        return good_matches;
    }
}

void
featureTrackingWithOpticalFlow(const cv::Mat &prevFrame, const cv::Mat &frame, std::vector<cv::Point2f> &prevKeypoints,
                               std::vector<cv::Point2f> &keyPoints, std::vector<uchar> &status) {
    std::vector<float> error;
    // default window size for the calcOpticalFlowPyrLK
    auto winSize = cv::Size(5, 5);
    // termination criteria for iterative algorithms
    // default criterion for the calcOpticalFlowPyrLK
    auto criterion = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 60, 0.01);
    // Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
    if (DEBUG) std::cout << "before tracking: " << prevKeypoints.size() << " " << keyPoints.size() << std::endl;
    cv::calcOpticalFlowPyrLK(prevFrame, frame, prevKeypoints, keyPoints, status, error, winSize, 5, criterion, 0,
                             0.001);
    if (DEBUG) std::cout << "after tracking: " << prevKeypoints.size() << " " << keyPoints.size() << std::endl;

    // filter out failed at Tracking
    int indexCorrection = 0;
    for (size_t i = 0; i < status.size(); i++) {
        cv::Point2f pt = keyPoints.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            prevKeypoints.erase(prevKeypoints.begin() + (i - indexCorrection));
            keyPoints.erase(keyPoints.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}