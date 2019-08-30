//
// Created by Do Hyung Kwon on 8/14/19.
//

#ifndef TOYSLAM_SEQUENCES_H
#define TOYSLAM_SEQUENCES_H

#include <algorithm>

#include <opencv4/opencv2/opencv.hpp>

#include "featureExtractionandMatching.h"
#include "drawMap.h"

void sequenceFromKitti2D2D();

void sequenceFromKitti3D3D();

void sequenceFromKittiOpticalFlow();

void sequenceFromKittiOpticalFlow3D();

#endif //TOYSLAM_SEQUENCES_H
