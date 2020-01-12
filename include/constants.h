//
// Created by vr-lab on 8/12/19.
//

#ifndef TOYSLAM_CONSTANTS_H
#define TOYSLAM_CONSTANTS_H

const double PIXEL_NOISE = 1.0;
const double OUTLIER_RATIO = 0.0;
const bool ROBUST_KERNEL = true;
const bool STRUCTURE_ONLY = false;
const bool DENSE = false;
const int OPTIMIZE_COUNT = 70;
const int MAX_FEATURES = 2000;
const float RATIO_THRESH = 0.7f;
const int MIN_FEATURE_THRESHOLD = 1500;
const bool FEATURE_DEBUG = false;
const bool DEBUG = false;
const bool SHOW = false;
const bool RAWSHOW = false;
const int SEQUENCE_NUM = 0;
const double SCALE_THRESHOLD = 0.08;
const int MAX_FLOW_VIS = 100;
const bool MINMAX_FILTER = true;
const bool TEST = true;

const int FEATURE_TYPE = 4;             // 1: ORB, 2: FAST, 3: SURF, 4: SIFT
const int MOTION_TYPE = 2;              // 1: Optical Flow, 2: 2D2D, 3: 2D3D, 4: 3D3D

#endif //TOYSLAM_CONSTANTS_H
