//
// Created by Do Hyung Kwon on 7/24/19.
//

#ifndef TOYSLAM_BUNDLEADJUSTMENT_H
#define TOYSLAM_BUNDLEADJUSTMENT_H

#include <iostream>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/StdVector>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
//#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>

#include <suitesparse/cholmod.h>

#include "cameraParamters.h"

static constexpr double PIXEL_NOISE = 1.0;
static constexpr double OUTLIER_RATIO = 0.0;
static constexpr bool ROBUST_KERNEL = true;
static constexpr bool STRUCTURE_ONLY = false;
static constexpr bool DENSE = false;
static constexpr int OPTIMIZE_COUNT = 70;

int bundleAdjustment3d2d(const std::vector<cv::Point3f> &points_3d,
                     const std::vector<cv::Point2f> &points_2d,
                     const cv::Mat &K,
                     cv::Mat &R,
                     cv::Mat &t);

#endif //TOYSLAM_BUNDLEADJUSTMENT_H