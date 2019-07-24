//
// Created by dodo on 7/24/19.
//

#ifndef TOYSLAM_BUNDLEADJUSTMENT_H
#define TOYSLAM_BUNDLEADJUSTMENT_H

#include <iostream>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/StdVector>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>

const double PIXEL_NOISE = 1.0;
const double OUTLIER_RATIO = 0.0;
const bool ROBUST_KERNEL = false;
const bool STRUCTURE_ONLY = false;
const bool DENSE = false;

#endif //TOYSLAM_BUNDLEADJUSTMENT_H
