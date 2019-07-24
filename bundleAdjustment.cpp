//
// Created by dodo on 7/24/19.
//

#include "bundleAdjustment.h"

// from g2o demo

int BA() {
    std::cout << "PIXEL_NOISE: " <<  PIXEL_NOISE << std::endl;
    std::cout << "OUTLIER_RATIO: " << OUTLIER_RATIO<<  std::endl;
    std::cout << "ROBUST_KERNEL: " << ROBUST_KERNEL << std::endl;
    std::cout << "STRUCTURE_ONLY: " << STRUCTURE_ONLY<< std::endl;
    std::cout << "DENSE: "<<  DENSE << std::endl;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    if (DENSE) {
        linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    } else {
        linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3 ::PoseMatrixType>>();
    }
}