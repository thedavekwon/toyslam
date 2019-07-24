//
// Created by dodo on 7/24/19.
//

#include "bundleAdjustment.h"

// from g2o demo

class Sample {
public:
    static int uniform(int from, int to);
    static double uniform();
    static double gaussian(double sigma);
};

static double uniform_rand(double lowerBndr, double upperBndr){
    return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma){
    double x, y, r2;
    do {
        x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        r2 = x * x + y * y;
    } while (r2 > 1.0 || r2 == 0.0);
    return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int Sample::uniform(int from, int to){
    return static_cast<int>(uniform_rand(from, to));
}

double Sample::uniform(){
    return uniform_rand(0., 1.);
}

double Sample::gaussian(double sigma){
    return gauss_rand(0., sigma);
}

int BA() {
    std::cout << "PIXEL_NOISE: " <<  PIXEL_NOISE << std::endl;
    std::cout << "OUTLIER_RATIO: " << OUTLIER_RATIO<<  std::endl;
    std::cout << "ROBUST_KERNEL: " << ROBUST_KERNEL << std::endl;
    std::cout << "STRUCTURE_ONLY: " << STRUCTURE_ONLY<< std::endl;
    std::cout << "DENSE: "<<  DENSE << std::endl;

    // optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    // BlockSolver<BlockSolverTraits<6, 3>> operates on the blocks of Hessian Matrix
    // Hessian Matrix: square matrix of second-order partial derivatives of a scalar-valued function, or scalar field.
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    if (DENSE) {
        linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    } else {
        linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3 ::PoseMatrixType>>();
    }

    // solver
    // Levenberg–Marquardt algorithm: damped least-squares method to solve non-linear least sqaure problems
    // generally for least-squares curving fitting problem
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );
    optimizer.setAlgorithm(solver);

    // input
    std::vector<Eigen::Vector3d> true_points;
    for (int i = 0; i < 500; i++) {
        true_points.emplace_back((Sample::uniform()-0.5)*3, Sample::uniform()-0.5, Sample::uniform()+3);
    }

    // camera information
    double focal_length = 1000.;
    Eigen::Vector2d principal_point(320., 240.);
    std::vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> true_poses;
    auto* cam_params = new g2o::CameraParameters(focal_length, principal_point, 0.);
    cam_params->setId(0);
    if (!optimizer.addParameter(cam_params)) assert(false);

    // add poses
    int vertex_id = 0;
    for (int i = 0; i < 15; i++) {
        Eigen::Vector3d trans(i*0.04-1., 0, 0);
        Eigen::Quaterniond q;
        q.setIdentity();
        g2o::SE3Quat pose(q, trans);
        auto* v_sec3 = new g2o::VertexSE3Expmap();
        v_sec3->setId(vertex_id);
        if (i < 2) v_sec3->setFixed(true);
        v_sec3->setEstimate(pose);
        optimizer.addVertex(v_sec3);
        true_poses.push_back(pose);
        vertex_id++;
    }

    // add map points
    int point_id = vertex_id;
    int point_num = 0;
    double sum_diff2 = 0;

    std::cout << std::endl;
    std::unordered_map<int, int> pointid_2_trueid;
    std::unordered_set<int> inliers;

    for (auto & true_point : true_points) {
        auto* v_p = new g2o::VertexSBAPointXYZ();
        v_p->setId(point_id);
        v_p->setMarginalized(true);
        v_p->setEstimate(true_point + Eigen::Vector3d(Sample::gaussian(1),
                                                      Sample::gaussian(1),
                                                      Sample::gaussian(1)));
        int num_obs = 0;
        for (auto & true_pose : true_poses) {
            Eigen::Vector2d z = cam_params->cam_map(true_pose.map(true_point));
            if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480) num_obs++;
        }

        if (num_obs >= 2) {
            optimizer.addVertex(v_p);
            bool inlier = true;
            for (auto & true_pose: true_poses) {
                Eigen::Vector2d z = cam_params->cam_map(true_pose.map(true_point));
                if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480) {
                    double sam = Sample::uniform();
                    if (sam < OUTLIER_RATIO) {
                        z = Eigen::Vector2d(Sample::uniform(0, 640),
                                            Sample::uniform(0, 480));
                        inlier = false;
                    }
                    z += Eigen::Vector2d(Sample::gaussian(PIXEL_NOISE),
                                         Sample::gaussian(PIXEL_NOISE));
                }
            }
        }



    }
}