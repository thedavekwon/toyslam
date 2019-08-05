//
// Created by Do Hyung Kwon on 7/24/19.
//

#include "bundleAdjustment.h"

// from g2o BA Demo

class Sample {
public:
    static int uniform(int from, int to);

    static double uniform();

    static double gaussian(double sigma);
};

static double uniform_rand(double lowerBndr, double upperBndr) {
    return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma) {
    double x, y, r2;
    do {
        x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        r2 = x * x + y * y;
    } while (r2 > 1.0 || r2 == 0.0);
    return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int Sample::uniform(int from, int to) {
    return static_cast<int>(uniform_rand(from, to));
}

double Sample::uniform() {
    return uniform_rand(0., 1.);
}

double Sample::gaussian(double sigma) {
    return gauss_rand(0., sigma);
}

int sampleBA() {
    std::cout << "PIXEL_NOISE: " << PIXEL_NOISE << std::endl;
    std::cout << "OUTLIER_RATIO: " << OUTLIER_RATIO << std::endl;
    std::cout << "ROBUST_KERNEL: " << ROBUST_KERNEL << std::endl;
    std::cout << "STRUCTURE_ONLY: " << STRUCTURE_ONLY << std::endl;
    std::cout << "DENSE: " << DENSE << std::endl;

    // optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    // BlockSolver<BlockSolverTraits<6, 3>> operates on the blocks of Hessian Matrix
    // Hessian Matrix: square matrix of second-order partial derivatives of a scalar-valued function, or scalar field.
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    if (DENSE) {
        linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    } else {
        linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    }

    // solver
    // Levenberg–Marquardt algorithm: damped least-squares method to solve non-linear least sqaure problems
    // generally for least-squares curving fitting problem
    auto *solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );
    optimizer.setAlgorithm(solver);

    // input
    std::vector<Eigen::Vector3d> true_points;
    for (int i = 0; i < 500; i++) {
        true_points.emplace_back((Sample::uniform() - 0.5) * 3, Sample::uniform() - 0.5, Sample::uniform() + 3);
    }

    // camera calibration matrix
    // K = [[f_x s x_o]
    //      [0 f_y y_0]
    //      [0   0  1]]
    double focal_length = 1000.;
    Eigen::Vector2d principal_point(320., 240.);
    std::vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat>> true_poses;
    auto *cam_params = new g2o::CameraParameters(focal_length, principal_point, 0.);
    cam_params->setId(0);
    if (!optimizer.addParameter(cam_params)) assert(false);

    // add camera poses
    int vertex_id = 0;
    for (int i = 0; i < 15; i++) {
        // translation of the pose
        Eigen::Vector3d trans(i * 0.04 - 1., 0, 0);
        // Rotation of the pose
        Eigen::Quaterniond q;
        q.setIdentity();
        g2o::SE3Quat pose(q, trans);
        // SE3 Vertex Exponential Map
        // Represent robot poses in SE3 Space

        // tangent space: vector space with the same dimension as the number of degrees of freedom of the group transformation
        // exponential map: converts any elements of the tangent space exactly into a transformation in a group
        auto *v_sec3 = new g2o::VertexSE3Expmap();
        v_sec3->setId(vertex_id);
        // set fixed during the optimization
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

    for (int i = 0; i < true_points.size(); i++) {
        // VertexSBAPointXYZ: represent 3-D point
        auto *v_p = new g2o::VertexSBAPointXYZ();
        v_p->setId(point_id);
        // set marginalized during the optimization
        v_p->setMarginalized(true);
        // add noise to the measurement of landmarks
        v_p->setEstimate(true_points.at(i) + Eigen::Vector3d(Sample::gaussian(1),
                                                             Sample::gaussian(1),
                                                             Sample::gaussian(1)));
        int num_obs = 0;
        for (auto &true_pose : true_poses) {
            Eigen::Vector2d z = cam_params->cam_map(true_pose.map(true_points.at(i)));
            if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480) num_obs++;
        }
        if (num_obs >= 2) {
            optimizer.addVertex(v_p);
            bool inlier = true;
            // the edges connecting map point vertex and corresponding keyframe vertex are added
            for (int j = 0; j < true_poses.size(); j++) {
                Eigen::Vector2d z = cam_params->cam_map(true_poses.at(j).map(true_points.at(i)));
                if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480) {
                    double sam = Sample::uniform();
                    if (sam < OUTLIER_RATIO) {
                        z = Eigen::Vector2d(Sample::uniform(0, 640),
                                            Sample::uniform(0, 480));
                        inlier = false;
                    }
                    // add noise
                    z += Eigen::Vector2d(Sample::gaussian(PIXEL_NOISE),
                                         Sample::gaussian(PIXEL_NOISE));
                    // EdgeProjectXYZ2UV: observations of 3D points in camera image plane
                    // For each edge, two vertices connecting should be specified.
                    // An information matrix should be given, which represents how reliable the measurement is.
                    // More Reliable Larger the information Matrix is.
                    auto *e = new g2o::EdgeProjectXYZ2UV();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(v_p));
                    e->setVertex(1,
                                 dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertices().find(j)->second));
                    e->setMeasurement(z);
                    // information Matrix
                    e->information() = Eigen::Matrix2d::Identity();
                    if (ROBUST_KERNEL) {
                        auto *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                    }
                    e->setParameterId(0, 0);
                    optimizer.addEdge(e);
                }
            }

            if (inlier) {
                inliers.insert(point_id);
                Eigen::Vector3d diff = v_p->estimate() - true_points.at(i);
                sum_diff2 += diff.dot(diff);
            }

            pointid_2_trueid.insert(std::make_pair(point_id, i));
            point_id++;
            point_num++;
        }
    }
    std::cout << std::endl;
    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    if (STRUCTURE_ONLY) {
        g2o::StructureOnlySolver<3> structure_only_ba;
        std::cout << "Performing structure only BA:" << std::endl;
        g2o::OptimizableGraph::VertexContainer points;
        for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin();
             it != optimizer.vertices().end(); it++) {
            g2o::OptimizableGraph::Vertex *v = static_cast<g2o::OptimizableGraph::Vertex *>(it->second);
            if (v->dimension() == 3) points.push_back(v);
        }
        structure_only_ba.calc(points, 10);
    }
    std::cout << std::endl;
    std::cout << "Performing Full BA:" << std::endl;
    optimizer.optimize(10);
    std::cout << std::endl;
    std::cout << "Point error before optimisation (inliers only): " << std::sqrt(sum_diff2 / inliers.size())
              << std::endl;
    point_num = 0;
    sum_diff2 = 0;

    for (auto & it : pointid_2_trueid) {
        auto v_it = optimizer.vertices().find(it.first);
        if (v_it == optimizer.vertices().end()) {
            std::cerr << "Vertex " << it.first << " not in graph!" << std::endl;
            exit(-1);
        }

        auto *v_p = dynamic_cast<g2o::VertexSBAPointXYZ *>(v_it->second);
        if (v_p == nullptr) {
            std::cerr << "Vertex " << it.first << "is not a Point XYZ!" << std::endl;
            exit(-1);
        }

        Eigen::Vector3d diff = v_p->estimate() - true_points[it.second];
        if (inliers.find(it.first) == inliers.end()) continue;
        sum_diff2 += diff.dot(diff);
        point_num++;
    }
    std::cout << "Point error after optimisation (inliers only): " << std::sqrt(sum_diff2 / inliers.size())
              << std::endl;
    std::cout << std::endl;
}

int bundleAdjustment3d2d(const std::vector<cv::Point3f> &points_3d,
                     const std::vector<cv::Point2f> &points_2d,
                     const cv::Mat &K,
                     cv::Mat &R,
                     cv::Mat &t) {
    // optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    // BlockSolver<BlockSolverTraits<6, 3>> operates on the blocks of Hessian Matrix
    // Hessian Matrix: square matrix of second-order partial derivatives of a scalar-valued function, or scalar field.
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    if (DENSE) {
        linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    } else {
        linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    }

    // solver
    // Levenberg–Marquardt algorithm: damped least-squares method to solve non-linear least sqaure problems
    // generally for least-squares curving fitting problem
    auto *solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );
    optimizer.setAlgorithm(solver);

    auto *cam_params = new g2o::CameraParameters(
            K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0
    );
    cam_params->setId(0);
    if (!optimizer.addParameter(cam_params)) assert(false);

    // add camera poses
    auto *pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat <<
          R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
          R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
           R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    pose->setId(0);
    pose->setEstimate(
            g2o::SE3Quat(
                    R_mat, Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))
            )
    );
    optimizer.addVertex(pose);


    // add map points
    int point_id = 1;
    for (const auto &p : points_3d) {
        auto *point = new g2o::VertexSBAPointXYZ();
        point->setId(point_id++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }

    int edge_id = 1;
    for (const auto &p : points_2d) {
        auto *edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(edge_id);
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *> (optimizer.vertex(edge_id)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        edge_id++;
    }

    optimizer.initializeOptimization();
    optimizer.setVerbose(true);
    optimizer.optimize(OPTIMIZE_COUNT);

    auto T = Eigen::Isometry3d(pose->estimate()).matrix();

    std::cout << T << std::endl;

    t = (cv::Mat_<float>(3, 1) << T(0, 3), T(1, 3), T(2, 3));
    R = (cv::Mat_<float>(3, 3) <<
            T(0, 0), T(0, 1), T(0, 2),
            T(1, 0), T(1, 1), T(1, 2),
            T(2, 0), T(2, 1), T(2, 2));
}