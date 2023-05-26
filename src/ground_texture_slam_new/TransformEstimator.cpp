#include "TransformEstimator.h"

#if defined(BUILD_PYTHON)
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#endif  // BUILD_PYTHON

namespace ground_texture_slam_new {
TransformEstimator::TransformEstimator(Options options) {
  // Verify the weight and standard deviation are greater than zero.
  if (options.weight <= 0.0) {
    throw std::invalid_argument("Estimator weight must be greater than zero!");
  }
  if (options.measurement_sigma <= 0.0) {
    throw std::invalid_argument("Measurement sigma must be greater than zero!");
  }
  auto measurement_noise =
      gtsam::noiseModel::Isotropic::Sigma(/*dim=*/2, options.measurement_sigma);
  switch (options.type) {
    case Type::HUBER:
    default:
      noise_model = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Huber::Create(options.weight),
          measurement_noise);
      break;
    case Type::CAUCHY:
      noise_model = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Cauchy::Create(options.weight),
          measurement_noise);
      break;
    case Type::GEMAN_MCCLURE:
      noise_model = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::GemanMcClure::Create(options.weight),
          measurement_noise);
      break;
  }
}

auto TransformEstimator::estimateTransform(
    const std::vector<gtsam::Point2>& points1,
    const std::vector<gtsam::Point2>& points2) const
    -> std::pair<gtsam::Pose2, gtsam::Matrix33> {
  // Verify the points lists aren't empty and match in size.
  if (points1.empty() || points2.empty()) {
    throw std::invalid_argument("Point lists must not be empty!");
  }
  if (points1.size() != points2.size()) {
    throw std::invalid_argument(
        "Point lists must be the same size as they are matched points!");
  }
  // Initialize the graph
  gtsam::Values initial_guess;
  initial_guess.insert(/*j=*/0, gtsam::Pose2::identity());
  gtsam::ExpressionFactorGraph graph;
  // This is the pose that is estimated to fit the functions. The underscore is
  // GTSAM's convention for expressions vs defined values.
  gtsam::Pose2_ estimated_transform_expression(0);
  for (size_t i = 0; i < points1.size(); i++) {
    gtsam::Point2_ predicted =
        gtsam::transformTo(estimated_transform_expression, points1[i]);
    graph.addExpressionFactor(predicted, points2[i], noise_model);
  }
  // Do the estimation and pull out the results.
  gtsam::Values results =
      gtsam::LevenbergMarquardtOptimizer(graph, initial_guess).optimize();
  auto estimated_transform = results.at<gtsam::Pose2>(/*j=*/0);
  gtsam::Marginals marginals(graph, results);
  gtsam::Matrix33 covariance = marginals.marginalCovariance(/*variable=*/0);
  return std::make_pair(estimated_transform, covariance);
}

auto TransformEstimator::estimateTransform(
    const Eigen::MatrixX2d& points1, const Eigen::MatrixX2d& points2) const
    -> std::pair<Eigen::Vector3d, Eigen::Matrix3d> {
  // Convert each argument before running.
  std::vector<gtsam::Point2> points1_vector;
  std::vector<gtsam::Point2> points2_vector;
  for (size_t i = 0; i < points1.rows(); i++) {
    gtsam::Point2 point(points1(i, 0), points1(i, 1));
    points1_vector.push_back(point);
  }
  for (size_t i = 0; i < points2.rows(); i++) {
    gtsam::Point2 point(points2(i, 0), points2(i, 1));
    points2_vector.push_back(point);
  }
  gtsam::Pose2 pose;
  gtsam::Matrix33 covariance = gtsam::Matrix33::Identity();
  std::tie(pose, covariance) =
      estimateTransform(points1_vector, points2_vector);
  // We only need to convert the pose back, because the Matrix33 is already an
  // Eigen matrix.
  Eigen::Vector3d pose_eigen = Eigen::Vector3d::Identity();
  pose_eigen(0) = pose.x();
  pose_eigen(1) = pose.y();
  pose_eigen(2) = pose.theta();
  return std::make_pair(pose_eigen, covariance);
}

#if defined(BUILD_PYTHON)
// GCOVR_EXCL_START
// NOLINTNEXTLINE(google-runtime-references) PyBind preferred signature.
void pybindTransformEstimator(pybind11::module_& module) {
  pybind11::class_<TransformEstimator, std::shared_ptr<TransformEstimator>>
      transform_estimator(module, /*name=*/"TransformEstimator");
  pybind11::enum_<TransformEstimator::Type>(transform_estimator,
                                            /*name=*/"Type")
      .value(/*name=*/"HUBER", TransformEstimator::Type::HUBER)
      .value(/*name=*/"CAUCHY", TransformEstimator::Type::CAUCHY)
      .value(/*name=*/"GEMAN_MCCLURE", TransformEstimator::Type::GEMAN_MCCLURE)
      .export_values();
  pybind11::class_<TransformEstimator::Options,
                   std::shared_ptr<TransformEstimator::Options>>
      options(transform_estimator, /*name=*/"Options");
  options.def(pybind11::init<>());
  options.def_readwrite(/*name=*/"type", &TransformEstimator::Options::type);
  options.def_readwrite(/*name=*/"weight",
                        &TransformEstimator::Options::weight);
  options.def_readwrite(/*name=*/"measurement_sigma",
                        &TransformEstimator::Options::measurement_sigma);
  transform_estimator.def(pybind11::init<TransformEstimator::Options>(),
                          pybind11::arg(/*name=*/"options"));
  transform_estimator.def(
      /*name_=*/"estimate_transform",
      pybind11::overload_cast<const Eigen::MatrixX2d&, const Eigen::MatrixX2d&>(
          &TransformEstimator::estimateTransform, pybind11::const_),
      pybind11::arg(/*name=*/"points1"), pybind11::arg(/*name=*/"points2"));
}
// GCOVR_EXCL_STOP
#endif  // BUILD_PYTHON
}  // namespace ground_texture_slam_new
