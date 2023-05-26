// NOLINTNEXTLINE Skip test macros
#include <gtest/gtest.h>

#include "TransformEstimator.h"

/**
 * @test Ensure the constructor rejects invalid measurement standard deviations.
 *
 */
// NOLINTNEXTLINE Skip test macros
TEST(TransformEstimator, ConstructRejectBadMeasurementSigma) {
  ground_texture_slam_new::TransformEstimator::Options options;
  options.measurement_sigma = 0.0;
  // NOLINTNEXTLINE Skip test macros
  ASSERT_THROW(ground_texture_slam_new::TransformEstimator estimator(options),
               std::invalid_argument);
}

/// @test Ensure the constructor rejects invalid weights.
// NOLINTNEXTLINE Skip test macros
TEST(TransformEstimator, ConstructRejectBadWeight) {
  ground_texture_slam_new::TransformEstimator::Options options;
  options.weight = 0.0;
  // NOLINTNEXTLINE Skip test macros
  ASSERT_THROW(ground_texture_slam_new::TransformEstimator estimator(options),
               std::invalid_argument);
}

/**
 * @test Ensure the estimation returns a reasonable result when given points
 * with a known transform.
 *
 */
// NOLINTNEXTLINE Skip test macros
TEST(TransformEstimator, EstimateCorrect) {
  double ACTUAL_X = 1.0;
  double ACTUAL_Y = 3.0;
  double ACTUAL_T = 0.0;
  gtsam::Pose2 actual_pose(ACTUAL_X, ACTUAL_Y, ACTUAL_T);
  Eigen::Vector3d expected_result;
  expected_result << ACTUAL_X, ACTUAL_Y, ACTUAL_T;
  Eigen::Matrix<double, 3, 2> points1 = Eigen::Matrix<double, 3, 2>::Zero();
  Eigen::Matrix<double, 3, 2> points2 = Eigen::Matrix<double, 3, 2>::Zero();
  for (size_t i = 0; i < 3; ++i) {
    gtsam::Point2 point1(i, i);
    gtsam::Point2 point2 = actual_pose.transformTo(point1);
    points1(i, 0) = point1.x();
    points1(i, 1) = point1.y();
    points2(i, 0) = point2.x();
    points2(i, 1) = point2.y();
  }
  // Test with each type of estimator model
  ground_texture_slam_new::TransformEstimator::Options options;
  for (size_t i = 0; i < 3; ++i) {
    auto type = ground_texture_slam_new::TransformEstimator::Type::HUBER;
    switch (i) {
      case 0:
        type = ground_texture_slam_new::TransformEstimator::Type::GEMAN_MCCLURE;
        break;
      case 1:
        type = ground_texture_slam_new::TransformEstimator::Type::CAUCHY;
        break;
      case 2:
        type = ground_texture_slam_new::TransformEstimator::Type::HUBER;
        break;
    }
    options.type = type;
    ground_texture_slam_new::TransformEstimator estimator(options);
    auto results = estimator.estimateTransform(points1, points2);
    // The zero doesn't play well with float comparisons, so bump up the
    // tolerance.
    ASSERT_TRUE(expected_result.isApprox(results.first, 1e-7));
  }
}

/// @test Ensure the estimation rejects empty lists.
TEST(TransformEstimator, EstimateEmptyVectors) {
  ground_texture_slam_new::TransformEstimator::Options options;
  ground_texture_slam_new::TransformEstimator estimator(options);
  std::vector<gtsam::Point2> empty_points;
  std::vector<gtsam::Point2> full_points;
  full_points.push_back(gtsam::Point2(1.0, 0.0));
  ASSERT_THROW(estimator.estimateTransform(empty_points, full_points),
               std::invalid_argument);
  ASSERT_THROW(estimator.estimateTransform(full_points, empty_points),
               std::invalid_argument);
  ASSERT_THROW(estimator.estimateTransform(empty_points, empty_points),
               std::invalid_argument);
}

/**
 * @test Ensure the estimation rejects requests where the number of points in
 * each set don't match.
 *
 */
TEST(TransformEstimator, EstimateRejectMismatchedPoints) {
  ground_texture_slam_new::TransformEstimator::Options options;
  ground_texture_slam_new::TransformEstimator estimator(options);
  std::vector<gtsam::Point2> points1;
  std::vector<gtsam::Point2> points2;
  for (size_t i = 0; i < 3; i++) {
    gtsam::Point2 point(i, i);
    points1.push_back(point);
    points2.push_back(point);
  }
  points1.push_back(gtsam::Point2(5, 5));
  ASSERT_THROW(estimator.estimateTransform(points1, points2),
               std::invalid_argument);
}