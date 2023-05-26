#include <gtest/gtest.h>

#include "BagOfWords.h"
#include "GroundTextureSLAM.h"

/// @test Ensure the fallback variance is greater than zero.
TEST(GroundTextureSLAM, RejectBadFallbackVariance) {
  ground_texture_slam_new::GroundTextureSLAM::Options options;
  options.fallback_variance = 0.0;
  Eigen::Matrix<uint8_t, -1, -1> start_image_eigen =
      Eigen::Matrix<uint8_t, -1, -1>::Zero(400, 600);
  Eigen::Vector3d start_pose_eigen = Eigen::Vector3d::Zero();
  Eigen::Matrix3d start_covariance_eigen = Eigen::Matrix3d::Identity();
  ASSERT_THROW(
      ground_texture_slam_new::GroundTextureSLAM(
          options, start_image_eigen, start_pose_eigen, start_covariance_eigen),
      std::invalid_argument);
  options.fallback_variance = -0.00000001;
  ASSERT_THROW(
      ground_texture_slam_new::GroundTextureSLAM(
          options, start_image_eigen, start_pose_eigen, start_covariance_eigen),
      std::invalid_argument);
}

/**
 * @test Ensure the system throws errors for bad threshold values.
 *
 * Since we don't know which scoring method is used, the only constraint is that
 * the bag of words threshold is greater than or equal to zero.
 */
TEST(GroundTextureSLAM, RejectBadThreshold) {
  ground_texture_slam_new::GroundTextureSLAM::Options options;
  options.bag_of_words_threshold = -0.00000000001;
  Eigen::Matrix<uint8_t, -1, -1> start_image_eigen =
      Eigen::Matrix<uint8_t, -1, -1>::Zero(400, 600);
  Eigen::Vector3d start_pose_eigen = Eigen::Vector3d::Zero();
  Eigen::Matrix3d start_covariance_eigen = Eigen::Matrix3d::Identity();
  ASSERT_THROW(
      ground_texture_slam_new::GroundTextureSLAM(
          options, start_image_eigen, start_pose_eigen, start_covariance_eigen),
      std::invalid_argument);
  cv::Mat start_image_cv = cv::Mat::zeros(400, 600, CV_8U);
  gtsam::Pose2 start_pose_gtsam = gtsam::Pose2::identity();
  gtsam::Matrix33 start_covariance_gtsam = gtsam::Matrix33::Identity();
  ASSERT_THROW(
      ground_texture_slam_new::GroundTextureSLAM(
          options, start_image_cv, start_pose_gtsam, start_covariance_gtsam),
      std::invalid_argument);
}