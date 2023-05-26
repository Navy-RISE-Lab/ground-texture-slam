#include <gtest/gtest.h>

#include "ImageParser.h"

/**
 * @brief Test that the image parser works as expected.
 *
 * These test values create an image with a single rectangle. The parser should
 * find the 4 corners. Since the rectangle is at a known point, the expected
 * projected values can be calculated.
 *
 */
TEST(ImageParser, ParseImageCorrect) {
  // Create the image manually.
  Eigen::Matrix<uint8_t, -1, -1> image(800, 600);
  image.setZero();
  image.block<60, 100>(40, 400).setConstant(255);
  Eigen::Matrix<float, 4, 2> expected_keypoint_pixels;
  expected_keypoint_pixels << 400.8, 40.8, 498.0, 40.8, 400.8, 98.4, 498.0,
      98.4;
  Eigen::Matrix<double, 4, 2> expected_keypoint_meters;
  expected_keypoint_meters << 1.296, -0.004, 1.296, -0.49, 1.008, -0.004, 1.008,
      -0.49;
  // Set options
  ground_texture_slam_new::ImageParser::Options options;
  options.levels = 2;
  Eigen::Matrix3d camera_matrix;
  camera_matrix << 50.0, 0.0, 400.0, 0.0, 50.0, 300.0, 0.0, 0.0, 1.0;
  options.camera_intrinsic_matrix = camera_matrix;
  Eigen::Matrix4d camera_pose;
  camera_pose << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.25,
      0.0, 0.0, 0.0, 1.0;
  options.camera_pose = camera_pose;
  ground_texture_slam_new::ImageParser parser(options);
  auto result = parser.parseImage(image);
  // Give a wider tolerance, since the pixel math is more exact than what I have
  // above.
  ASSERT_TRUE(std::get<1>(result).isApprox(expected_keypoint_pixels, 1e-5));
  ASSERT_TRUE(std::get<0>(result).isApprox(expected_keypoint_meters, 1e-5));
  // Make sure the same works with FAST_SCORE
  options.use_harris_score = false;
  parser = ground_texture_slam_new::ImageParser(options);
  result = parser.parseImage(image);
  ASSERT_TRUE(std::get<1>(result).isApprox(expected_keypoint_pixels, 1e-5));
  ASSERT_TRUE(std::get<0>(result).isApprox(expected_keypoint_meters, 1e-5));
}

/// @test Ensure only 8-bit unsigned images are used.
TEST(ImageParser, RejectWrongImageType) {
  ground_texture_slam_new::ImageParser::Options options;
  options.camera_intrinsic_matrix = Eigen::Matrix3d::Identity();
  options.camera_pose = Eigen::Matrix4d::Identity();
  ground_texture_slam_new::ImageParser parser(options);
  cv::Mat image = cv::Mat::zeros(400, 600, CV_32F);
  ASSERT_THROW(parser.parseImage(image), std::invalid_argument);
}