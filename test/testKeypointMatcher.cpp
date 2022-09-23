#include <gtest/gtest.h>

#include <tuple>

#include "KeypointMatcher.h"

/**
 * @brief Create some trivial data.
 *
 * This creates two series of 3 points along the x=y line. The descriptors are
 * set such that the first and second points are too close to threshold as a
 * match for a threshold of 0.7.
 *
 * @return A pair of two sets of points and their descriptors.
 */
auto createKeypointMatcherData()
    -> std::tuple<Eigen::Matrix<double, 3, 2>, Eigen::Matrix<float, 3, 1>,
                  Eigen::Matrix<double, 3, 2>, Eigen::Matrix<float, 3, 1>> {
  Eigen::Matrix<double, 3, 2> keypoints1;
  keypoints1 << 0.0, 0.0, 1.0, 1.0, 2.0, 2.0;
  Eigen::Matrix<double, 3, 2> keypoints2;
  keypoints2 << 0.0, 0.0, 10.0, 10.0, 20.0, 20.0;
  Eigen::Matrix<float, 3, 1> descriptors1;
  descriptors1 << 255.0, 128.0, 0.0;
  Eigen::Matrix<float, 3, 1> descriptors2;
  descriptors2 << 200.0, 200.0, 0.0;
  return std::make_tuple(keypoints1, descriptors1, keypoints2, descriptors2);
}

/// @test Ensure the function can correctly convert to CV_32F.
TEST(KeypointMatcher, ConvertDescriptorType) {
  // Get the data, but manually convert everything to the desired datatypes.
  auto data = createKeypointMatcherData();
  // Convert the keypoints.
  std::vector<gtsam::Point2> keypoints1;
  std::vector<gtsam::Point2> keypoints2;
  for (size_t i = 0; i < std::get<0>(data).rows(); ++i) {
    double x = std::get<0>(data)(i, 0);
    double y = std::get<0>(data)(i, 1);
    keypoints1.push_back(gtsam::Point2(x, y));
    x = std::get<2>(data)(i, 0);
    y = std::get<2>(data)(i, 1);
    keypoints2.push_back(gtsam::Point2(x, y));
  }
  // Convert the descriptors of image 1.
  cv::Mat descriptors1_cv32f(std::get<1>(data).rows(), std::get<1>(data).cols(),
                             CV_32F);
  cv::Mat descriptors1_cv8u(descriptors1_cv32f.rows, descriptors1_cv32f.cols,
                            CV_8U);
  cv::eigen2cv(std::get<1>(data), descriptors1_cv32f);
  descriptors1_cv32f.convertTo(descriptors1_cv8u, CV_8U);
  // Convert the descriptors of image 2.
  cv::Mat descriptors2_cv32f(std::get<3>(data).rows(), std::get<3>(data).cols(),
                             CV_32F);
  cv::Mat descriptors2_cv8u(descriptors2_cv32f.rows, descriptors2_cv32f.cols,
                            CV_8U);
  cv::eigen2cv(std::get<3>(data), descriptors2_cv32f);
  descriptors2_cv32f.convertTo(descriptors2_cv8u, CV_8U);
  // Do the actual matching.
  ground_texture_slam::KeypointMatcher::Options options;
  options.match_threshold = 0.7;
  ground_texture_slam::KeypointMatcher matcher(options);
  auto matches = matcher.findMatchedKeypoints(keypoints1, descriptors1_cv8u,
                                              keypoints2, descriptors2_cv8u);
  // Make sure the conversion didn't warp the values somehow.
  ASSERT_EQ(matches.first.size(), 1);
  ASSERT_EQ(matches.second.size(), 1);
  gtsam::Point2 result = matches.first[0];
  gtsam::Point2 expected_result = keypoints1[2];
  ASSERT_TRUE(result.isApprox(expected_result));
  result = matches.second[0];
  expected_result = keypoints2[2];
  ASSERT_TRUE(result.isApprox(expected_result));
}

/// @test Ensure the matching works for a simple trivial setup.
TEST(KeypointMatcher, FindMatchedKeypointsCorrect) {
  auto data = createKeypointMatcherData();
  ground_texture_slam::KeypointMatcher::Options options;
  options.match_threshold = 0.7;
  ground_texture_slam::KeypointMatcher matcher(options);
  auto matches =
      matcher.findMatchedKeypoints(std::get<0>(data), std::get<1>(data),
                                   std::get<2>(data), std::get<3>(data));
  // The sets should actual match.
  ASSERT_EQ(matches.first.rows(), matches.second.rows());
  // The only match should be point 3.
  auto result = matches.first;
  auto expected_result = std::get<0>(data).row(2);
  ASSERT_TRUE(result.isApprox(expected_result));
  result = matches.second;
  expected_result = std::get<2>(data).row(2);
  ASSERT_TRUE(result.isApprox(expected_result));
}

/// @test Ensure thresholds are within bounds.
TEST(KeypointMatcher, RejectBadThreshold) {
  ground_texture_slam::KeypointMatcher::Options options;
  options.match_threshold = -0.00000001;
  EXPECT_THROW(ground_texture_slam::KeypointMatcher matcher(options),
               std::invalid_argument);
  options.match_threshold = 1.000000001;
  EXPECT_THROW(ground_texture_slam::KeypointMatcher matcher(options),
               std::invalid_argument);
}

/// @test Ensure errors thrown if keypoint and descriptor sizes don't match.
TEST(KeypointMatcher, RejectMismatchedSize) {
  Eigen::MatrixX2d keypoints1;
  ground_texture_slam::KeypointMatcher::Options options;
  ground_texture_slam::KeypointMatcher matcher(options);
  Eigen::MatrixX2d keypoints2;
  Eigen::MatrixXf descriptors1;
  Eigen::MatrixXf descriptors2;
  std::tie(keypoints1, descriptors1, keypoints2, descriptors2) =
      createKeypointMatcherData();
  Eigen::Matrix<double, 10, 2> wrong_size_matrix =
      Eigen::Matrix<double, 10, 2>::Zero();
  EXPECT_THROW(matcher.findMatchedKeypoints(wrong_size_matrix, descriptors1,
                                            keypoints2, descriptors2),
               std::invalid_argument);
  EXPECT_THROW(matcher.findMatchedKeypoints(keypoints1, descriptors1,
                                            wrong_size_matrix, descriptors2),
               std::invalid_argument);
}

/// @test Ensure setting the seed to the same value produces the same results.
TEST(KeypointMatcher, SeedWorks) {
  // Create some random descriptor data to use for matching.
  Eigen::MatrixX2d keypoints1 = Eigen::MatrixX2d::Random(300, 2);
  Eigen::MatrixXf descriptors1 = Eigen::MatrixXf::Random(300, 32);
  Eigen::MatrixX2d keypoints2 = Eigen::MatrixX2d::Random(300, 2);
  Eigen::MatrixXf descriptors2 = Eigen::MatrixXf::Random(300, 32);
  // Run the matching twice with the same seed. The results should be the same.
  // Do the construction both times, because the seed is set at construction.
  ground_texture_slam::KeypointMatcher::Options options;
  options.match_threshold = 0.99;
  options.seed = 0;
  ground_texture_slam::KeypointMatcher matcher1(options);
  auto result1 = matcher1.findMatchedKeypoints(keypoints1, descriptors1,
                                               keypoints2, descriptors2);
  ground_texture_slam::KeypointMatcher matcher2(options);
  auto result2 = matcher2.findMatchedKeypoints(keypoints1, descriptors1,
                                               keypoints2, descriptors2);
  ASSERT_EQ(std::get<0>(result1).rows(), std::get<0>(result2).rows());
  ASSERT_EQ(std::get<0>(result1).cols(), std::get<0>(result2).cols());
  ASSERT_TRUE(std::get<0>(result1).isApprox(std::get<0>(result2)));
  ASSERT_TRUE(std::get<1>(result1).isApprox(std::get<1>(result2)));
  // Running without the seed should produce different values.
  options.seed = std::nullopt;
  ground_texture_slam::KeypointMatcher matcher3(options);
  auto result3 = matcher3.findMatchedKeypoints(keypoints1, descriptors1,
                                               keypoints2, descriptors2);
  // Since this is random, they matrices might or might not be the same size.
  // However, isApprox throws an error if the matrices are different sizes. So
  // check both cases.
  if (std::get<0>(result1).rows() == std::get<0>(result3).rows() &&
      std::get<0>(result1).cols() == std::get<0>(result3).cols()) {
    ASSERT_FALSE(std::get<0>(result1).isApprox(std::get<0>(result3)));
    ASSERT_FALSE(std::get<1>(result1).isApprox(std::get<1>(result3)));
  }
  // If the rows or columns are different sizes, the test case passes by
  // definition. So no need to check anything.
}