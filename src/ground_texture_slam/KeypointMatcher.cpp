#include "KeypointMatcher.h"

#if defined(BUILD_PYTHON)
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif  // BUILD_PYTHON

namespace ground_texture_slam {
KeypointMatcher::KeypointMatcher(Options options) {
  // Verify the threshold is a proportion.
  if (options.match_threshold < 0.0 || options.match_threshold > 1.0) {
    throw std::invalid_argument(
        "Threshold must be a proportion between 0 and 1 (inclusive)!");
  }
  // Set the RNG seed, if requested.
  if (options.seed.has_value()) {
    cv::setRNGSeed(options.seed.value());
  }

  this->match_threshold = options.match_threshold;
  matcher = cv::FlannBasedMatcher::create();
}

auto KeypointMatcher::findMatchedKeypoints(
    const std::vector<gtsam::Point2>& keypoints1, const cv::Mat& descriptors1,
    const std::vector<gtsam::Point2>& keypoints2, const cv::Mat& descriptors2)
    -> std::pair<std::vector<gtsam::Point2>, std::vector<gtsam::Point2>> {
  // Verify all the sets are the same size.
  if (keypoints1.size() != descriptors1.rows) {
    throw std::invalid_argument(
        "Keypoints and descriptors from the first image must be the same "
        "size!");
  }
  if (keypoints2.size() != descriptors2.rows) {
    throw std::invalid_argument(
        "Keypoints and descriptors from the second image must be the same "
        "size!");
  }
  // If either set is zero, this is a trivial problem to solve.
  if (keypoints1.empty() || keypoints2.empty()) {
    std::vector<gtsam::Point2> empty_vector;
    return std::make_pair(empty_vector, empty_vector);
  }
  // Ensure the descriptors are 32 bit floats, since that is the only one FLANN
  // takes.
  cv::Mat descriptors1_32f;
  if (descriptors1.type() != CV_32F) {
    descriptors1.convertTo(descriptors1_32f, CV_32F);
  } else {
    descriptors1_32f = descriptors1;
  }
  cv::Mat descriptors2_32f;
  if (descriptors2.type() != CV_32F) {
    descriptors2.convertTo(descriptors2_32f, CV_32F);
  } else {
    descriptors2_32f = descriptors2;
  }
  std::vector<std::vector<cv::DMatch>> matches;
  matcher->knnMatch(descriptors1_32f, descriptors2_32f, matches, /*k=*/2);
  std::vector<gtsam::Point2> points1_matched;
  std::vector<gtsam::Point2> points2_matched;
  // I don't think this can be parallelized, unless you can guarantee both
  // push_backs occur atomically.
  for (auto&& match : matches) {
    if (match[0].distance < match_threshold * match[1].distance) {
      int points1_index = match[0].queryIdx;
      int points2_index = match[0].trainIdx;
      points1_matched.push_back(keypoints1[points1_index]);
      points2_matched.push_back(keypoints2[points2_index]);
    }
  }
  return std::make_pair(points1_matched, points2_matched);
}

auto KeypointMatcher::findMatchedKeypoints(const Eigen::MatrixX2d& keypoints1,
                                           const Eigen::MatrixXf& descriptors1,
                                           const Eigen::MatrixX2d& keypoints2,
                                           const Eigen::MatrixXf& descriptors2)
    -> std::pair<Eigen::MatrixX2d, Eigen::MatrixX2d> {
  std::vector<gtsam::Point2> keypoints1_gtsam;
  for (size_t i = 0; i < keypoints1.rows(); i++) {
    gtsam::Point2 point(keypoints1(i, 0), keypoints1(i, 1));
    keypoints1_gtsam.push_back(point);
  }
  cv::Mat descriptors1_cv(descriptors1.rows(), descriptors1.cols(), CV_32F);
  cv::eigen2cv(descriptors1, descriptors1_cv);
  std::vector<gtsam::Point2> keypoints2_gtsam;
  for (size_t i = 0; i < keypoints2.rows(); i++) {
    gtsam::Point2 point(keypoints2(i, 0), keypoints2(i, 1));
    keypoints2_gtsam.push_back(point);
  }
  cv::Mat descriptors2_cv(descriptors2.rows(), descriptors2.cols(), CV_32F);
  cv::eigen2cv(descriptors2, descriptors2_cv);
  auto result = findMatchedKeypoints(keypoints1_gtsam, descriptors1_cv,
                                     keypoints2_gtsam, descriptors2_cv);
  Eigen::MatrixX2d points1(result.first.size(), 2);
  for (size_t i = 0; i < result.first.size(); i++) {
    points1(i, 0) = result.first[i].x();
    points1(i, 1) = result.first[i].y();
  }
  Eigen::MatrixX2d points2(result.second.size(), 2);
  for (size_t i = 0; i < result.second.size(); i++) {
    points2(i, 0) = result.second[i].x();
    points2(i, 1) = result.second[i].y();
  }
  return std::make_pair(points1, points2);
}

#if defined(BUILD_PYTHON)
// GCOVR_EXCL_START
// NOLINTNEXTLINE(google-runtime-references) PyBind preferred signature.
void pybindKeypointMatcher(pybind11::module_& module) {
  pybind11::class_<KeypointMatcher, std::shared_ptr<KeypointMatcher>>
      keypoint_matcher(module, /*name=*/"KeypointMatcher");
  pybind11::class_<KeypointMatcher::Options> options(keypoint_matcher,
                                                     /*name=*/"Options");
  options.def(pybind11::init<>());
  options.def_readwrite(/*name=*/"match_threshold",
                        &KeypointMatcher::Options::match_threshold);
  options.def_readwrite(/*name=*/"seed", &KeypointMatcher::Options::seed);
  keypoint_matcher.def(pybind11::init<KeypointMatcher::Options>(),
                       pybind11::arg(/*name=*/"options"));
  keypoint_matcher.def(
      /*name_=*/"find_matched_keypoints",
      pybind11::overload_cast<const Eigen::MatrixX2d&, const Eigen::MatrixXf&,
                              const Eigen::MatrixX2d&, const Eigen::MatrixXf&>(
          &KeypointMatcher::findMatchedKeypoints),
      pybind11::arg(/*name=*/"keypoints1"),
      pybind11::arg(/*name=*/"descriptors1"),
      pybind11::arg(/*name=*/"keypoints2"),
      pybind11::arg(/*name=*/"descriptors2"));
}
// GCOVR_EXCL_STOP
#endif  // BUILD_PYTHON
}  // namespace ground_texture_slam
