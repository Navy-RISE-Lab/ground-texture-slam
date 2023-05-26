#include "GroundTextureSLAM.h"

#if defined(BUILD_PYTHON)
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#endif  // BUILD_PYTHON

namespace ground_texture_slam_new {
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
GroundTextureSLAM::GroundTextureSLAM(const Options& options,
                                     const cv::Mat& start_image,
                                     const gtsam::Pose2& start_pose,
                                     const gtsam::Matrix33& start_covariance) {

  setNumericOptions(options);
  createComponents(options);
  previous_estimated_transform = gtsam::Pose2::identity();
  createGraph(start_image, start_pose, start_covariance);
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
GroundTextureSLAM::GroundTextureSLAM(
    const Options& options, const Eigen::Matrix<uint8_t, -1, -1>& start_image,
    const Eigen::Vector3d& start_pose,
    const Eigen::Matrix3d& start_covariance) {
  
  
  setNumericOptions(options);
  createComponents(options);
  previous_estimated_transform = gtsam::Pose2::identity();
  cv::Mat start_image_cv(start_image.rows(), start_image.cols(), CV_8U);
  cv::eigen2cv(start_image, start_image_cv);
  gtsam::Pose2 start_pose_gtsam(start_pose(0), start_pose(1), start_pose(2));
  // Covariance doesn't need converted, since it is a typedef of Eigen::Matrix3d
  // already.
  createGraph(start_image_cv, start_pose_gtsam, start_covariance);
}

auto GroundTextureSLAM::getPoseEstimates() -> std::vector<gtsam::Pose2> {
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, pose_estimates);
  gtsam::Values values = optimizer.optimize();
  std::vector<gtsam::Pose2> results;
  // NOLINTNEXTLINE(modernize-loop-convert) Old style because of cast.
  for (size_t i = 0; i < values.size(); ++i) {
    results.push_back(values.at<gtsam::Pose2>(i));
  }
  return results;
}

auto GroundTextureSLAM::getPoseEstimatesMatrix() -> Eigen::MatrixX3d {
  std::vector<gtsam::Pose2> poses_vector = getPoseEstimates();
  Eigen::MatrixX3d poses(poses_vector.size(), 3);
  for (size_t i = 0; i < poses_vector.size(); ++i) {
    poses(i, 0) = poses_vector[i].x();
    poses(i, 1) = poses_vector[i].y();
    poses(i, 2) = poses_vector[i].theta();
  }
  return poses;
}

void GroundTextureSLAM::insertMeasurement(const cv::Mat& image) {
  // std::cout << "changing is valid!!!!!!!!!!!!" << std::endl;
  auto processed_image = image_parser->parseImage(image);
  descriptors.push_back(std::get<2>(processed_image));
  keypoints.push_back(std::get<0>(processed_image));
  size_t current_index = descriptors.size() - 1;
  localOdometry(current_index);
  std::cout << "Warning: loop closing detection not performing!" << std::endl;
  // loopClosure(current_index);
}

void GroundTextureSLAM::insertMeasurement(
    const Eigen::Matrix<uint8_t, -1, -1>& image) {
  // std::cout << "changing is valid!!!!!!!!!!!!" << std::endl;
  cv::Mat image_cv(image.rows(), image.cols(), CV_8U);
  cv::eigen2cv(image, image_cv);
  insertMeasurement(image_cv);
}

void GroundTextureSLAM::createComponents(const Options& options) {
  bag_of_words =
      std::make_shared<BagOfWords>(BagOfWords(options.bag_of_words_options));
  image_parser =
      std::make_shared<ImageParser>(ImageParser(options.image_parser_options));
  keypoint_matcher = std::make_shared<KeypointMatcher>(
      KeypointMatcher(options.keypoint_matcher_options));
  transform_estimator = std::make_shared<TransformEstimator>(
      TransformEstimator(options.transform_estimator_options));
}

void GroundTextureSLAM::createGraph(const cv::Mat& start_image,
                                    const gtsam::Pose2& start_pose,
                                    const gtsam::Matrix33& start_covariance) {
  descriptors.clear();
  keypoints.clear();
  graph = gtsam::NonlinearFactorGraph();
  pose_estimates.clear();
  auto parsed_image = image_parser->parseImage(start_image);
  auto noise_model = gtsam::noiseModel::Gaussian::Covariance(start_covariance);
  graph.addPrior(/*key=*/0, start_pose, noise_model);
  pose_estimates.insert(/*j=*/0, start_pose);
  keypoints.push_back(std::get<0>(parsed_image));
  descriptors.push_back(std::get<2>(parsed_image));
}

void GroundTextureSLAM::localOdometry(size_t current_index) {
  auto current_keypoints = keypoints[current_index];
  auto current_descriptors = descriptors[current_index];
  size_t previous_index = current_index - 1;
  auto previous_keypoints = keypoints[previous_index];
  auto previous_descriptors = descriptors[previous_index];
  gtsam::Pose2 estimated_transform;
  gtsam::Matrix33 estimated_covariance;
  try {
    std::vector<gtsam::Point2> current_matched_keypoints;
    std::vector<gtsam::Point2> previous_matched_keypoints;
    std::tie(current_matched_keypoints, previous_matched_keypoints) =
        keypoint_matcher->findMatchedKeypoints(
            current_keypoints, current_descriptors, previous_keypoints,
            previous_descriptors);
    std::tie(estimated_transform, estimated_covariance) =
        transform_estimator->estimateTransform(previous_matched_keypoints,
                                               current_matched_keypoints);
  } catch (const std::exception& e) {
    estimated_transform = previous_estimated_transform;
    estimated_covariance = gtsam::Matrix33::Identity() * fallback_variance;
  }
  auto estimated_noise_model =
      gtsam::noiseModel::Gaussian::Covariance(estimated_covariance);
  auto estimated_previous_pose =
      pose_estimates.at<gtsam::Pose2>(previous_index);
  gtsam::Pose2 estimated_current_pose =
      estimated_previous_pose.compose(estimated_transform);
  graph.add(gtsam::BetweenFactor<gtsam::Pose2>(previous_index, current_index,
                                               estimated_transform,
                                               estimated_noise_model));
  pose_estimates.insert(current_index, estimated_current_pose);
  previous_estimated_transform = estimated_transform;
}

void GroundTextureSLAM::loopClosure(size_t current_index) {
  auto current_keypoints = keypoints[current_index];
  auto current_descriptors = descriptors[current_index];
  if (current_index >= sliding_window) {
    bag_of_words->insertToDatabase(descriptors[current_index - sliding_window]);
  }
  std::map<unsigned int, double> query_results =
      bag_of_words->queryDatabase(current_descriptors);
  std::cout << "need to query " << query_results.size() << " frames" << std::endl;
  // Check each candidate pair for loop closure
  for (auto&& query_result : query_results) {
    // Only continue if the bow score is high enough.
    if (query_result.second < bag_of_words_threshold) {
      continue;
    }
    auto candidate_index = static_cast<size_t>(query_result.first);
    auto candidate_keypoints = keypoints[candidate_index];
    auto candidate_descriptors = descriptors[candidate_index];
    std::vector<gtsam::Point2> current_matched_keypoints;
    std::vector<gtsam::Point2> candidate_matched_keypoints;
    std::tie(current_matched_keypoints, candidate_matched_keypoints) =
        keypoint_matcher->findMatchedKeypoints(
            current_keypoints, current_descriptors, candidate_keypoints,
            candidate_descriptors);
    // Only continue if the match count is high enough.
    if (current_matched_keypoints.size() < keypoint_match_threshold) {
      continue;
    }
    try {
      gtsam::Pose2 estimated_transform;
      gtsam::Matrix33 estimated_covariance;
      std::tie(estimated_transform, estimated_covariance) =
          transform_estimator->estimateTransform(current_matched_keypoints,
                                                 candidate_matched_keypoints);
      double covariance_score =
          log10(estimated_covariance.eigenvalues().real().maxCoeff());
      if (covariance_score > covariance_threshold) {
        continue;
      }
      auto estimated_noise_model =
          gtsam::noiseModel::Gaussian::Covariance(estimated_covariance);
      graph.add(gtsam::BetweenFactor(current_index, candidate_index,
                                     estimated_transform,
                                     estimated_noise_model));

    } catch (const std::exception& e) {
      // We don't care if it errors, that just means an invalid loop closure.
    }
  }
}

void GroundTextureSLAM::setNumericOptions(const Options& options) {
  // Bag of words must be 0 or more.
  if (options.bag_of_words_threshold < 0.0) {
    throw std::invalid_argument(
        "Bag of Words threshold must be between 0 and 1, inclusive!");
  }
  // Fallback covariance must be greater than zero.
  if (options.fallback_variance <= 0.0) {
    throw std::invalid_argument("Fallback variance must be above zero!");
  }
  bag_of_words_threshold = options.bag_of_words_threshold;
  covariance_threshold = options.covariance_threshold;
  keypoint_match_threshold = options.keypoint_match_threshold;
  fallback_variance = options.fallback_variance;
  sliding_window = options.sliding_window;
}

#if defined(BUILD_PYTHON)
// GCOVR_EXCL_START
// NOLINTNEXTLINE(google-runtime-references) PyBind preferred signature.
void pybindGroundTextureSLAM(pybind11::module_& module) {
  pybind11::class_<GroundTextureSLAM, std::shared_ptr<GroundTextureSLAM>>
      gtslam(module, /*name=*/"GroundTextureSLAM");
  pybind11::class_<GroundTextureSLAM::Options,
                   std::shared_ptr<GroundTextureSLAM::Options>>
      options(gtslam, /*name=*/"Options");
  options.def(pybind11::init<>());
  options.def_readwrite(/*name=*/"bag_of_words_threshold",
                        &GroundTextureSLAM::Options::bag_of_words_threshold);
  options.def_readwrite(/*name=*/"keypoint_match_threshold",
                        &GroundTextureSLAM::Options::keypoint_match_threshold);
  options.def_readwrite(/*name=*/"covariance_threshold",
                        &GroundTextureSLAM::Options::covariance_threshold);
  options.def_readwrite(/*name=*/"sliding_window",
                        &GroundTextureSLAM::Options::sliding_window);
  options.def_readwrite(/*name=*/"fallback_variance",
                        &GroundTextureSLAM::Options::fallback_variance);
  options.def_readwrite(/*name=*/"bag_of_words_options",
                        &GroundTextureSLAM::Options::bag_of_words_options);
  options.def_readwrite(/*name=*/"image_parser_options",
                        &GroundTextureSLAM::Options::image_parser_options);
  options.def_readwrite(/*name=*/"keypoint_matcher_options",
                        &GroundTextureSLAM::Options::keypoint_matcher_options);
  options.def_readwrite(
      /*name=*/"transform_estimator_options",
      &GroundTextureSLAM::Options::transform_estimator_options);
  gtslam.def(pybind11::init<GroundTextureSLAM::Options,
                            const Eigen::Matrix<uint8_t, -1, -1>&,
                            const Eigen::Vector3d&, const Eigen::Matrix3d&>(),
             pybind11::arg(/*name=*/"options"),
             pybind11::arg(/*name=*/"start_image"),
             pybind11::arg(/*name=*/"start_pose"),
             pybind11::arg(/*name=*/"start_covariance"));
  gtslam.def(/*name_=*/"get_pose_estimates_matrix",
             &GroundTextureSLAM::getPoseEstimatesMatrix);
  gtslam.def(/*name_=*/"insert_measurement",
             pybind11::overload_cast<const Eigen::Matrix<uint8_t, -1, -1>&>(
                 &GroundTextureSLAM::insertMeasurement),
             pybind11::arg(/*name=*/"image"));
}
// GCOVR_EXCL_STOP
#endif  // BUILD_PYTHON
}  // namespace ground_texture_slam_new
