#if !defined(GROUND_TEXTURE_SLAM_GROUND_TEXTURE_SLAM_H_)
#define GROUND_TEXTURE_SLAM_GROUND_TEXTURE_SLAM_H_

#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <stdexcept>

#include "BagOfWords.h"
#include "ImageParser.h"
#include "KeypointMatcher.h"
#include "TransformEstimator.h"

/// @brief The namespace for the entire project.
namespace ground_texture_slam {
/**
 * @brief The top level class that performs the entire mapping pipeline.
 *
 * This class accepts images taken at various points and uses the SLAM algorithm
 * to estimate poses up to that point using only all images received to date.
 *
 */
class GroundTextureSLAM {
 public:
  /// @brief The customization options for the system.
  struct Options {
    // The order is modified to reduced excessive padding.
    /// @brief Options for keypoint detection, descriptions, and projection.
    ImageParser::Options image_parser_options;
    /// @brief The bag of words score that must be met or exceeded to be
    /// considered a possible loop closure.
    double bag_of_words_threshold = 0.0;
    /// @brief The covariance score that must not be exceeded to be considered a
    /// possible loop closure.
    double covariance_threshold = std::numeric_limits<double>::max();
    /// @brief The number of images that must be received prior to add an image
    /// into the bag of words database.
    size_t sliding_window = 10;  // NOLINT Default value
    /// @brief The variance to use when local odometry fails.
    double fallback_variance = 50.0;  // NOLINT Default value
    /// @brief Options for matching keypoints between images.
    KeypointMatcher::Options keypoint_matcher_options;
    /// @brief Options for estimating transforms between projected keypoints.
    TransformEstimator::Options transform_estimator_options;
    /// @brief Options for the bag of words algorithm.
    BagOfWords::Options bag_of_words_options;
    /// @brief The number of matched keypoints score that must be met or
    /// exceeded to be considered a possible loop closure.
    unsigned int keypoint_match_threshold = 0;
  };

  /**
   * @brief Construct a new GroundTextureSLAM object.
   *
   * @note This method does not have a direct Python equivalent.
   *
   * @param options The list of customizations for the system.
   * @param start_image An image to use at the starting pose.
   * @param start_pose A starting pose estimate. This is primarily to prevent
   * ill-posed results. Typically, this will be either zero or the estimated
   * starting pose of your data.
   * @param start_covariance The covariance associated with the starting
   * estimate.
   */
  GroundTextureSLAM(const Options& options, const cv::Mat& start_image,
                    const gtsam::Pose2& start_pose,
                    const gtsam::Matrix33& start_covariance);

  /**
   * @brief Construct a new GroundTextureSLAM object.
   *
   * @note This is an overloaded method for Python binding. It adds additional
   * overhead for data conversions.
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam.GroundTextureSLAM(
   *     options: ground_texture_slam.GroundTextureSLAM.Options,
   *     start_image: numpy.ndarray[numpy.unint8[m, n]],
   *     start_pose: numpy.ndarray[numpy.float64[3, 1]],
   *     start_covariance: numpy.ndarray[numpy.float64[3, 3]]
   * )
   * @endcode
   *
   * @param options The list of customizations for the system.
   * @param start_image An image to use at the starting pose.
   * @param start_pose A starting pose estimate. This is primarily to prevent
   * ill-posed results. Typically, this will be either zero or the estimated
   * starting pose of your data.
   * @param start_covariance The covariance associated with the starting
   * estimate.
   */
  GroundTextureSLAM(const Options& options,
                    const Eigen::Matrix<uint8_t, -1, -1>& start_image,
                    const Eigen::Vector3d& start_pose,
                    const Eigen::Matrix3d& start_covariance);

  /**
   * @brief Get the most up to date pose estimate.
   *
   * This will apply optimization and return all pose estimates associated with
   * each image. Updated estimates are not saved for use by the system.
   *
   * @note This method does not have a direct Python equivalent.
   *
   * @return std::vector<gtsam::Pose2> The vector of poses associated with each
   * image.
   */
  auto getPoseEstimates() -> std::vector<gtsam::Pose2>;

  /**
   * @brief Get the most up to date pose estimate.
   *
   * This will apply optimization and return all pose estimates associated with
   * each image. Updated estimates are not saved for use by the system.
   *
   * @note This is an overloaded method for Python binding. It adds additional
   * overhead for data conversions.
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam.GroundTextureSLAM.get_pose_estimates_matrix() ->
   * numpy.ndarray[numpy.float64[m, 3]]
   * @endcode
   *
   * @return Eigen::MatrixX3d The vector of poses associated with each
   * image.
   */
  auto getPoseEstimatesMatrix() -> Eigen::MatrixX3d;

  /**
   * @brief Add a new image observation to the system.
   *
   * This image is assumed undistorted.
   *
   * @note This method does not have a direct Python equivalent.
   *
   * @param image The image to add. Type must be CV_8U.
   */
  void insertMeasurement(const cv::Mat& image);

  /**
   * @brief Add a new image observation to the system.
   *
   * This image is assumed undistorted.
   *
   * @note This is an overloaded method for Python binding. It adds additional
   * overhead for data conversions.
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam.GroundTextureSLAM.insert_measurement(
   *     image: numpy.ndarray[numpy.uint8[m, n]]
   * ) -> None
   * @endcode
   *
   * @param image The image to add.
   */
  void insertMeasurement(const Eigen::Matrix<uint8_t, -1, -1>& image);

 private:
  /**
   * @brief Construct each major component of the class.
   *
   * @param options The customization options for everything, which contains the
   * options for the components.
   */
  void createComponents(const Options& options);

  /**
   * @brief Construct the nonlinear factor graph for the system.
   *
   * This initializes the graph and sets the start pose according to the
   * provided arguments.
   *
   * @param start_image An image to use at the starting pose.
   * @param start_pose A starting pose estimate. This is primarily to prevent
   * ill-posed results. Typically, this will be either zero or the estimated
   * starting pose of your data.
   * @param start_covariance The covariance associated with the starting
   * estimate.
   */
  void createGraph(const cv::Mat& start_image, const gtsam::Pose2& start_pose,
                   const gtsam::Matrix33& start_covariance);

  /**
   * @brief Perform local odometry estimation.
   *
   * This uses just the current image and previous image to estimate the
   * transform between them. If this estimation fails, the previous transform is
   * used instead. Results are added to the factor graph.
   *
   * @param current_index The index of the current image, used to find the right
   * keypoints and descriptors.
   */
  void localOdometry(size_t current_index);

  /**
   * @brief Perform loop closure.
   *
   * This attempts to identify all previous loop closures. It uses the 3
   * threshold values described in the paper to determine if a loop closure is
   * valid or not.
   *
   * @param current_index The index of the current image, used to find the right
   * keypoints and descriptors.
   */
  void loopClosure(size_t current_index);

  /**
   * @brief Validate and set the various options that are not major components.
   *
   * @param options The total list of options, which contains the threshold
   * values, sliding window, and fallback variance.
   * @throws std::invalid_argument thrown if the bag of words threshold is not
   * greater than or equal to 0 or the fallback covariance is not greater than
   * 0.
   */
  void setNumericOptions(const Options& options);

  /// @brief The component performing bag of words matching and scoring.
  std::shared_ptr<BagOfWords> bag_of_words;
  /// @brief The bag of words score that must be met or exceeded to be
  /// considered a possible loop closure.
  double bag_of_words_threshold;
  /// @brief The covariance score that must not be exceeded to be considered a
  /// possible loop closure.
  double covariance_threshold;
  /// @brief All descriptors of images previously seen.
  std::vector<cv::Mat> descriptors;
  /// @brief The variance to use when local odometry fails.
  double fallback_variance;
  /// @brief The factor graph for pose estimation.
  gtsam::NonlinearFactorGraph graph;
  /// @brief The component performing keypoint detection, descriptions, and
  /// projection.
  std::shared_ptr<ImageParser> image_parser;
  /// @brief All projected keypoints of images previously seen.
  std::vector<std::vector<gtsam::Point2>> keypoints;
  /// @brief The number of matched keypoints score that must be met or exceeded
  /// to be considered a possible loop closure.
  unsigned int keypoint_match_threshold;
  /// @brief The component matching keypoints between pairs of images.
  std::shared_ptr<KeypointMatcher> keypoint_matcher;
  /// @brief Initial estimates for poses associated with each image.
  gtsam::Values pose_estimates;
  /// @brief The last estimated transform, in case local odometry fails.
  gtsam::Pose2 previous_estimated_transform;
  /// @brief The number of images that must be received prior to add an image
  /// into the bag of words database.
  size_t sliding_window;
  /// @brief The component estimating the best 2D transform between matched
  /// projected keypoints.
  std::shared_ptr<TransformEstimator> transform_estimator;
};
}  // namespace ground_texture_slam

#endif  // GROUND_TEXTURE_SLAM_GROUND_TEXTURE_SLAM_H_
