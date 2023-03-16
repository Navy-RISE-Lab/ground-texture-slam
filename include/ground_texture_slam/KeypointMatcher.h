#if !defined(GROUND_TEXTURE_SLAM_KEYPOINT_MATCHER_H_)
#define GROUND_TEXTURE_SLAM_KEYPOINT_MATCHER_H_

#include <gtsam/geometry/Point2.h>

#include <memory>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ground_texture_slam {
/**
 * @brief A matcher to determine which keypoints from two sets are the same.
 *
 * This is done by comparing the descriptors using a FLANN-based matcher. It
 * finds the nearest 2 descriptors and then compares the distance (in descriptor
 * space) to identify if this is a reliable match.
 *
 */
class KeypointMatcher {
 public:
  /// The customizable options for the class.
  struct Options {
    /// The required difference between 1st and 2nd closest neighbors.
    double match_threshold = 0.7;  // NOLINT default value
    /// An optional random seed to set, since FLANN is non-deterministic.
    std::optional<int> seed = std::nullopt;
  };
  /**
   * @brief Construct a new KeypointMatcher object
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam.GroundTextureSLAM.KeypointMatcher(
   *     options: ground_texture_slam.KeypointMatcher.Options
   * )
   * @endcode
   *
   * @param options The customization for this object.
   * @throws std::invalid_argument thrown if the threshold is outside of the
   * range [0, 1].
   */
  explicit KeypointMatcher(Options options);

  /**
   * @brief Given the keypoint/descriptor sets from two images, find the subset
   * that is matched on each image.
   *
   * This uses a K-nearest-neighbor approach to compare the distance between
   * descriptors. A set of keypoints is considered a match if the distance to
   * the best match is a certain proportion of the distance to the second
   * closest match. This proportion is set by threshold.
   *
   * @note This method does not have a direct Python equivalent.
   *
   * @param keypoints1 The keypoints from the first image.
   * @param descriptors1 The descriptors for the first image.
   * @param keypoints2 The keypoints from the second image.
   * @param descriptors2 The descriptors for the second image.
   * @return std::pair<std::vector<gtsam::Point2>, std::vector<gtsam::Point2>>
   * Matching pairs of points. These points are the same in both images.
   * @throws std::invalid_argument thrown if the lists of keypoints and
   * associated descriptors are not the same lengths (e.g. if points1.size() !=
   * descriptors1.size()).
   */
  auto findMatchedKeypoints(const std::vector<gtsam::Point2>& keypoints1,
                            const cv::Mat& descriptors1,
                            const std::vector<gtsam::Point2>& keypoints2,
                            const cv::Mat& descriptors2)
      -> std::pair<std::vector<gtsam::Point2>, std::vector<gtsam::Point2>>;

  /**
   * @brief Given the keypoint/descriptor sets from two images, find the subset
   * that is matched on each image.
   *
   * This uses a K-nearest-neighbor approach to compare the distance between
   * descriptors. A set of keypoints is considered a match if the distance to
   * the best match is a certain proportion of the distance to the second
   * closest match. This proportion is set by threshold.
   *
   * @note This is an overloaded method for Python binding. It adds additional
   * overhead for data conversions.
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam.KeypointMatcher.find_matched_keypoints(
   *     keypoints1: numpy.ndarray[numpy.float64[m, 2]],
   *     descriptors1: numpy.ndarray[numpy.float32[m, n]],
   *     keypoints2: numpy.ndarray[numpy.float64[m, 2]],
   *     descriptors2: numpy.ndarray[numpy.float32[m, n]]
   * ) -> Tuple[
   *     numpy.ndarray[numpy.float64[m, 2]],
   *     numpy.ndarray[numpy.float64[m, 2]]
   * ]
   * @endcode
   *
   * @param keypoints1 The keypoints from the first image.
   * @param descriptors1 The descriptors for the first image.
   * @param keypoints2 The keypoints from the second image.
   * @param descriptors2 The descriptors for the second image.
   * @return std::pair<Eigen::MatrixX2d, Eigen::MatrixX2d> Matching pairs of
   * points. These points are the same in both images.
   * @throws std::invalid_argument thrown if the lists of keypoints and
   * associated descriptors are not the same lengths (e.g. if points1.rows() !=
   * descriptors1.rows()).
   */
  auto findMatchedKeypoints(const Eigen::MatrixX2d& keypoints1,
                            const Eigen::MatrixXf& descriptors1,
                            const Eigen::MatrixX2d& keypoints2,
                            const Eigen::MatrixXf& descriptors2)
      -> std::pair<Eigen::MatrixX2d, Eigen::MatrixX2d>;

 private:
  /// @brief The required proportional difference between 1st and 2nd closest
  /// neighbors.
  double match_threshold;
  /// @brief The FLANN matcher to use.
  cv::Ptr<cv::FlannBasedMatcher> matcher;
};
}  // namespace ground_texture_slam

#endif  // GROUND_TEXTURE_SLAM_KEYPOINT_MATCHER_H_
