#if !defined(GROUND_TEXTURE_SLAM_IMAGE_PARSER_H_)
#define GROUND_TEXTURE_SLAM_IMAGE_PARSER_H_

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/SimpleCamera.h>

#include <Eigen/Core>
#include <exception>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

namespace ground_texture_slam_new {
/**
 * @brief Performs image processing related to keypoints.
 *
 * This class extracts keypoints and descriptors. It also projects them into the
 * ground plane.
 *
 */
class ImageParser {
 public:
  /// @brief The customization options for this class.
  struct Options {
    /**
     * @brief The pose of the camera, as measured from teh robot's frame.
     *
     * This is also the transform that projects from the camera's frame into the
     * robot's.
     *
     * @note The camera frame here refers to the usual robot-sensor orientation
     * of X out, Z up. This is not the camera convention of X along the top edge
     * of the image.
     *
     */
    Eigen::Matrix4d camera_pose = Eigen::Matrix4d::Identity();
    /// @brief The calibrated intrinsic matrix for the camera.
    Eigen::Matrix3d camera_intrinsic_matrix = Eigen::Matrix3d::Identity();
    /// @brief The max keypoints to detect.
    int features = 500;  // NOLINT Default value
    /// @brief See OpenCV::ORB::create for a description.
    float scale_factor = 1.2;  // NOLINT Default value
    /// @brief See OpenCV::ORB::create for a description.
    int levels = 8;  // NOLINT Default value
    /// @brief See OpenCV::ORB::create for a description.
    int edge_threshold = 31;  // NOLINT Default value
    /// @brief See OpenCV::ORB::create for a description.
    int first_level = 0;  // NOLINT Default value
    /// @brief See OpenCV::ORB::create for a description.
    int WTA_K = 2;  // NOLINT Default value
    /**
     * @brief See OpenCV::ORB::create for a description.
     *
     * This is a boolean version of their enum. If true,
     * ORB::ScoreType::HARRIS_SCORE will be used, otherwise
     * ORB::ScoreType::FAST_SCORE will be used.
     */
    bool use_harris_score = true;
    /// @brief See OpenCV::ORB::create for a description.
    int patch_size = 31;  // NOLINT Default value
    /// @brief See OpenCV::ORB::create for a description.
    int fast_threshold = 20;  // NOLINT Default value
  };

  /**
   * @brief Construct a new ImageParser object.
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam_new.ImageParser(
   *     options: ground_texture_slam_new.ImageParser.Options
   * )
   * @endcode
   *
   * @param options The customization to use for parsing.
   */
  explicit ImageParser(Options options);

  /**
   * @brief Run detection and description on a given image.
   *
   * @note This method does not have a direct Python equivalent.
   *
   * @param image The OpenCV image to parse. Must be type CV_8U.
   * @return Returns a tuple containing the list of keypoints, translated into
   * the robot's frame of reference; the original keypoints; and their
   * associated descriptors. The length of the keypoint vectors are guaranteed
   * to be the same length as the rows of the cv::Mat.
   * @throws std::invalid_argument thrown if the image is not a CV_8U type.
   */
  // NOLINTNEXTLINE(modernize-use-nodiscard) Okay to discard, no state change.
  auto parseImage(const cv::Mat& image) const
      -> std::tuple<std::vector<gtsam::Point2>, std::vector<cv::KeyPoint>,
                    cv::Mat>;

  /**
   * @brief Run detection and description on a given image.
   *
   * @note This is an overloaded method to provide convenient argument and
   * return types. It adds additional overhead for data conversions under the
   * hood.
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam_new.ImageParser.parse_image(
   *     image: numpy.ndarray[numpy.uint8[m, n]]
   * ) -> Tuple[
   *     numpy.ndarray[numpy.float64[m, 2]],
   *     numpy.ndarray[numpy.float32[m, 2]],
   *     numpy.ndarray[numpy.uint8[m, n]]
   * ]
   * @endcode
   *
   * @param image The image to parse
   * @return Returns a tuple containing the list of keypoints, translated into
   * the robot's frame of reference; the original keypoints; and their
   * associated descriptors. The number of rows are guaranteed to be the same in
   * each element.
   */
  // NOLINTNEXTLINE(modernize-use-nodiscard) Okay to discard, no state change.
  auto parseImage(const Eigen::Matrix<uint8_t, -1, -1>& image) const
      -> std::tuple<Eigen::MatrixX2d, Eigen::MatrixX2f,
                    Eigen::Matrix<uint8_t, -1, -1>>;

 private:
  /**
   * @brief The transform that converts 3D points from the camera frame to the
   * robot frame.
   *
   */
  gtsam::PinholeCameraCal3_S2 camera;
  /// @brief The keypoint detector and describer algorithm.
  cv::Ptr<cv::ORB> detector;
  /**
   * @brief The transform that converts 3D points from the image frame to camera
   * frame.
   *
   * This transform primarily consists of rotations so that the image frame (X
   * to the left, Z out) becomes aligned with the conventional frame (X out, Z
   * up).
   *
   */
  gtsam::Pose3 image_2_camera_transform;
};
}  // namespace ground_texture_slam_new

#endif  // GROUND_TEXTURE_SLAM_IMAGE_PARSER_H_
