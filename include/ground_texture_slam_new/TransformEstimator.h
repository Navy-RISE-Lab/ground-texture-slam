#if !defined(GROUND_TEXTURE_SLAM_TRANSFORM_ESTIMATOR_H_)
#define GROUND_TEXTURE_SLAM_TRANSFORM_ESTIMATOR_H_

#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/slam/expressions.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace ground_texture_slam_new {
/**
 * @brief Estimate transforms between two images using robust M-Estimators.
 *
 * This follows the method described in [Look Ma, No
 * RANSAC](https://gtsam.org/2019/09/20/robust-noise-model.html) to estimate a
 * 2D transpose between a given set of matched points. It also calculates a
 * covariance matrix for the parameters (so a 3x3 matrix).
 *
 */
class TransformEstimator {
 public:
  /**
   * @brief The different types of estimator functions used in the model.
   *
   * See
   * [Look Ma, No RANSAC](https://gtsam.org/2019/09/20/robust-noise-model.html)
   * for graphs of each function.
   *
   */
  enum class Type {
    /// @brief Use [Huber loss
    /// function](https://en.wikipedia.org/wiki/Huber_loss).
    HUBER,
    /// @brief Use Cauchy loss function.
    CAUCHY,
    /// @brief Use Geman-McClure loss function.
    GEMAN_MCCLURE
  };

  /**
   * @brief The customizable options for the @ref TransformEstimator class.
   *
   */
  struct Options {
    /// @brief Which estimator type to use.
    Type type = Type::HUBER;
    /// @brief  The weight value for the estimator.
    double weight = 1.345;  // NOLINT okay for defaults
    /**
     * @brief The standard deviation of the point measurements passed in during
     * the estimation.
     *
     * This standard deviation is assumed to be the same for both the X and Y
     * components of the measurement.
     */
    double measurement_sigma = 1.0;  // NOLINT okay for defaults
  };

  /**
   * @brief Construct a new TransformEstimator object
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam_new.TransformEstimator(
   *     options: ground_texture_slam_new.TransformEstimator.Options
   * )
   * @endcode
   *
   * @param options The list of customizations to use.
   * @throws std::invalid_argument thrown if the weight or measurement_sigma is
   * less than or equal to zero.
   */
  explicit TransformEstimator(Options options);

  /**
   * @brief Given two sets of matched keypoints, estimate a 2D transform between
   * them.
   *
   * This will use the M-Estimator function specified at construction to
   * estimate a 2D transform between the points. It uses GTSAM's expression
   * graphs. The points in each set should be in the same relative frame of
   * reference. For example, both in the robot's frame of reference at the time
   * the data was taken.
   *
   * @note This method does not have a direct Python equivalent.
   *
   * @param points1 A list of matched keypoints from the first frame.
   * @param points2 A list of matched keypoints from the second frame. This
   * vector should have the same size as points1.
   * @return A tuple containing the best 2D pose estimate and the covariance.
   * The pose is the pose that transforms points from the first list to the
   * second. The covariance is a 3x3 covariance matrix for the transform
   * parameters (x, y, and theta).
   * @throw std::invalid_argument Thrown if the lengths of the points don't
   * match or they are empty vectors.
   */
  // NOLINTNEXTLINE(modernize-use-nodiscard) Okay to discard, no state change.
  auto estimateTransform(const std::vector<gtsam::Point2>& points1,
                         const std::vector<gtsam::Point2>& points2) const
      -> std::pair<gtsam::Pose2, gtsam::Matrix33>;

  /**
   * @brief Given two sets of matched keypoints, estimate a 2D transform between
   * them.
   *
   * @note This is an overloaded method that provides pure Eigen arguments and
   * return types. There is a slight additional overhead with this
   * implementation versus the other one.
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam_new.TransformEstimator.estimate_transform(
   *     points1: numpy.ndarray[numpy.float64[m, 2]],
   *     points2: numpy.ndarray[numpy.float64[m, 2]]
   * ) -> Tuple[
   *     numpy.ndarray[numpy.float64[3, 1]],
   *     numpy.ndarray[numpy.float64[3, 3]]
   * ]
   * @endcode
   *
   * @param points1 A matrix of matched keypoints from the first frame.
   * @param points2 A matrix of matched keypoints from the second frame. If the
   * first parameter is size Nx2, this should also be size Nx2.
   * @return A tuple containing the best 2D pose estimate and the covariance.
   * The pose is the pose that transforms the points from the first frame to the
   * second and is stored as a vector of form [x, y, theta]. The covariance is a
   * 3x3 covariance matrix for the transform parameters (x, y, theta).
   * @throw std::invalid_argument Thrown if the number of points are not the
   * same or if there are no points provided.
   */
  // NOLINTNEXTLINE(modernize-use-nodiscard) Okay to discard, no state change.
  auto estimateTransform(const Eigen::MatrixX2d& points1,
                         const Eigen::MatrixX2d& points2) const
      -> std::pair<Eigen::Vector3d, Eigen::Matrix3d>;

 private:
  /// The pointer holding the specific estimator function.
  gtsam::noiseModel::Robust::shared_ptr noise_model;
};
}  // namespace ground_texture_slam_new

#endif  // GROUND_TEXTURE_SLAM_TRANSFORM_ESTIMATOR_H_