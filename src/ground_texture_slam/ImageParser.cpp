#include "ImageParser.h"

#if defined(BUILD_PYTHON)
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#endif  // BUILD_PYTHON

namespace ground_texture_slam {
ImageParser::ImageParser(Options options) {
  cv::ORB::ScoreType score_type = cv::ORB::ScoreType::HARRIS_SCORE;
  if (!options.use_harris_score) {
    score_type = cv::ORB::ScoreType::FAST_SCORE;
  }
  detector = cv::ORB::create(options.features, options.scale_factor,
                             options.levels, options.edge_threshold,
                             options.first_level, options.WTA_K, score_type,
                             options.patch_size, options.fast_threshold);
  Eigen::Affine3d image_2_camera_eigen = Eigen::Affine3d::Identity();
  image_2_camera_eigen.rotate(
      Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitX()));
  image_2_camera_eigen.rotate(
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitY()));
  image_2_camera_transform = gtsam::Pose3(image_2_camera_eigen.matrix());
  auto camera_matrix = gtsam::Cal3_S2(options.camera_intrinsic_matrix(0, 0),
                                      options.camera_intrinsic_matrix(1, 1),
                                      options.camera_intrinsic_matrix(0, 1),
                                      options.camera_intrinsic_matrix(0, 2),
                                      options.camera_intrinsic_matrix(1, 2));
  auto camera_pose = gtsam::Pose3(options.camera_pose);
  camera = gtsam::PinholeCameraCal3_S2(
      camera_pose.compose(image_2_camera_transform), camera_matrix);
}

auto ImageParser::parseImage(const cv::Mat& image) const
    -> std::tuple<std::vector<gtsam::Point2>, std::vector<cv::KeyPoint>,
                  cv::Mat> {
  // Verify the correct type first.
  if (image.type() != CV_8U) {
    throw std::invalid_argument("Image must be 8-bit unsigned!");
  }
  // Get the keypoints and descriptors.
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
  // Convert the keypoints into the robot's frame with GTSAM types. This could
  // be done as an Eigen matrix, which would be faster. This relies less on
  // manual transform math though. It can be parallelized at least.
  std::vector<gtsam::Point2> keypoints_meters;
  for (auto&& keypoint : keypoints) {
    gtsam::Point2 point_pixel(keypoint.pt.x, keypoint.pt.y);
    gtsam::Point3 point_robot =
        camera.backproject(point_pixel, camera.pose().translation().z());
    gtsam::Point2 point_robot_2d(point_robot.x(), point_robot.y());
    keypoints_meters.push_back(point_robot_2d);
  }
  return std::make_tuple(keypoints_meters, keypoints, descriptors);
}

auto ImageParser::parseImage(const Eigen::Matrix<uint8_t, -1, -1>& image) const
    -> std::tuple<Eigen::MatrixX2d, Eigen::MatrixX2f,
                  Eigen::Matrix<uint8_t, -1, -1>> {
  cv::Mat image_cv(image.rows(), image.cols(), CV_8U);
  cv::eigen2cv(image, image_cv);
  auto results = parseImage(image_cv);
  size_t num_keypoints = std::get<0>(results).size();
  Eigen::MatrixX2d keypoints_meter(num_keypoints, 2);
  Eigen::MatrixX2f keypoints_pixels(num_keypoints, 2);
  for (size_t i = 0; i < num_keypoints; ++i) {
    keypoints_meter(i, 0) = std::get<0>(results)[i].x();
    keypoints_meter(i, 1) = std::get<0>(results)[i].y();
    keypoints_pixels(i, 0) = std::get<1>(results)[i].pt.x;
    keypoints_pixels(i, 1) = std::get<1>(results)[i].pt.y;
  }
  Eigen::Matrix<uint8_t, -1, -1> descriptors(num_keypoints,
                                             std::get<2>(results).cols);
  cv::cv2eigen(std::get<2>(results), descriptors);
  return std::make_tuple(keypoints_meter, keypoints_pixels, descriptors);
}

#if defined(BUILD_PYTHON)
// GCOVR_EXCL_START
// NOLINTNEXTLINE(google-runtime-references) PyBind preferred signature.
void pybindImageParser(pybind11::module_& module) {
  pybind11::class_<ImageParser, std::shared_ptr<ImageParser>> image_parser(
      module, /*name=*/"ImageParser");
  pybind11::class_<ImageParser::Options> options(image_parser,
                                                 /*name=*/"Options");
  options.def(pybind11::init<>());
  options.def_readwrite(/*name=*/"camera_pose",
                        &ImageParser::Options::camera_pose);
  options.def_readwrite(/*name=*/"camera_intrinsic_matrix",
                        &ImageParser::Options::camera_intrinsic_matrix);
  options.def_readwrite(/*name=*/"features", &ImageParser::Options::features);
  options.def_readwrite(/*name=*/"scale_factor",
                        &ImageParser::Options::scale_factor);
  options.def_readwrite(/*name=*/"levels", &ImageParser::Options::levels);
  options.def_readwrite(/*name=*/"edge_threshold",
                        &ImageParser::Options::edge_threshold);
  options.def_readwrite(/*name=*/"first_level",
                        &ImageParser::Options::first_level);
  options.def_readwrite(/*name=*/"WTA_K", &ImageParser::Options::WTA_K);
  options.def_readwrite(/*name=*/"use_harris_score",
                        &ImageParser::Options::use_harris_score);
  options.def_readwrite(/*name=*/"patch_size",
                        &ImageParser::Options::patch_size);
  options.def_readwrite(/*name=*/"fast_threshold",
                        &ImageParser::Options::fast_threshold);
  image_parser.def(pybind11::init<ImageParser::Options>(),
                   pybind11::arg(/*name=*/"options"));
  image_parser.def(
      /*name_=*/"parse_image",
      pybind11::overload_cast<const Eigen::Matrix<uint8_t, -1, -1>&>(
          &ImageParser::parseImage, pybind11::const_),
      pybind11::arg(/*name=*/"image"));
}
// GCOVR_EXCL_STOP
#endif  // BUILD_PYTHON
}  // namespace ground_texture_slam
