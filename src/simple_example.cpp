#include <iostream>

#include "GroundTextureSLAM.h"

/**
 * @brief Create a series of fake images. This is just a rectangle translating
 * across the image.
 *
 * @return std::vector<Eigen::Matrix<uint8_t, -1, -1>> The images in series.
 */
auto createImages() -> std::vector<Eigen::Matrix<uint8_t, -1, -1>> {
  std::vector<Eigen::Matrix<uint8_t, -1, -1>> images;
  for (size_t i = 0; i < 10; ++i) {
    Eigen::Matrix<uint8_t, -1, -1> image(600, 800);
    image.setZero();
    // Add the image as it moves to the right in increments of 10.
    image.block<60, 100>(200, i * 10).setConstant(255);
    images.push_back(image);
  }
  return images;
}

/// @brief Build a vocabulary tree out of random noise descriptors and save it
/// locally for later use by the SLAM system.
void createVocabulary() {
  std::cout << "Building vocabulary from random descriptors..." << std::endl;
  std::vector<Eigen::Matrix<uint8_t, -1, -1>> all_descriptors;
  for (size_t i = 0; i < 100; ++i) {
    all_descriptors.push_back(Eigen::Matrix<uint8_t, -1, -1>::Random(500, 32));
  }
  // The defaults are fine, since this is just an example. But they could be set
  // here.
  ground_texture_slam::BagOfWords::VocabOptions vocab_options;
  vocab_options.descriptors = all_descriptors;
  ground_texture_slam::BagOfWords bag_of_words(vocab_options);
  bag_of_words.saveVocabulary("example_vocab.bow");
  std::cout << "\tDone!" << std::endl;
}

auto main() -> int {
  // Build a vocabulary to use later.
  createVocabulary();
  // Set all the parameters of this fake SLAM system. The important ones
  // are the camera intrinsic matrix and pose, and the vocabulary for bag of
  // words.
  std::cout << "Loading system..." << std::endl;
  ground_texture_slam::GroundTextureSLAM::Options options;
  options.bag_of_words_options.vocab_file = "example_vocab.bow";
  options.keypoint_matcher_options.match_threshold = 0.6;
  // Normally, you want a sliding window so that successive images don't get
  // compared for loop closure. However, I am specifically allowing that here
  // since there are only a few images.
  options.sliding_window = 0;
  Eigen::Matrix3d camera_matrix;
  camera_matrix << 50.0, 0.0, 400.0, 0.0, 50.0, 300.0, 0.0, 0.0, 1.0;
  options.image_parser_options.camera_intrinsic_matrix = camera_matrix;
  Eigen::Matrix4d camera_pose;
  camera_pose << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.25,
      0.0, 0.0, 0.0, 1.0;
  options.image_parser_options.camera_pose = camera_pose;
  // Get the images "captured" by the robot.
  // Also, create some fake start pose info. This doesn't matter so set it to
  // the origin. You could also set it to a known start pose to align with any
  // data you are comparing against.
  // I am using Eigen inputs, but GTSAM/OpenCV inputs are also okay.
  auto images = createImages();
  Eigen::Vector3d start_pose = Eigen::Vector3d::Identity();
  Eigen::Matrix3d start_covariance = 1e-9 * Eigen::Matrix3d::Identity();
  ground_texture_slam::GroundTextureSLAM system(options, images[0], start_pose,
                                                start_covariance);
  std::cout << "Adding images..." << std::endl;
  // Now add each image. This would be done as images are received.
  for (size_t i = 1; i < images.size(); ++i) {
    system.insertMeasurement(images[i]);
  }
  // Once done, get the optimized poses. This can be done incrementally after
  // each image as well. The results probably won't be any good, since this is a
  // bunch of random images. But it illustrates the point.
  std::cout << "Results:" << std::endl;
  std::vector<gtsam::Pose2> pose_estimates = system.getPoseEstimates();
  for (auto &&pose : pose_estimates) {
    std::cout << pose.x() << ", " << pose.y() << ", " << pose.theta()
              << std::endl;
  }
}