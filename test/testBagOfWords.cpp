#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <vector>

#include "BagOfWords.h"

typedef Eigen::Matrix<uint8_t, -1, -1> MatrixU;

/**
 * @brief Create some fake vocabulary data for building the tree.
 *
 * This just uses a bunch of random matrices.
 *
 * @return std::vector<MatrixU> The list of descriptors.
 */
auto createVocabularyData() -> std::vector<MatrixU> {
  std::vector<MatrixU> result;
  for (size_t i = 0; i < 50; ++i) {
    result.push_back(MatrixU::Random(500, 32));
  }
  return result;
}

/**
 * @brief Create some predefined descriptor data for testing.
 *
 * These are all uniform values.
 *
 * @return std::vector<MatrixU> All the descriptors. Each element is as if it
 * came from one image.
 */
auto createDatabaseData() -> std::vector<MatrixU> {
  // Create the three descriptors.
  MatrixU descriptor0 = MatrixU::Random(500, 32);
  MatrixU descriptor1 = MatrixU::Random(500, 32);
  MatrixU descriptor2 = MatrixU::Random(500, 32);
  // Add them to the total vector.
  std::vector<MatrixU> all_descriptors;
  all_descriptors.push_back(descriptor0);
  all_descriptors.push_back(descriptor1);
  all_descriptors.push_back(descriptor2);
  return all_descriptors;
}

/// @test Test the function rejects matrices that are not CV_8U type.
TEST(BagOfWords, RejectBadType) {
  auto vocabulary = createVocabularyData();
  auto descriptors = createDatabaseData();
  ground_texture_slam::BagOfWords::VocabOptions vocab_options;
  vocab_options.descriptors = vocabulary;
  ground_texture_slam::BagOfWords bag_of_words(vocab_options);
  cv::Mat wrong_type = cv::Mat::zeros(500, 32, CV_32F);
  ASSERT_THROW(bag_of_words.insertToDatabase(wrong_type),
               std::invalid_argument);
  ASSERT_THROW(bag_of_words.queryDatabase(wrong_type), std::invalid_argument);
}

/// @test Ensure that the system throws an error if the vocabulary bag file
/// doesn't exist.
TEST(BagOfWords, RejectNonexistantVocab) {
  ground_texture_slam::BagOfWords::Options options;
  options.vocab_file = "fake_file.bow";
  ASSERT_THROW(ground_texture_slam::BagOfWords bow(options),
               std::invalid_argument);
}

/// @test Test that the scoring works as expected.
TEST(BagOfWords, ScoreCorrectly) {
  auto vocabulary = createVocabularyData();
  auto descriptors = createDatabaseData();
  size_t target_descriptor = 1;
  ground_texture_slam::BagOfWords::VocabOptions vocab_options;
  vocab_options.descriptors = vocabulary;
  // Iterate over each score and weight type. They should all produce the same
  // results.
  std::vector<ground_texture_slam::BagOfWords::WeightingType> all_weights = {
      ground_texture_slam::BagOfWords::WeightingType::BINARY,
      ground_texture_slam::BagOfWords::WeightingType::IDF,
      ground_texture_slam::BagOfWords::WeightingType::TF,
      ground_texture_slam::BagOfWords::WeightingType::TF_IDF};
  std::vector<ground_texture_slam::BagOfWords::ScoringType> all_scores = {
      ground_texture_slam::BagOfWords::ScoringType::BHATTACHARYYA,
      ground_texture_slam::BagOfWords::ScoringType::CHI_SQUARE,
      ground_texture_slam::BagOfWords::ScoringType::DOT_PRODUCT,
      ground_texture_slam::BagOfWords::ScoringType::KL,
      ground_texture_slam::BagOfWords::ScoringType::L1_NORM,
      ground_texture_slam::BagOfWords::ScoringType::L2_NORM};
  for (auto &&score : all_scores) {
    for (auto &&weight : all_weights) {
      vocab_options.weight = weight;
      vocab_options.scoring = score;
      ground_texture_slam::BagOfWords bag_of_words(vocab_options);
      // Add every descriptor to the database and score. The closest match
      // should be the same one.
      for (auto &&descriptor : descriptors) {
        bag_of_words.insertToDatabase(descriptor);
      }
      std::map<unsigned int, double> results =
          bag_of_words.queryDatabase(descriptors[target_descriptor]);
      ASSERT_EQ(results.size(), descriptors.size());
      // The identical result should always have the largest value, unless KL is
      // used, then it is the lowest.
      std::map<unsigned int, double>::iterator closest_match;
      switch (score) {
        case ground_texture_slam::BagOfWords::ScoringType::BHATTACHARYYA:
        case ground_texture_slam::BagOfWords::ScoringType::CHI_SQUARE:
        case ground_texture_slam::BagOfWords::ScoringType::DOT_PRODUCT:
        case ground_texture_slam::BagOfWords::ScoringType::L1_NORM:
        case ground_texture_slam::BagOfWords::ScoringType::L2_NORM:
        default:
          closest_match = std::max_element(
              results.begin(), results.end(),
              [](const std::pair<unsigned int, double> &a,
                 const std::pair<unsigned int, double> &b) -> bool {
                return a.second < b.second;
              });
          break;
        case ground_texture_slam::BagOfWords::ScoringType::KL:
          closest_match = std::min_element(
              results.begin(), results.end(),
              [](const std::pair<unsigned int, double> &a,
                 const std::pair<unsigned int, double> &b) -> bool {
                return a.second < b.second;
              });
          break;
      }
      ASSERT_EQ(closest_match->first, target_descriptor);
    }
  }
}

/// @test Ensure the ID values returned by the database are ordered integers.
TEST(BagOfWords, SequentialIndices) {
  auto vocabulary = createVocabularyData();
  auto descriptors = createDatabaseData();
  ground_texture_slam::BagOfWords::VocabOptions vocab_options;
  vocab_options.descriptors = vocabulary;
  ground_texture_slam::BagOfWords bag_of_words(vocab_options);
  // Make sure each ID matches.
  for (unsigned int i = 0; i < 50; ++i) {
    unsigned int id = bag_of_words.insertToDatabase(descriptors[0]);
    ASSERT_EQ(id, i);
  }
}