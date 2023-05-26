#include "BagOfWords.h"
#include <iostream>
#include <fstream>

#if defined(BUILD_PYTHON)
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#endif  // BUILD_PYTHON

namespace ground_texture_slam_new {
BagOfWords::BagOfWords(const Options& options) {
  try {
    vocabulary =
        std::make_shared<OrbVocabulary>(OrbVocabulary(options.vocab_file));
  } catch (const std::string& _) {
    throw std::invalid_argument("Vocabulary file does not exist!");
  }
  database = std::make_shared<OrbDatabase>(
      OrbDatabase(*vocabulary, /*use_di=*/false, /*di_levels=*/0));
}

BagOfWords::BagOfWords(VocabOptions vocab_options) {
  // The DBoW2 method takes ints, so safely cast.
  // std::string text = "Hello, world!";
  // std::ofstream file("output.txt");

  // if (file.is_open()) {
  //     file << text << std::endl;
  //     file.close();
  //     std::cout << "Successfully wrote text to file." << std::endl;
  // } else {
  //     std::cerr << "Failed to open file." << std::endl;
  // }
  int dbow_branching_factor = static_cast<int>(vocab_options.branching_factor);
  int dbow_levels = static_cast<int>(vocab_options.levels);
  DBoW2::WeightingType dbow_weight = DBoW2::WeightingType::TF_IDF;
  switch (vocab_options.weight) {
    case BagOfWords::WeightingType::BINARY:
      dbow_weight = DBoW2::WeightingType::BINARY;
      break;
    case BagOfWords::WeightingType::IDF:
      dbow_weight = DBoW2::WeightingType::IDF;
      break;
    case BagOfWords::WeightingType::TF:
      dbow_weight = DBoW2::WeightingType::TF;
      break;
    case BagOfWords::WeightingType::TF_IDF:
    default:
      dbow_weight = DBoW2::WeightingType::TF_IDF;
      break;
  }
  DBoW2::ScoringType dbow_score = DBoW2::ScoringType::L1_NORM;
  switch (vocab_options.scoring) {
    case BagOfWords::ScoringType::BHATTACHARYYA:
      dbow_score = DBoW2::ScoringType::BHATTACHARYYA;
      break;
    case BagOfWords::ScoringType::CHI_SQUARE:
      dbow_score = DBoW2::ScoringType::CHI_SQUARE;
      break;
    case BagOfWords::ScoringType::DOT_PRODUCT:
      dbow_score = DBoW2::ScoringType::DOT_PRODUCT;
      break;
    case BagOfWords::ScoringType::KL:
      dbow_score = DBoW2::ScoringType::KL;
      break;
    case BagOfWords::ScoringType::L1_NORM:
    default:
      dbow_score = DBoW2::ScoringType::L1_NORM;
      break;
    case BagOfWords::ScoringType::L2_NORM:
      dbow_score = DBoW2::ScoringType::L2_NORM;
      break;
  }
  vocabulary = std::make_shared<OrbVocabulary>(OrbVocabulary(
      dbow_branching_factor, dbow_levels, dbow_weight, dbow_score));
  std::vector<std::vector<cv::Mat>> all_descriptors;
  all_descriptors.resize(vocab_options.descriptors.size());
  for (size_t i = 0; i < vocab_options.descriptors.size(); ++i) {
    cv::Mat descriptors_cv = convertDescriptors(vocab_options.descriptors[i]);
    std::vector<cv::Mat> descriptors_vector =
        convertDescriptors(descriptors_cv);
    all_descriptors[i] = descriptors_vector;
  }
  vocabulary->create(all_descriptors);
  database = std::make_shared<OrbDatabase>(
      OrbDatabase(*vocabulary, /*use_di=*/false, /*di_levels=*/0));
}

auto BagOfWords::insertToDatabase(const cv::Mat& descriptors) -> unsigned int {
  if (descriptors.type() != CV_8U) {
    throw std::invalid_argument("Descriptors must be 8-bit unsigned integer!");
  }
  std::vector<cv::Mat> descriptors_vector = convertDescriptors(descriptors);
  unsigned int id = database->add(descriptors_vector);
  return id;
}

auto BagOfWords::insertToDatabase(
    const Eigen::Matrix<uint8_t, -1, -1>& descriptors) -> unsigned int {
  cv::Mat descriptors_cv = convertDescriptors(descriptors);
  unsigned int id = insertToDatabase(descriptors_cv);
  return id;
}

auto BagOfWords::queryDatabase(const cv::Mat& descriptors) const
    -> std::map<unsigned int, double> {
  if (descriptors.type() != CV_8U) {
    throw std::invalid_argument("Descriptors must be 8-bit unsigned integer!");
  }
  std::vector<cv::Mat> descriptors_vector = convertDescriptors(descriptors);
  DBoW2::QueryResults query_results;
  database->query(descriptors_vector, query_results, /*max_results=*/-1,
                  /*max_id=*/-1);
  std::map<unsigned int, double> results;
  for (auto&& query_result : query_results) {
    std::pair<unsigned int, double> result =
        std::make_pair(query_result.Id, query_result.Score);
    results.insert(result);
  }
  return results;
}

auto BagOfWords::queryDatabase(
    const Eigen::Matrix<uint8_t, -1, -1>& descriptors) const
    -> std::map<unsigned int, double> {
  cv::Mat descriptors_cv = convertDescriptors(descriptors);
  std::map<unsigned int, double> results = queryDatabase(descriptors_cv);
  return results;
}

void BagOfWords::saveVocabulary(const std::string& vocab_file) const {
  vocabulary->save(vocab_file);
}

auto BagOfWords::convertDescriptors(const cv::Mat& descriptors)
    -> std::vector<cv::Mat> {
  std::vector<cv::Mat> descriptors_vector;
  descriptors_vector.resize(descriptors.rows);
  for (size_t i = 0; i < descriptors.rows; ++i) {
    descriptors_vector[i] = descriptors.row(i);
  }
  return descriptors_vector;
}

auto BagOfWords::convertDescriptors(
    const Eigen::Matrix<uint8_t, -1, -1>& descriptors) -> cv::Mat {
  cv::Mat descriptors_cv(descriptors.rows(), descriptors.cols(), CV_8U);
  cv::eigen2cv(descriptors, descriptors_cv);
  return descriptors_cv;
}

#if defined(BUILD_PYTHON)
// GCOVR_EXCL_START
// NOLINTNEXTLINE(google-runtime-references) PyBind preferred signature.
void pybindBagOfWords(pybind11::module_& module) {
  pybind11::class_<BagOfWords, std::shared_ptr<BagOfWords>> bag_of_words(
      module, /*name=*/"BagOfWords");
  pybind11::enum_<BagOfWords::WeightingType>(bag_of_words,
                                             /*name=*/"WeightingType")
      .value(/*name=*/"BINARY", BagOfWords::WeightingType::BINARY)
      .value(/*name=*/"IDF", BagOfWords::WeightingType::IDF)
      .value(/*name=*/"TF", BagOfWords::WeightingType::TF)
      .value(/*name=*/"TF_IDF", BagOfWords::WeightingType::TF_IDF)
      .export_values();
  pybind11::enum_<BagOfWords::ScoringType>(bag_of_words, /*name=*/"ScoringType")
      .value(/*name=*/"BHATTACHARYYA", BagOfWords::ScoringType::BHATTACHARYYA)
      .value(/*name=*/"CHI_SQUARE", BagOfWords::ScoringType::CHI_SQUARE)
      .value(/*name=*/"DOT_PRODUCT", BagOfWords::ScoringType::DOT_PRODUCT)
      .value(/*name=*/"KL", BagOfWords::ScoringType::KL)
      .value(/*name=*/"L1_NORM", BagOfWords::ScoringType::L1_NORM)
      .value(/*name=*/"L2_NORM", BagOfWords::ScoringType::L2_NORM)
      .export_values();
  pybind11::class_<BagOfWords::VocabOptions,
                   std::shared_ptr<BagOfWords::VocabOptions>>
      vocab_options(bag_of_words, /*name=*/"VocabOptions");
  vocab_options.def(pybind11::init<>());
  vocab_options.def_readwrite(/*name=*/"descriptors",
                              &BagOfWords::VocabOptions::descriptors);
  vocab_options.def_readwrite(/*name=*/"branching_factor",
                              &BagOfWords::VocabOptions::branching_factor);
  vocab_options.def_readwrite(/*name=*/"levels",
                              &BagOfWords::VocabOptions::levels);
  vocab_options.def_readwrite(/*name=*/"weight",
                              &BagOfWords::VocabOptions::weight);
  vocab_options.def_readwrite(/*name=*/"scoring",
                              &BagOfWords::VocabOptions::scoring);
  pybind11::class_<BagOfWords::Options, std::shared_ptr<BagOfWords::Options>>
      options(bag_of_words, /*name=*/"Options");
  options.def(pybind11::init<>());
  options.def_readwrite(/*name=*/"vocab_file",
                        &BagOfWords::Options::vocab_file);
  bag_of_words.def(pybind11::init<BagOfWords::Options>(),
                   pybind11::arg(/*name=*/"options"));
  bag_of_words.def(pybind11::init<BagOfWords::VocabOptions>(),
                   pybind11::arg(/*name=*/"vocab_options"));
  bag_of_words.def(
      /*name_=*/"insert_to_database",
      pybind11::overload_cast<const Eigen::Matrix<uint8_t, -1, -1>&>(
          &BagOfWords::insertToDatabase),
      pybind11::arg(/*name=*/"descriptors"));
  bag_of_words.def(
      /*name_=*/"query_database",
      pybind11::overload_cast<const Eigen::Matrix<uint8_t, -1, -1>&>(
          &BagOfWords::queryDatabase, pybind11::const_),
      pybind11::arg(/*name=*/"descriptors"));
  bag_of_words.def(/*name_=*/"save_vocabulary", &BagOfWords::saveVocabulary,
                   pybind11::arg(/*name=*/"vocab_file"));
}
// GCOVR_EXCL_STOP
#endif  // BUILD_PYTHON
}  // namespace ground_texture_slam_new
