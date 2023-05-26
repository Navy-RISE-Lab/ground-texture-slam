#if !defined(GROUND_TEXTURE_SLAM_BAG_OF_WORDS_H_)
#define GROUND_TEXTURE_SLAM_BAG_OF_WORDS_H_

#include <DBoW2/DBoW2.h>

#include <Eigen/Dense>
#include <exception>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <string>
#include <vector>

namespace ground_texture_slam_new {
/**
 * @brief A class that uses the bag of words method to identify potential loop
 * closures.
 *
 * Specifically, it uses the algorithms implemented here:
 * https://github.com/dorian3d/DBoW2.
 *
 */
class BagOfWords {
 public:
  /// The algorithm to weight different descriptors.
  enum class WeightingType { BINARY, IDF, TF, TF_IDF };
  /// The algorithm to score differences between images.
  enum class ScoringType {
    BHATTACHARYYA,
    CHI_SQUARE,
    DOT_PRODUCT,
    KL,
    L1_NORM,
    L2_NORM
  };
  /// @brief The options to customize this class when building a new vocabulary.
  struct VocabOptions {
    /// @brief A set of descriptors to use during vocabulary building.
    std::vector<Eigen::Matrix<uint8_t, -1, -1>> descriptors =
        std::vector<Eigen::Matrix<uint8_t, -1, -1>>();
    /// @brief How many branches each node in the tree should have.
    unsigned int branching_factor = 9;  // NOLINT Default value
    /// @brief The height of the tree.
    unsigned int levels = 5;  // NOLINT Default value
    /// @brief The weight algorithm to use when comparing images.
    WeightingType weight = WeightingType::TF_IDF;
    /// @brief The scoring algorithm to use when comparing descriptors.
    ScoringType scoring = ScoringType::L1_NORM;
  };
  /// @brief The options to customize this class when loading vocabulary from
  /// file.
  struct Options {
    /// @brief The existing vocabulary file to load.
    std::string vocab_file = "vocabulary.bow";
  };

  /**
   * @brief Construct a new BagOfWords object from a previously built vocabulary
   * tree.
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam_new.BagOfWords(
   *     options: ground_texture_slam_new.BagOfWords.Options
   * )
   * @endcode
   *
   * @param options The customization options for this object.
   * @throws std::invalid_argument thrown if the vocabulary file doesn't exist.
   */
  explicit BagOfWords(const Options& options);

  /**
   * @brief Construct a new BagOfWords object with a brand new vocabulary tree.
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam_new.BagOfWords(
   *     vocab_options: ground_texture_slam_new.BagOfWords.VocabOptions
   * )
   * @endcode
   *
   * @param vocab_options The customization options for this object.
   */
  explicit BagOfWords(VocabOptions vocab_options);

  /**
   * @brief Insert descriptors for a single image into the database.
   *
   * @note This method does not have a direct Python equivalent.
   *
   * @param descriptors The descriptors to insert. They must be CV_8U types.
   * @return unsigned int The ID of the inserted descriptors.
   * @throws std::invalid_argument thrown if the descriptor type is incorrect.
   */
  auto insertToDatabase(const cv::Mat& descriptors) -> unsigned int;

  /**
   * @brief Insert descriptors for a single image into the database.
   *
   * @note This is an overloaded method for Python binding. It adds
   * additional overhead for data conversions.
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam_new.BagOfWords.insert_to_database(
   *     descriptors: numpy.ndarray[numpy.uint8[m, n]]
   * ) -> int
   * @endcode
   *
   * @param descriptors The descriptors to insert.
   * @return unsigned int The ID of the inserted descriptors.
   */
  auto insertToDatabase(const Eigen::Matrix<uint8_t, -1, -1>& descriptors)
      -> unsigned int;

  /**
   * @brief Determine match scores against all descriptors already in the
   * database.
   *
   * @note This method does not have a direct Python equivalent.
   *
   * @param descriptors The descriptors to find matches against. Type must be
   * CV_8U.
   * @return std::map<unsigned int, double>  A map of the matching IDs and
   * associated scores.
   * @throws std::invalid_argument thrown if the descriptor type is incorrect.
   */
  // NOLINTNEXTLINE(modernize-use-nodiscard) Okay to discard, no state change.
  auto queryDatabase(const cv::Mat& descriptors) const
      -> std::map<unsigned int, double>;

  /**
   * @brief Determine match scores against all descriptors already in the
   * database.
   *
   * @note This is an overloaded method for Python binding. It adds additional
   * overhead for data conversions.
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam_new.BagOfWords.query_database(
   *     descriptors: numpy.ndarray[numpy.uint8[m, n]]
   * ) -> Dict[int, float]
   * @endcode
   *
   * @param descriptors The descriptors to find matches against.
   * @return std::map<unsigned int, double> A map of the matching IDs and
   * associated scores.
   */
  // NOLINTNEXTLINE(modernize-use-nodiscard) Okay to discard, no state change.
  auto queryDatabase(const Eigen::Matrix<uint8_t, -1, -1>& descriptors) const
      -> std::map<unsigned int, double>;

  /**
   * @brief Save the vocabulary tree to file for later import.
   *
   * @note Python syntax:
   * @code {.py}
   * ground_texture_slam_new.BagOfWords.save_vocabulary(vocab_file: str) -> None
   * @endcode
   *
   * @param vocab_file The file to write to.
   */
  void saveVocabulary(const std::string& vocab_file) const;

 private:
  /**
   * @brief Split a cv::Mat into a vector of individual 1 x M cv::Mats.
   *
   * @param descriptors The incoming single matrix.
   * @return std::vector<cv::Mat> A split up matrix. If the incoming is N x M,
   * this vector is size N where each element is a 1 x M cv::Mat.
   */
  // NOLINTNEXTLINE(modernize-use-nodiscard) Okay to discard, no state change.
  static auto convertDescriptors(const cv::Mat& descriptors)
      -> std::vector<cv::Mat>;
  /**
   * @brief Convert from Eigen descriptors to cv::Mat.
   *
   * @param descriptors The Eigen matrix to convert.
   * @return cv::Mat The equivalent cv::Mat of type CV_8U. Shapes are identical.
   */
  // NOLINTNEXTLINE(modernize-use-nodiscard) Okay to discard, no state change.
  static auto convertDescriptors(
      const Eigen::Matrix<uint8_t, -1, -1>& descriptors) -> cv::Mat;
  /// @brief The database used to match to previous images.
  std::shared_ptr<OrbDatabase> database;
  /// @brief The vocabulary to facilitate description matching.
  std::shared_ptr<OrbVocabulary> vocabulary;
};
}  // namespace ground_texture_slam_new

#endif  // GROUND_TEXTURE_SLAM_BAG_OF_WORDS_H_
