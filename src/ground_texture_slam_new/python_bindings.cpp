#include <pybind11/pybind11.h>
// GCOVR_EXCL_START
namespace ground_texture_slam_new {
// Forward declare all the functions so the compiler doesn't complain.
/**
 * @brief Python bindings for the @ref BagOfWords class.
 *
 * @param module The Python module to place the class in.
 */
// NOLINTNEXTLINE(google-runtime-references) PyBind preferred signature.
void pybindBagOfWords(pybind11::module_& module);
/**
 * @brief Python bindings for the @ref GroundTextureSLAM class.
 *
 * @param module The Python module to place the class in.
 */
// NOLINTNEXTLINE(google-runtime-references) PyBind preferred signature.
void pybindGroundTextureSLAM(pybind11::module_& module);
/**
 * @brief Python bindings for the @ref ImageParser class.
 *
 * @param module The Python module to place the class in.
 */
// NOLINTNEXTLINE(google-runtime-references) PyBind preferred signature.
void pybindImageParser(pybind11::module_& module);
/**
 * @brief Python bindings for the @ref KeypointMatcher class.
 *
 * @param module The Python module to place the class in.
 */
// NOLINTNEXTLINE(google-runtime-references) PyBind preferred signature.
void pybindKeypointMatcher(pybind11::module_& module);
/**
 * @brief Python bindings for the @ref TransformEstimator class.
 *
 * @param module The Python module to place the class in.
 */
// NOLINTNEXTLINE(google-runtime-references) PyBind preferred signature.
void pybindTransformEstimator(pybind11::module_& module);

/// @brief The macro to enable Python bindings.
// NOLINTNEXTLINE Don't check external library
PYBIND11_MODULE(ground_texture_slam_new, module) {
  pybindBagOfWords(module);
  pybindImageParser(module);
  pybindKeypointMatcher(module);
  pybindTransformEstimator(module);
  // Call this one last, as it depends on some of the other bindings.
  pybindGroundTextureSLAM(module);
}
}  // namespace ground_texture_slam_new
// GCOVR_EXCL_STOP
