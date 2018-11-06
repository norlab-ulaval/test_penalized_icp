cmake_minimum_required(VERSION 3.5)

project(pypm VERSION 0.1)

find_package(Eigen3 REQUIRED)


add_library(pypm INTERFACE)
target_include_directories(pypm BEFORE INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
  ${EIGEN3_INCLUDE_DIR})

target_compile_features(pypm INTERFACE cxx_auto_type cxx_range_for cxx_rvalue_references)

add_subdirectory(python)


# === INSTALLATION ===
include(CMakePackageConfigHelpers)
set(PYPM_INSTALL_DIR lib/pypm)

# Generate the pypm-config file which makes pypm available to find_package commands.
#configure_package_config_file(pypm-config.cmake.in ${CMAKE_BINARY_DIR}/pypm-config.cmake
#  INSTALL_DESTINATION ${PYPM_INSTALL_DIR}/cmake)


# Install the header files of the library.
install(DIRECTORY include/pypm
  DESTINATION include
  FILES_MATCHING REGEX ".h$|.hpp$")

# Define lieroy_export as the export for target lieroy.
#install(TARGETS lieroy distribution_of_se3
#  EXPORT lieroy-targets
#  RUNTIME DESTINATION bin)

# Install an export file which allows consumers to use the target lieroy::lieroy in their project.
#install(EXPORT pypm-targets
#  NAMESPACE pypm::
#  DESTINATION ${LIEROY_INSTALL_DIR}/cmake)

# Installe the auto generated files to make the targets available through find_package.
#install(FILES ${CMAKE_BINARY_DIR}/lieroy-config.cmake
#  DESTINATION ${LIEROY_INSTALL_DIR}/cmake)
