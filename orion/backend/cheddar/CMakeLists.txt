cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(orion_cheddar_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CHEDDAR_ROOT)
  message(FATAL_ERROR "Set -DCHEDDAR_ROOT=/abs/path/to/cheddar-fhe")
endif()

get_filename_component(CHEDDAR_ROOT "${CHEDDAR_ROOT}" ABSOLUTE)
set(CHEDDAR_BUILD_DIR "${CHEDDAR_ROOT}/build")

find_library(
  CHEDDAR_LIBRARY
  NAMES cheddar libcheddar
  PATHS "${CHEDDAR_BUILD_DIR}"
  NO_DEFAULT_PATH
  REQUIRED
)

find_package(CUDAToolkit REQUIRED)

add_library(cheddar_backend SHARED compat.cpp)

target_include_directories(
  cheddar_backend
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CHEDDAR_ROOT}/include"
    "${CHEDDAR_BUILD_DIR}/_deps/rmm-src/cpp/include"
    "${CHEDDAR_BUILD_DIR}/_deps/rmm-build/include"
    "${CHEDDAR_BUILD_DIR}/_deps/cccl-src/lib/cmake/libcudacxx/../../../libcudacxx/include"
    "${CHEDDAR_BUILD_DIR}/_deps/cccl-src/lib/cmake/thrust/../../../thrust"
    "${CHEDDAR_BUILD_DIR}/_deps/cccl-src/lib/cmake/cub/../../../cub"
    "${CHEDDAR_BUILD_DIR}/_deps/rapids_logger-src/include"
)

target_link_libraries(
  cheddar_backend
  PRIVATE
    "${CHEDDAR_LIBRARY}"
    CUDA::cudart
)

target_compile_options(
  cheddar_backend
  PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
)

target_compile_definitions(
  cheddar_backend
  PRIVATE
    ENABLE_EXTENSION
    LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
    CCCL_DISABLE_PDL
    CUB_DISABLE_NAMESPACE_MAGIC
    CUB_IGNORE_NAMESPACE_MAGIC_ERROR
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
    THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP
    THRUST_DISABLE_ABI_NAMESPACE
    THRUST_IGNORE_ABI_NAMESPACE_ERROR
)

if(APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(_backend_name "cheddar-mac-arm64")
  else()
    set(_backend_name "cheddar-mac")
  endif()
elseif(WIN32)
  set(_backend_name "cheddar-windows")
else()
  set(_backend_name "cheddar-linux")
endif()

set_target_properties(
  cheddar_backend
  PROPERTIES
    OUTPUT_NAME "${_backend_name}"
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    BUILD_RPATH "${CHEDDAR_BUILD_DIR}"
)
