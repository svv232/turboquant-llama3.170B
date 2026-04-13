cmake_minimum_required(VERSION 3.27)

project(turboquant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build as shared library" ON)

# ----------------------------- MLX Dependency -----------------------------
# Point directly at the built MLX source tree
set(MLX_SOURCE_DIR "/tmp/mlx-source" CACHE PATH "Path to MLX source")
set(MLX_BUILD_DIR "${MLX_SOURCE_DIR}/build" CACHE PATH "Path to MLX build")

# Include MLX headers
include_directories(${MLX_SOURCE_DIR})
include_directories(${MLX_BUILD_DIR})

# Metal C++ headers (fetched by MLX build)
set(METAL_CPP_DIR "${MLX_BUILD_DIR}/_deps/metal_cpp-src" CACHE PATH "Metal C++ headers")
include_directories(${METAL_CPP_DIR})

# Link MLX static lib
find_library(MLX_LIB mlx PATHS ${MLX_BUILD_DIR} NO_DEFAULT_PATH REQUIRED)

# System frameworks needed by MLX
find_library(METAL_LIB Metal REQUIRED)
find_library(FOUNDATION_LIB Foundation REQUIRED)
find_library(METALKIT_LIB MetalKit REQUIRED)
find_library(ACCELERATE_LIB Accelerate REQUIRED)
find_library(QUARTZ_LIB QuartzCore REQUIRED)

# ----------------------------- TurboQuant Library -----------------------------
add_library(turboquant
  engine/polarquant.cpp
  engine/sdpa_int4.cpp
  engine/sdpa_polar.cpp
  engine/sdpa_qjl.cpp
  engine/qjl.cpp
  engine/llama_loader.cpp
  engine/llama_model.cpp
)

target_include_directories(turboquant PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib
  ${MLX_SOURCE_DIR}
)

target_link_libraries(turboquant PUBLIC
  ${MLX_LIB}
  ${METAL_LIB}
  ${FOUNDATION_LIB}
  ${ACCELERATE_LIB}
  ${QUARTZ_LIB}
)

target_compile_definitions(turboquant PUBLIC _METAL_)

# ----------------------------- Metal Kernels -----------------------------
# Build metallib from .metal sources
# We use mlx_build_metallib if available, otherwise manual xcrun
set(METAL_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/engine/polarquant.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/engine/sdpa_int4.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/engine/sdpa_polar.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/engine/sdpa_qjl.metal
)

set(METALLIB_OUTPUT ${CMAKE_BINARY_DIR}/turboquant.metallib)

# Get MLX Metal include path for kernel headers
set(MLX_METAL_INCLUDE ${MLX_SOURCE_DIR})

add_custom_command(
  OUTPUT ${METALLIB_OUTPUT}
  COMMAND xcrun -sdk macosx metal
    -std=metal3.0
    -O2
    -I ${MLX_METAL_INCLUDE}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -c ${CMAKE_CURRENT_SOURCE_DIR}/engine/polarquant.metal
    -o ${CMAKE_BINARY_DIR}/polarquant.air
  COMMAND xcrun -sdk macosx metal
    -std=metal3.0
    -O2
    -I ${MLX_METAL_INCLUDE}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -c ${CMAKE_CURRENT_SOURCE_DIR}/engine/sdpa_int4.metal
    -o ${CMAKE_BINARY_DIR}/sdpa_int4.air
  COMMAND xcrun -sdk macosx metal
    -std=metal3.0
    -O2
    -I ${MLX_METAL_INCLUDE}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -c ${CMAKE_CURRENT_SOURCE_DIR}/engine/sdpa_polar.metal
    -o ${CMAKE_BINARY_DIR}/sdpa_polar.air
  COMMAND xcrun -sdk macosx metal
    -std=metal3.0
    -O2
    -I ${MLX_METAL_INCLUDE}
    -I ${CMAKE_CURRENT_SOURCE_DIR}
    -c ${CMAKE_CURRENT_SOURCE_DIR}/engine/sdpa_qjl.metal
    -o ${CMAKE_BINARY_DIR}/sdpa_qjl.air
  COMMAND xcrun -sdk macosx metallib
    ${CMAKE_BINARY_DIR}/polarquant.air
    ${CMAKE_BINARY_DIR}/sdpa_int4.air
    ${CMAKE_BINARY_DIR}/sdpa_polar.air
    ${CMAKE_BINARY_DIR}/sdpa_qjl.air
    -o ${METALLIB_OUTPUT}
  DEPENDS ${METAL_SOURCES}
  COMMENT "Building Metal library"
)

add_custom_target(turboquant_metallib DEPENDS ${METALLIB_OUTPUT})
add_dependencies(turboquant turboquant_metallib)

# ----------------------------- Tests / Validation Binaries -----------------------------
add_executable(polarquant_validate tests/polarquant_validate.cpp)
target_link_libraries(polarquant_validate PRIVATE turboquant)
target_compile_definitions(polarquant_validate PRIVATE _METAL_)

add_executable(sdpa_int4_bench tests/sdpa_int4_bench.cpp)
target_link_libraries(sdpa_int4_bench PRIVATE turboquant)
target_compile_definitions(sdpa_int4_bench PRIVATE _METAL_)

add_executable(sdpa_polar_bench tests/sdpa_polar_bench.cpp)
target_link_libraries(sdpa_polar_bench PRIVATE turboquant)
target_compile_definitions(sdpa_polar_bench PRIVATE _METAL_)

add_executable(qjl_validate tests/qjl_validate.cpp)
target_link_libraries(qjl_validate PRIVATE turboquant)
target_compile_definitions(qjl_validate PRIVATE _METAL_)

add_executable(qjl_real tests/qjl_real_activations.cpp)
target_link_libraries(qjl_real PRIVATE turboquant)
target_compile_definitions(qjl_real PRIVATE _METAL_)

add_executable(decode_profile tests/decode_profile.cpp)
target_link_libraries(decode_profile PRIVATE turboquant)
target_compile_definitions(decode_profile PRIVATE _METAL_)

add_executable(context_scaling tests/context_scaling.cpp)
target_link_libraries(context_scaling PRIVATE turboquant)
target_compile_definitions(context_scaling PRIVATE _METAL_)

add_executable(chunk_size_bench tests/chunk_size_bench.cpp)
target_link_libraries(chunk_size_bench PRIVATE turboquant)
target_compile_definitions(chunk_size_bench PRIVATE _METAL_)

add_executable(kv_memory_test tests/kv_memory_test.cpp)
target_link_libraries(kv_memory_test PRIVATE turboquant)
target_compile_definitions(kv_memory_test PRIVATE _METAL_)

add_executable(qmatmul_bench tests/qmatmul_bench.cpp)
target_link_libraries(qmatmul_bench PRIVATE turboquant)
target_compile_definitions(qmatmul_bench PRIVATE _METAL_)

add_executable(sdpa_qjl_bench tests/sdpa_qjl_bench.cpp)
target_link_libraries(sdpa_qjl_bench PRIVATE turboquant)
target_compile_definitions(sdpa_qjl_bench PRIVATE _METAL_)

add_executable(llama_e2e tests/llama_e2e.cpp)
target_link_libraries(llama_e2e PRIVATE turboquant)
target_compile_definitions(llama_e2e PRIVATE _METAL_)
