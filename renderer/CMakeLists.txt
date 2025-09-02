cmake_minimum_required(VERSION 3.10)
project(simplER)

# Compilation options
set(USE_GCC 1)
set(PRODUCE_REPORTS 0)
set(DEBUG_MODE 0)
set(USE_PRINTING 0)
set(PRINT_DEBUGLOG 0)
set(RUNTIME_DEBUGLOG 1)
set(PROJECTOR 1)
set(USE_MATLAB 0)
set(USE_PYBIND 1)
set(USE_SFMT 1)
set(USE_CERES 0)
set(USE_OPENEXR 0)
set(USE_THREADED 1)
set(USE_DOUBLE_PRECISION 1)
set(USE_PIXEL_SHARING 1)
set(USE_NO_FRESNEL 0)
set(USE_FILE 0)
set(USE_WEIGHT_NORMALIZATION 0)
set(USE_SIMPLIFIED_TIMING 0)
set(USE_OMEGA_TRACKING 0)
set(USE_CUDA 0)
set(USE_RIF_SPLINE 0)
set(USE_RIF_FUS 0)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SRC_FILES
    src/image.cpp
    src/phase.cpp
    src/sampler.cpp
    src/matrix.cpp
    src/photon.cpp
    src/scene.cpp
    src/warp.cpp
    src/vmf.cpp
    src/bsdf.cpp
)

# Headers
set(HEADERS
    include/constants.h
    include/debug_util.h
    include/image.h
    include/medium.h
    include/phase.h
    include/renderer.h
    include/sampler.h
    include/spline.h
    include/tvector.h
    include/matrix.h
    include/util.h
    include/photon.h
    include/scene.h
    include/warp.h
    include/vmf.h
    include/bsdf.h
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

if(USE_GCC)
    set(CMAKE_CXX_COMPILER "gcc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_GCC")
endif()

if(USE_SFMT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_SFMT")
    list(APPEND HEADERS include/rng_sse.h include/sse.h)
    list(APPEND SRC_FILES src/rng_sse.cpp)
else()
    list(APPEND HEADERS include/rng_boost.h)
endif()

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -maxrregcount 64 -m64 -std=c++14 -arch=sm_75")
    list(APPEND HEADERS include/cuda_renderer.h include/cuda_image.cuh include/cuda_scene.cuh include/cuda_utils.cuh include/cuda_vector.cuh)
    list(APPEND SRC_FILES src/cuda_renderer.cu)
endif()

# Add executable
add_executable(simplER ${SRC_FILES})

# Link libraries
if(USE_CUDA)
    target_link_libraries(simplER cuda cudart curand)
endif()

# Pybind11
if(USE_PYBIND)
    find_package(Python3 COMPONENTS Interpreter Development)
    execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --includes OUTPUT_VARIABLE PYBIND_INCLUDES)
    string(STRIP ${PYBIND_INCLUDES} PYBIND_INCLUDES)
    include_directories(${PYBIND_INCLUDES})
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" OUTPUT_VARIABLE PYBIND_SUFFIX)
    string(STRIP ${PYBIND_SUFFIX} PYBIND_SUFFIX)

    add_library(scene_pybind MODULE pybindFiles/scene_pybind.cpp)
    set_target_properties(scene_pybind PROPERTIES PREFIX "" SUFFIX ${PYBIND_SUFFIX})
    target_link_libraries(scene_pybind ${LIBS})

    add_library(photon_pybind MODULE pybindFiles/photon_pybind.cpp)
    set_target_properties(photon_pybind PROPERTIES PREFIX "" SUFFIX ${PYBIND_SUFFIX})
    target_link_libraries(photon_pybind ${LIBS})

    add_library(tvector_pybind MODULE pybindFiles/tvector_pybind.cpp)
    set_target_properties(tvector_pybind PROPERTIES PREFIX "" SUFFIX ${PYBIND_SUFFIX})
    target_link_libraries(tvector_pybind ${LIBS})

    add_library(medium_pybind MODULE pybindFiles/medium_pybind.cpp)
    set_target_properties(medium_pybind PROPERTIES PREFIX "" SUFFIX ${PYBIND_SUFFIX})
    target_link_libraries(medium_pybind ${LIBS})

    add_library(phase_pybind MODULE pybindFiles/phase_pybind.cpp)
    set_target_properties(phase_pybind PROPERTIES PREFIX "" SUFFIX ${PYBIND_SUFFIX})
    target_link_libraries(phase_pybind ${LIBS})

    add_library(image_pybind MODULE pybindFiles/image_pybind.cpp)
    set_target_properties(image_pybind PROPERTIES PREFIX "" SUFFIX ${PYBIND_SUFFIX})
    target_link_libraries(image_pybind ${LIBS})
endif()

# Clean and distclean targets
add_custom_target(clean
    COMMAND ${CMAKE_COMMAND} -E remove *.o *~
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/mexfiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/samples
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/src
)

add_custom_target(distclean
    COMMAND ${CMAKE_COMMAND} -E remove *.o *~
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/pybindFiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/mexfiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/samples
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_SOURCE_DIR}/src
)