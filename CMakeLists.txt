# ---- CMakeLists.txt for simplER ----  
cmake_minimum_required(VERSION 3.20)

# ---- Terminal color formatting ----
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
endif()


# ---- Configuration Flags ----
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
set(USE_DOUBLE_PRECISION 0)
set(USE_PIXEL_SHARING 1)
set(USE_NO_FRESNEL 0)
set(USE_FILE 0)
set(USE_WEIGHT_NORMALIZATION 0)
set(USE_SIMPLIFIED_TIMING 0)
set(USE_OMEGA_TRACKING 0)
set(USE_CUDA 0)

set(USE_RIF_SPLINE 0)
set(USE_RIF_FUS 1)
set(USE_RIF_PARABOLIC 0)

# ---- Compile Options ----
message("")
message("")
get_cmake_property(_variableNames VARIABLES)
foreach(_variableName ${_variableNames})
    if(_variableName MATCHES "^USE_")
        if(${_variableName})
            message("${BoldGreen}  ${_variableName} = ON${ColourReset}")
        else()
            message("${BoldRed}  ${_variableName} = OFF${ColourReset}")
        endif()
    endif()
endforeach()
message("")
message("")

# ---- Project Name ----
project(simplER LANGUAGES C CXX)
message("${BoldGreen}Build directory: ${CMAKE_BINARY_DIR}${ColourReset}")

# ---- Compiler overrides ----
if(NOT DEFINED CMAKE_C_COMPILER)
  set(CMAKE_C_COMPILER /usr/bin/gcc-11 CACHE FILEPATH "" FORCE)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER /usr/bin/g++-11 CACHE FILEPATH "" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11 CACHE FILEPATH "" FORCE)
endif()

message("${BoldGreen}C compile definitions: ${CMAKE_C_COMPILER}${ColourReset}")
message("${BoldGreen}C++ compile definitions: ${CMAKE_CXX_COMPILER}${ColourReset}")
message("${BoldGreen}CUDA compile definitions: ${CMAKE_CUDA_HOST_COMPILER}${ColourReset}")

# ---- MATLAB and Python Paths ----
set(MATLAB_DIR $ENV{MATLAB_DIR})
message("${BoldGreen}Using MATLAB_DIR: ${MATLAB_DIR}${ColourReset}")
message("${BoldGreen}CMake toolchain file: ${CMAKE_TOOLCHAIN_FILE}${ColourReset}")

find_package(Python3 COMPONENTS Development REQUIRED)
message("${BoldGreen}Python3 include directories: ${Python3_INCLUDE_DIRS}${ColourReset}")
include_directories(${Python3_INCLUDE_DIRS})

execute_process(
    COMMAND python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
    OUTPUT_VARIABLE PYTHON_LIBDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND python -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))"
    OUTPUT_VARIABLE PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(PYTHON_LIBRARY ${PYTHON_LIBDIR}/libpython${PYTHON_VERSION}.so)
message("${BoldGreen}Python library: ${PYTHON_LIBRARY}${ColourReset}")

# ---- Project Flags ----
set(CMAKE_VERBOSE_MAKEFILE ON)

if(DEBUG_MODE)
    set(NDEBUG 1)
endif()



# ---- Compiler flags ----
set(GENERALFLAGS -fPIC -std=c++17)
set(SHARED_OPTIMFLAGS
    -march=native -ffast-math -fopenmp -pthread -pipe -msse2
    -ftree-vectorize -mfpmath=sse -funsafe-math-optimizations
    -fno-rounding-math -fno-signaling-nans -fno-math-errno -fomit-frame-pointer
)
set(OPTIMFLAGS $<$<BOOL:${DEBUG_MODE}>:-O0> $<$<NOT:$<BOOL:${DEBUG_MODE}>>:-O3> ${SHARED_OPTIMFLAGS})
set(REPORTSFLAGS $<$<BOOL:${PRODUCE_REPORTS}>:-Winline -Wimplicit>)

# ---- CUDA Setup ----
if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()


# ---- NumPy include path ----
execute_process(
    COMMAND python -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${NUMPY_INCLUDE_DIR})
message("${BoldGreen}NumPy include directory: ${NUMPY_INCLUDE_DIR}${ColourReset}")

# ---- Include and Source Paths ----
include_directories(${CMAKE_SOURCE_DIR}/renderer/include)

# Explicit main entry point
set(SIMPLER_MAIN_SRC ${CMAKE_SOURCE_DIR}/simplER.cpp)


# Collect all renderer implementation files
file(GLOB_RECURSE RENDERER_SRC_FILES
    ${CMAKE_SOURCE_DIR}/renderer/src/*.cpp
)

# Combine everything into one list
set(SIMPLER_SRC_FILES ${SIMPLER_MAIN_SRC} ${RENDERER_SRC_FILES})

# ---- Executable ----
add_executable(simplER ${SIMPLER_SRC_FILES})

# ---- Macro Flags Propagation ----
set(ALL_MACRO_FLAGS
  USE_GCC
  PRODUCE_REPORTS
  USE_PRINTING
  PRINT_DEBUGLOG
  RUNTIME_DEBUGLOG
  PROJECTOR

  USE_MATLAB
  USE_PYBIND
  USE_SFMT

  USE_CERES
  USE_OPENEXR

  USE_THREADED
  USE_DOUBLE_PRECISION
  USE_PIXEL_SHARING
  USE_NO_FRESNEL
  USE_FILE
  USE_WEIGHT_NORMALIZATION
  USE_SIMPLIFIED_TIMING
  USE_OMEGA_TRACKING
  USE_CUDA

  # NDEBUG
  
  USE_RIF_SPLINE
  USE_RIF_FUS
  USE_RIF_PARABOLIC
)

foreach(flag IN LISTS ALL_MACRO_FLAGS)
  target_compile_definitions(simplER PRIVATE ${flag}=${${flag}})
endforeach()


if(USE_MATLAB)
# ---- MATLAB Support ----
set(MATLAB_ARCH glnxa64)
execute_process(COMMAND ${MATLAB_DIR}/bin/mexext OUTPUT_VARIABLE MEXEXT OUTPUT_STRIP_TRAILING_WHITESPACE)
include_directories(${MATLAB_DIR}/extern/include)
link_directories(${MATLAB_DIR}/bin/${MATLAB_ARCH})
add_compile_definitions(MATLAB_MEX_FILE _GNU_SOURCE)
add_compile_options(-fexceptions -fno-omit-frame-pointer)

# ---- Linker RPATH workaround ----
set(RPATH_FLAG "-Wl,-rpath-link,${MATLAB_DIR}/bin/${MATLAB_ARCH}")
target_link_options(simplER PRIVATE ${RPATH_FLAG})
endif()

# ---- Compiler Options ----
add_compile_options(${GENERALFLAGS} ${OPTIMFLAGS} ${REPORTSFLAGS})

# ---- Dependencies ----
find_package(Eigen3 CONFIG REQUIRED)
find_package(boost_core CONFIG REQUIRED)
find_package(boost_algorithm CONFIG REQUIRED)
find_package(boost_static_assert CONFIG REQUIRED)
find_package(boost_random CONFIG REQUIRED)
find_package(boost_iterator CONFIG REQUIRED)
find_package(boost_math CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)

# ---- Ceres (optional) ----
if(USE_CERES)
  find_package(Ceres CONFIG REQUIRED)
  target_link_libraries(simplER PRIVATE Ceres::ceres)
  add_compile_options(
    -Wmissing-declarations -Wno-unknown-pragmas -Wno-sign-compare
    -Wno-unused-parameter -Wno-missing-field-initializers
    -DCERES_GFLAGS_NAMESPACE=google
  )
endif()

# ---- Pybind11 (optional) ----
execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYBIND11_INCLUDE_DIR)
  include_directories(${PYBIND11_INCLUDE_DIR})
  message("${BoldGreen}Pybind11 include directory: ${PYBIND11_INCLUDE_DIR}${ColourReset}")
else()
  message("${BoldRed}Pybind11 include directory not found.${ColourReset}")
endif()


find_package(Imath CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# ---- Linking ----
target_link_libraries(simplER PRIVATE Imath::Imath)
target_link_libraries(simplER PRIVATE fmt::fmt)
target_link_libraries(simplER PRIVATE Eigen3::Eigen)
target_link_libraries(simplER PRIVATE Boost::core Boost::algorithm Boost::static_assert Boost::random Boost::iterator Boost::math)
target_link_libraries(simplER PRIVATE OpenBLAS::OpenBLAS)
target_link_libraries(simplER PRIVATE ${PYTHON_LIBRARY})
target_link_libraries(simplER PRIVATE OpenMP::OpenMP_CXX)
