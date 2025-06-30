cmake_minimum_required(VERSION 3.20)

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()


# Force use of gcc-11 if available
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

# Define the project name as a variable

set(MATLAB_DIR $ENV{MATLAB_DIR})


# Print the Python executable path using `which python`
execute_process(
    COMMAND which python
    OUTPUT_VARIABLE PYTHON_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(PYTHON_PATH ${PYTHON_EXECUTABLE})

message("${BoldGreen}Using MATLAB_DIR: ${MATLAB_DIR}${ColourReset}")
message("${BoldGreen}CMake toolchain file: ${CMAKE_TOOLCHAIN_FILE}${ColourReset}")
message("${BoldGreen}Python executable: ${PYTHON_EXECUTABLE}${ColourReset}")
message("${BoldGreen}Using Python at: ${PYTHON_PATH}${ColourReset}")

find_package(Python3 REQUIRED)

# Enable verbose makefile output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Debug mode flag (set to 1 for debug mode, 0 for release mode)
set(DEBUG_MODE 1)

# Flag to use CUDA (set to 1 to enable CUDA, 0 to disable)
set(USE_CUDA 1)


# Flag to use Ceres Solver (set to 1 to enable, 0 to disable)
set(USE_CERES 0)

# Flag to produce reports (set to 1 to enable, 0 to disable)
set(PRODUCE_REPORTS 0)

# General compiler flags
set(GENERALFLAGS -fPIC -std=c++17)

# Shared optimization flags
set(SHARED_OPTIMFLAGS 
    -march=native 
    -ffast-math 
    -fopenmp 
    -pthread 
    -pipe 
    -msse2 
    -ftree-vectorize 
    -mfpmath=sse 
    -funsafe-math-optimizations 
    -fno-rounding-math 
    -fno-signaling-nans 
    -fno-math-errno 
    -fomit-frame-pointer
)

# Set optimization flags based on debug mode
if(DEBUG_MODE)
    set(OPTIMFLAGS -O0 ${SHARED_OPTIMFLAGS})
else()
    set(OPTIMFLAGS -O3 ${SHARED_OPTIMFLAGS})
endif()

# Flags for producing reports (empty by default)
if(PRODUCE_REPORTS)
    set(REPORTSFLAGS -Winline -Wimplicit)
else()
    set(REPORTSFLAGS "")
endif()

# Compiler flags for Ceres
set(CERESFLAGS 
    -Wmissing-declarations 
    -Wno-unknown-pragmas 
    -Wno-sign-compare 
    -Wno-unused-parameter 
    -Wno-missing-field-initializers 
    -DNDEBUG 
    -DCERES_GFLAGS_NAMESPACE=google
)

# Base paths for vcpkg and other specific libraries
set(VCPKG_PACKAGES_SO_LIB_DIR ${VCPKG_ROOT}/installed/x64-linux-dynamic/lib)

# Libraries to link against
set(LIBS
    -lf77blas
    -latlas
    -lrt
    -lpthread
)
message("${BoldGreen}Matlab Dir directory: ${MATLAB_DIR}${ColourReset}")
set(MATLAB_ARCH glnxa64)
execute_process(COMMAND ${MATLAB_DIR}/bin/mexext OUTPUT_VARIABLE MEXEXT OUTPUT_STRIP_TRAILING_WHITESPACE)
set(MAPFILE mexFunction.map)

set(RPATH "-Wl,-rpath-link,${MATLAB_DIR}/bin/${MATLAB_ARCH}")
set(LIBS ${RPATH})

include_directories(${MATLAB_DIR}/extern/include)
link_directories(${MATLAB_DIR}/bin/${MATLAB_ARCH})

# Compiler definitions
add_compile_definitions(MATLAB_MEX_FILE _GNU_SOURCE)
add_compile_options(-fexceptions -fno-omit-frame-pointer)

set(PYFLAGS "-O3 -Wall -shared -std=c++14 -fPIC -march=native -ffast-math" "-fopenmp" "-pthread -DUSE_GCC -DUSE_SFMT")
if(NOT DEBUG_MODE)
list(APPEND PYFLAGS "-DNDEBUG")
endif()

# Define the project name
set(PROJECT_NAME simplER)
project(${PROJECT_NAME})

# Print build directories in green
message("${BoldGreen}Build directory: ${CMAKE_BINARY_DIR}${ColourReset}")
message("${BoldGreen}Project Build directory: ${PROJECT_BINARY_DIR}${ColourReset}")

# Combine all flags
add_compile_options(${GENERALFLAGS} ${OPTIMFLAGS} ${REPORTSFLAGS})

# Find required packages
find_package(Eigen3 CONFIG REQUIRED)
find_package(boost_core CONFIG REQUIRED)
find_package(boost_algorithm CONFIG REQUIRED)
find_package(boost_static_assert CONFIG REQUIRED)
find_package(boost_random CONFIG REQUIRED)
find_package(boost_iterator CONFIG REQUIRED)
find_package(boost_math CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)

# Find Ceres package if enabled
if(USE_CERES)
find_package(Ceres CONFIG REQUIRED)
endif()




# Add CUDA include directory if enabled
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    enable_language(CUDA)
    include_directories(/usr/local/cuda/include/)
    link_directories(/usr/local/cuda/lib64/)
    # set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-use_fast_math;-maxrregcount=64;-m64;-ccbin=${CMAKE_CXX_COMPILER};-std=c++14;-arch=sm_75)
    
    # set(CUDA_SOURCE "${CMAKE_SOURCE_DIR}/simplER.cpp")
    # set(CUDA_OBJECT "${CMAKE_BINARY_DIR}/simplER.o")
    # add_custom_command(
    #     OUTPUT ${CUDA_OBJECT}
    #     COMMAND ${CUDA_NVCC_EXECUTABLE} ${CUDA_SOURCE} ${CUDA_NVCC_FLAGS} -Xcompiler "${CMAKE_CXX_FLAGS}" ${INCLUDES} -c -o ${CUDA_OBJECT}
    #     DEPENDS ${CUDA_SOURCE}
    # )
    # add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:${GENERALFLAGS} ${OPTIMFLAGS} ${REPORTSFLAGS} ${PYFLAGS}>)
    # add_custom_target(cuda_objects ALL DEPENDS ${CUDA_OBJECT})
    add_executable(${PROJECT_NAME} simplER.cpp)
    # add_dependencies(${PROJECT_NAME} cuda_objects)
    # target_sources(${PROJECT_NAME} PRIVATE ${CUDA_OBJECT})
    # target_link_libraries(${PROJECT_NAME} PRIVATE cudart curand)
else()
    add_executable(${PROJECT_NAME} simplER.cpp)
endif()

# Link libraries to the target
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::core Boost::algorithm 
Boost::static_assert Boost::random 
Boost::iterator Boost::math)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS::OpenBLAS)
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
# target_link_libraries(${PROJECT_NAME} PRIVATE mex mat mx)

# PYBIND11 from Python PIP
execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Link Ceres library if enabled
if(USE_CERES)
add_compile_options(${CERESFLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE Ceres::ceres)
endif()


# Find Python libraries and include directories
find_package(Python3 COMPONENTS Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
message("${BoldGreen}Python3 include directories: ${Python3_INCLUDE_DIRS}${ColourReset}")
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

# Check if PYBIND11_INCLUDE_DIR is found
if(NOT PYBIND11_INCLUDE_DIR)
message("${BoldRed}Pybind11 include directory not found.${ColourReset}")
else()
message("${BoldGreen}Pybind11 include directory: ${PYBIND11_INCLUDE_DIR}${ColourReset}")
endif()
include_directories(${PYBIND11_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PYTHON_LIBRARY})
    
