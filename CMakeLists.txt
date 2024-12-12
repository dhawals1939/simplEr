cmake_minimum_required(VERSION 3.20)

# Define the project name as a variable

# Set the VCPKG_ROOT variable
set(VCPKG_ROOT $ENV{VCPKG_ROOT})

set(CMAKE_VERBOSE_MAKEFILE ON)

# Debug mode flag (set to 1 for debug mode, 0 for release mode)
set(DEBUG_MODE 1)

# Flag to use CUDA (set to 0 by default)
set(USE_CUDA 1)

set (USE_CERES 1)

# Flag to produce reports (set to 0 by default)
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

# Set the vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "/opt/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Path to vcpkg toolchain")

# Use the triplet specified in the presets or fallback to x64-linux-dynamic
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "x64-linux-dynamic" CACHE STRING "Specify vcpkg triplet")
endif()

# # Find the fmt library
# find_package(fmt CONFIG REQUIRED)

# # Find the glog library
# find_package(glog CONFIG REQUIRED)


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

# Include directories for Ceres
set(CERESINCLUDE -isystem /usr/local/eigen3)
include_directories(${CERESINCLUDE})

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
add_compile_options(${CERESFLAGS})

set(VCPKG_PACKAGES_BASE_DIR ${CMAKE_BINARY_DIR}/vcpkg_installed/x64-linux-dynamic/lib)

# Print the top-level build directory
message("Build folder: ${CMAKE_BINARY_DIR}")

# Print all paths
message(STATUS "Dynamic Library Files: ${CMAKE_BINARY_DIR}")

set(LIBS
    ${VCPKG_PACKAGES_BASE_DIR}/libceres.so
    
    ${VCPKG_PACKAGES_BASE_DIR}/libglog.so
    
    ${VCPKG_PACKAGES_BASE_DIR}/libgflags.so

    ${VCPKG_PACKAGES_BASE_DIR}/libspqr.a
    ${VCPKG_PACKAGES_BASE_DIR}/libcholmod.a
    ${VCPKG_PACKAGES_BASE_DIR}/libccolamd.a
    ${VCPKG_PACKAGES_BASE_DIR}/libcamd.a
    ${VCPKG_PACKAGES_BASE_DIR}/libcolamd.a
    ${VCPKG_PACKAGES_BASE_DIR}/libamd.a
    ${VCPKG_PACKAGES_BASE_DIR}/libsuitesparseconfig.a
    ${VCPKG_PACKAGES_BASE_DIR}/libcxsparse.a
    
    ${VCPKG_PACKAGES_BASE_DIR}/liblapack.a
    
    ${VCPKG_PACKAGES_BASE_DIR}/libopenblas.a
    
    # ${EIGEN3_LIB_DIR}/libeigen3.a
    # /opt/ATLAS/lib/libf77blas.a
    # /opt/ATLAS/lib/libatlas.a

    /usr/lib/x86_64-linux-gnu/librt.so
    -lpthread
)

set(PROJECT_NAME simplER)
project(${PROJECT_NAME})



# Combine all flags
add_compile_options(${GENERALFLAGS} ${OPTIMFLAGS} ${REPORTSFLAGS})

# Find packages
find_package(Eigen3 CONFIG REQUIRED)

# Add executable target
add_executable(${PROJECT_NAME} simplER.cpp)

# Link libraries
find_package(fmt CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(boost_static_assert CONFIG REQUIRED)
find_package(boost_core CONFIG REQUIRED)
find_package(boost_iterator CONFIG REQUIRED)
find_package(boost_algorithm CONFIG REQUIRED)
find_package(boost_random CONFIG REQUIRED)
find_package(boost_math CONFIG REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)
# find_package(Ceres CONFIG REQUIRED)

# # Link all libraries to the target
target_link_libraries(${PROJECT_NAME}
PRIVATE
fmt::fmt
Eigen3::Eigen
Boost::static_assert
Boost::core
Boost::iterator
Boost::algorithm
Boost::random
Boost::math
OpenBLAS::OpenBLAS
)

if(USE_CERES)
# find_package(Ceres CONFIG REQUIRED)
# target_link_libraries(${PROJECT_NAME} PRIVATE Ceres::ceres)
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
endif()