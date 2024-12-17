cmake_minimum_required(VERSION 3.20)

# Define the project name as a variable

# Set the VCPKG_ROOT variable from the environment
set(VCPKG_ROOT $ENV{VCPKG_ROOT})

# Enable verbose makefile output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Debug mode flag (set to 1 for debug mode, 0 for release mode)
set(DEBUG_MODE 1)

# Flag to use CUDA (set to 1 to enable CUDA, 0 to disable)
set(USE_CUDA 1)

# Flag to use Ceres Solver (set to 1 to enable, 0 to disable)
set(USE_CERES 1)

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
add_compile_options(${CERESFLAGS})

# Base paths for vcpkg and other specific libraries
set(VCPKG_PACKAGES_SO_LIB_DIR ${VCPKG_ROOT}/installed/x64-linux-dynamic/lib)

# Libraries to link against
set(LIBS
    # /opt/ATLAS/lib/libf77blas.a
    # /opt/ATLAS/lib/libatlas.a
    # /usr/lib/x86_64-linux-gnu/librt.so
    -lrt
    -lpthread
)

# Define the project name
set(PROJECT_NAME simplER)
project(${PROJECT_NAME})

# Print build directories
message("Build directory: ${CMAKE_BINARY_DIR}")
message("Project Build directory: ${PROJECT_BINARY_DIR}")

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

# Add executable target
add_executable(${PROJECT_NAME} simplER.cpp)

# Link libraries to the target
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::core Boost::algorithm 
                                              Boost::static_assert Boost::random 
                                              Boost::iterator Boost::math)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS::OpenBLAS)
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

# Link Ceres library if enabled
if(USE_CERES)
    target_link_libraries(${PROJECT_NAME} PRIVATE Ceres::ceres)
endif()
