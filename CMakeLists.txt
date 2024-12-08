cmake_minimum_required(VERSION 3.20)

# Define the project name as a variable
set(PROJECT_NAME simplER)

project(${PROJECT_NAME})


# Set the path to vcpkg toolchain file
set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

# Find the fmt library
find_package(fmt CONFIG REQUIRED)

# Set the C++ compiler (optional; typically auto-detected)
set(CMAKE_CXX_COMPILER g++)

# Debug mode flag (set to 1 for debug mode, 0 for release mode)
set(DEBUG_MODE 1)

# Flag to use CUDA (set to 0 by default)
set(USE_CUDA 1)

# Flag to produce reports (set to 0 by default)
set(PRODUCE_REPORTS 0)

# General compiler flags
set(GENERALFLAGS -fPIC -std=c++17)

# Shared optimization flags
set(SHARED_OPTIMFLAGS -march=native -ffast-math -fopenmp -pthread -pipe -msse2 -ftree-vectorize -mfpmath=sse -funsafe-math-optimizations -fno-rounding-math -fno-signaling-nans -fno-math-errno -fomit-frame-pointer)

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
set(CERESFLAGS -Wmissing-declarations -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers -DNDEBUG -DCERES_GFLAGS_NAMESPACE=google)
add_compile_options(${CERESFLAGS})

# Libraries for Ceres
set(LIBS /lib/libceres.so /lib/libglog.so /lib/libgflags.so.2.2.2 -lpthread /lib/libspqr.so /lib/libcholmod.so /lib/libccolamd.so /lib/libcamd.so /lib/libcolamd.so /lib/libamd.so /lib/liblapack.so /opt/ATLAS/lib/libf77blas.a /opt/ATLAS/lib/libatlas.a /lib/libsuitesparseconfig.so /lib/librt.so /lib/libcxsparse.so)
if(USE_CERES)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})
endif()

# Combine all flags
add_compile_options(${GENERALFLAGS} ${OPTIMFLAGS} ${REPORTSFLAGS})

# Find packages
find_package(Eigen3 CONFIG REQUIRED)
if(USE_CERES)
    find_package(Ceres CONFIG REQUIRED)
endif()

# Add executable target
add_executable(${PROJECT_NAME} simplER.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt Eigen3::Eigen)
if(USE_CERES)
    target_link_libraries(${PROJECT_NAME} PRIVATE Ceres::ceres)
endif()