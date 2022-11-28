## compilation options
USE_GCC = 1
PRODUCE_REPORTS = 0
DEBUG_MODE = 0
USE_PRINTING = 0
PRINT_DEBUGLOG = 0
RUNTIME_DEBUGLOG = 1
PROJECTOR = 1

## MATLAB options
USE_MATLAB = 0

## Pybind options
USE_PYBIND = 1

## sampler options
USE_SFMT = 1

## CERES options
USE_CERES = 0

## OpenEXR options
USE_OPENEXR = 0

## optimization options
USE_THREADED = 1
USE_DOUBLE_PRECISION = 1
USE_PIXEL_SHARING = 1
USE_NO_FRESNEL = 0
USE_FILE = 0
USE_WEIGHT_NORMALIZATION = 0
USE_SIMPLIFIED_TIMING = 0
USE_OMEGA_TRACKING = 0
USE_CUDA = 0

## whether to use bessel RIF or spline RIF or FUS RIF. cannot have both USE_SPLINE_RIF and USE_FUS_RIF
USE_SPLINE_RIF = 0
USE_FUS_RIF = 0

# initialize variables
LIBS =
INCLUDES =
CFLAGS =
LDFLAGS =

# Use floats with cuda as doubles are very slow
ifeq ($(USE_CUDA), 1)
	USE_DOUBLE_PRECISION = 0
	PRINT_DEBUGLOG = 0
endif

## specify compiler, adds to CFLAGS
ifeq ($(USE_GCC), 1)
	include gcc.mk
	CFLAGS += -DUSE_GCC
	PYFLAGS += -DUSE_GCC
else
	include icc.mk
endif

## specify boost version to use
include boost.mk

## specify MATLAB flags, adds to INCLUDES and CFLAGS
ifeq ($(USE_MATLAB), 1)
	include matlab.mk
endif

## specify sampler-related options
# specify RNG to use
ifeq ($(USE_RNG), BOOST)
	CFLAGS += -DUSE_BOOST
	PYFLAGS += -DUSE_BOOST
endif

# specify algorithm to use
ifeq ($(USE_SFMT), 1)
	CFLAGS += -DUSE_SFMT
	PYFLAGS += -DUSE_SFMT
endif

## specify CERES flags, adds to LIBS and INCLUDES, available only with GCC
ifeq ($(USE_CERES), 1)
ifeq ($(USE_GCC), 1)
	include ceres.mk
endif
endif

## specify OPENEXR flags, adds to LIBS and INCLUDES, available only with GCC
ifeq ($(USE_OPENEXR), 1)
ifeq ($(USE_GCC), 1)
	include openexr.mk
	CFLAGS += -DUSE_OPENEXR
	PYFLAGS += -DUSE_OPENEXR
endif
endif

ifeq ($(USE_THREADED), 1)
	CFLAGS += -DUSE_THREADED
	PYFLAGS += -DUSE_THREADED
endif

ifeq ($(USE_DOUBLE_PRECISION), 1)
	CFLAGS += -DUSE_DOUBLE_PRECISION
	PYFLAGS += -DUSE_DOUBLE_PRECISION
endif

ifeq ($(USE_DIRECT), 1)
	CFLAGS += -DUSE_DIRECT
	PYFLAGS += -DUSE_DIRECT
endif

ifeq ($(USE_PIXEL_SHARING), 1)
	CFLAGS += -DUSE_PIXEL_SHARING
	PYFLAGS += -DUSE_PIXEL_SHARING
endif

ifeq ($(USE_NO_FRESNEL), 1)
	CFLAGS += -DUSE_NO_FRESNEL
	PYFLAGS += -DUSE_NO_FRESNEL
endif

ifeq ($(USE_FILE), 1)
	CFLAGS += -DUSE_FILE
	PYFLAGS += -DUSE_FILE
endif

ifeq ($(USE_WEIGHT_NORMALIZATION), 1)
	CFLAGS += -DUSE_WEIGHT_NORMALIZATION
	PYFLAGS += -DUSE_WEIGHT_NORMALIZATION
endif

ifeq ($(PRINT_DEBUGLOG), 1)
	CFLAGS += -DPRINT_DEBUGLOG
	PYFLAGS += -DPRINT_DEBUGLOG
endif

ifeq ($(RUNTIME_DEBUGLOG), 1)
	CFLAGS += -DRUNTIME_DEBUGLOG
	PYFLAGS += -DRUNTIME_DEBUGLOG
endif

ifeq ($(PROJECTOR), 1)
	CFLAGS += -DPROJECTOR
	PYFLAGS += -DPROJECTOR
endif

ifeq ($(USE_PRINTING), 1)
	CFLAGS += -DUSE_PRINTING
	PYFLAGS += -DUSE_PRINTING
endif

ifeq ($(USE_SIMPLIFIED_TIMING), 1)
    CFLAGS += -DUSE_SIMPLIFIED_TIMING
    PYFLAGS += -DUSE_SIMPLIFIED_TIMING
endif

ifeq ($(USE_OMEGA_TRACKING), 1)
    CFLAGS += -DOMEGA_TRACKING
    PYFLAGS += -DOMEGA_TRACKING
endif

ifeq ($(USE_SPLINE_RIF), 1)
    CFLAGS += -DSPLINE_RIF
    PYFLAGS += -DSPLINE_RIF
endif

ifeq ($(USE_FUS_RIF), 1)
    CFLAGS += -DFUS_RIF
    PYFLAGS += -DFUS_RIF
endif

ifeq ($(USE_CUDA), 1)
	CFLAGS += -DUSE_CUDA
	PYFLAGS += -DUSE_CUDA
endif

ifeq ($(USE_MATLAB), 1)
	MATLIBS += $(LIBS)
endif


# ROOTDIR = $(shell pwd)
ROOTDIR = .
INCLUDEDIR = $(ROOTDIR)/include
SRCDIR = $(ROOTDIR)/src
MEXDIR = $(ROOTDIR)/mexfiles
PYDIR = $(ROOTDIR)/pybindFiles
SAMPLEDIR = $(ROOTDIR)/samples

INCLUDES += -I$(INCLUDEDIR)
PYBIND_INCLUDE := $(shell echo `python3 -m pybind11 --includes`)
PYBIND_SUFF := $(shell echo `python3-config --extension-suffix`)

HEADERS = \
	$(INCLUDEDIR)/constants.h \
	$(INCLUDEDIR)/debug_util.h \
	$(INCLUDEDIR)/image.h \
	$(INCLUDEDIR)/medium.h \
	$(INCLUDEDIR)/phase.h \
	$(INCLUDEDIR)/renderer.h \
	$(INCLUDEDIR)/sampler.h \
	$(INCLUDEDIR)/spline.h \
	$(INCLUDEDIR)/tvector.h \
	$(INCLUDEDIR)/matrix.h \
	$(INCLUDEDIR)/util.h \
	$(INCLUDEDIR)/photon.h \
	$(INCLUDEDIR)/scene.h \
	$(INCLUDEDIR)/warp.h \
	$(INCLUDEDIR)/vmf.h \
	$(INCLUDEDIR)/bsdf.h

OBJ = \
	$(SRCDIR)/image.o \
	$(SRCDIR)/phase.o \
	$(SRCDIR)/sampler.o \
	$(SRCDIR)/matrix.o \
	$(SRCDIR)/photon.o \
	$(SRCDIR)/scene.o \
	$(SRCDIR)/warp.o \
	$(SRCDIR)/vmf.o \
	$(SRCDIR)/bsdf.o

PYBIND_OBJS = \
    $(PYDIR)/scene_pybind \
    $(PYDIR)/photon_pybind \
    $(PYDIR)/tvector_pybind \
    $(PYDIR)/medium_pybind \
    $(PYDIR)/phase_pybind \
    $(PYDIR)/image_pybind


ifeq ($(USE_SFMT), 1)
	HEADERS += $(INCLUDEDIR)/rng_sse.h $(INCLUDEDIR)/sse.h
	OBJ += $(SRCDIR)/rng_sse.o
else
	HEADERS += $(INCLUDEDIR)/rng_boost.h
endif

ifeq ($(USE_CUDA), 1)
	LIBS+= -L/usr/local/cuda/lib64/ -lcudart -lcurand
	INCLUDES+=-I/usr/local/cuda/include/

	NVCCFLAGS= -O3 -use_fast_math -maxrregcount 64 -m64 -ccbin $(CC) -std=c++14 -arch=sm_75
	NVCC=nvcc

	HEADERS += \
		$(INCLUDEDIR)/cuda_renderer.h \
		$(INCLUDEDIR)/cuda_image.cuh \
		$(INCLUDEDIR)/cuda_scene.cuh \
		$(INCLUDEDIR)/cuda_utils.cuh \
		$(INCLUDEDIR)/cuda_vector.cuh

	OBJ += \
		$(SRCDIR)/cuda_renderer.o

endif

ifeq ($(USE_MATLAB), 1)
all: mexfiles
else ifeq ($(USE_PYBIND), 1)
all: $(PYBIND_OBJS)
else
all: samples
endif

samples: \
	$(SAMPLEDIR)/renderer3d_sample_bin

mexfiles: \
	$(MEXDIR)/renderImage3d_mex.$(MEXEXT)

$(SAMPLEDIR)/%_bin: \
	$(SAMPLEDIR)/%.o \
	$(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

$(MEXDIR)/%.$(MEXEXT): \
	$(MEXDIR)/%.o \
	$(OBJ)
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $^ $(MATLIBS)

# sample object files
$(SAMPLEDIR)/%.o: $(SAMPLEDIR)/%.cpp $(INCLUDEDIR)/renderer.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $(SAMPLEDIR)/$*.cpp

# mex object files
$(MEXDIR)/%.o: $(MEXDIR)/%.cpp \
	$(INCLUDEDIR)/mex_wrapper.h \
	$(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# src object files
$(SRCDIR)/cuda_%.o: \
	$(SRCDIR)/cuda_%.cu \
	$(HEADERS)
	$(NVCC) $< $(NVCCFLAGS) -Xcompiler "$(CFLAGS)" $(INCLUDES) -c -o $@

$(SRCDIR)/rng_sse.o: \
	$(SRCDIR)/rng_sse.cpp \
	$(INCLUDEDIR)/rng_sse.h \
	$(INCLUDEDIR)/sse.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(SRCDIR)/rng_%.o: \
	$(SRCDIR)/rng_%.cpp \
	$(INCLUDEDIR)/rng_%.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(SRCDIR)/%.o: \
	$(SRCDIR)/%.cpp \
	$(HEADERS)
	$(CC) $< $(CFLAGS) $(INCLUDES) -c -o $@

ifeq ($(USE_CUDA), 1)
# To generate photon_pybind, must have object files
$(PYDIR)/photon_pybind: $(OBJ)
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(PYFLAGS)" $(PYBIND_INCLUDE) $@.cpp -o $@$(PYBIND_SUFF) -I$(INCLUDEDIR) $^ $(LIBS)

# Other pybinds
$(PYDIR)/%_pybind:
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(PYFLAGS)" $(PYBIND_INCLUDE) $@.cpp -o $@$(PYBIND_SUFF) -I$(INCLUDEDIR)
else
# To generate photon_pybind, must have object files
$(PYDIR)/photon_pybind: $(OBJ)
	$(CC) $(PYFLAGS) $(PYBIND_INCLUDE) $@.cpp -o $@$(PYBIND_SUFF) -I$(INCLUDEDIR) $^ $(LIBS)

# Other pybinds
$(PYDIR)/%_pybind:
	$(CC) $(PYFLAGS) $(PYBIND_INCLUDE) $@.cpp -o $@$(PYBIND_SUFF) -I$(INCLUDEDIR)
endif

clean:
	rm -rf *.o *~
	rm -rf $(MEXDIR)/*.o $(MEXDIR)/*~
	rm -rf $(SAMPLEDIR)/*.o $(SAMPLEDIR)/*~
	rm -rf $(SRCDIR)/*.o $(SRCDIR)/*~

distclean:
	rm -rf *.o *~
	rm -rf $(PYDIR)/*.cpython*
	rm -rf $(MEXDIR)/*.o $(MEXDIR)/*~ $(MEXDIR)/*.$(MEXEXT)
	rm -rf $(SAMPLEDIR)/*.o $(SAMPLEDIR)/*~ $(SAMPLEDIR)/*_bin
	rm -rf $(SRCDIR)/*.o $(SRCDIR)/*~
