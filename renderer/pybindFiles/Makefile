
INCLUDES =

PYBINDDIR = ../pybindFiles
INCLUDEDIR = ../include
SRCDIR = ../src
SAMPLEDIR = ../samples

PYOBJDIR = ./bin
TESTDIR = ./tests

INCLUDES += -I$(INCLUDEDIR)
PYBIND_INCLUDE := $(shell echo `python3 -m pybind11 --includes`)
PYBIND_SUFF := $(shell echo `python3-config --extension-suffix`)


CC = g++
CFLAGS = -g -fPIC -W -Wall -Wextra -pedantic -std=c++14 -march=native -O3
CFLAGS += -ffast-math -fopenmp -pthread
CFLAGS += -DNDEBUG -DUSE_GCC -DUSE_SFMT -DUSE_DOUBLE_PRECISION -DUSE_PIXEL_SHARING

PY_CFLAGS = -O3 -Wall -shared -std=c++14 -fPIC -march=native
PY_CFLAGS += -ffast-math -fopenmp -pthread
PY_CFLAGS += -DNDEBUG -DUSE_GCC -DUSE_SFMT -DUSE_DOUBLE_PRECISION -DUSE_PIXEL_SHARING


OBJ = \
	$(SRCDIR)/scene.o \
	$(SRCDIR)/bsdf.o \
	$(SRCDIR)/phase.o \
	$(SRCDIR)/photon.o \
	$(SRCDIR)/sampler.o \
	$(SRCDIR)/rng_sse.o \
	$(SRCDIR)/image.o \

PYBIND_FILES = \
  image_pybind


all: \
	$(PYBIND_FILES)

# To generate src object files:
$(SRCDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $^

# To generate photon_pybind, must have object files
photon_pybind: $(OBJ)
	$(CC) $(PY_CFLAGS) $(PYBIND_INCLUDE) $@.cpp -o $(PYOBJDIR)/$@$(PYBIND_SUFF) -I$(INCLUDEDIR) $^

# Other pybinds
%_pybind:
	$(CC) $(PY_CFLAGS) $(PYBIND_INCLUDE) $@.cpp -o $(PYOBJDIR)/$@$(PYBIND_SUFF) -I$(INCLUDEDIR)


clean:
	rm -rf *.o *~
	rm -rf $(SAMPLEDIR)/*.o $(SAMPLEDIR)/*~
	rm -rf $(SRCDIR)/*.o $(SRCDIR)/*~

distclean:
	rm -rf *.o *~
	rm -rf $(PYOBJDIR)/*.so *~
	rm -rf $(SAMPLEDIR)/*.o $(SAMPLEDIR)/*~ $(SAMPLEDIR)/*_bin
	rm -rf $(SRCDIR)/*.o $(SRCDIR)/*~
