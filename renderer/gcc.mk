CC = g++
#GENERALFLAGS = -fPIC -W -Wall -Wextra -g -pedantic -std=c++14
GENERALFLAGS = -fPIC -std=c++14
ifeq ($(DEBUG_MODE), 1)
    OPTIMFLAGS = -march=native -O0 -ffast-math -fopenmp -pthread '-pipe' '-march=native' '-msse2' '-ftree-vectorize' '-mfpmath=sse' '-funsafe-math-optimizations' '-fno-rounding-math' '-fno-signaling-nans' '-fno-math-errno' '-fomit-frame-pointer'
else
    OPTIMFLAGS = -march=native -O3 -ffast-math -fopenmp -pthread '-pipe' '-march=native' '-msse2' '-ftree-vectorize' '-mfpmath=sse' '-funsafe-math-optimizations' '-fno-rounding-math' '-fno-signaling-nans' '-fno-math-errno' '-fomit-frame-pointer'
endif
#REPORTSFLAGS = -Winline -Wimplicit
REPORTSFLAGS =
#DEBUGFLAG = -g
CFLAGS += $(DEBUGFLAG) $(GENERALFLAGS) $(OPTIMFLAGS)
ifeq ($(DEBUG_MODE), 0)
	CFLAGS += -DNDEBUG
endif
ifeq ($(PRODUCE_REPORTS), 1)
	CFLAGS += $(REPORTSFLAGS) 
endif

PYFLAGS = -O3 -Wall -shared -std=c++14 -fPIC -march=native
PYFLAGS += -ffast-math -fopenmp -pthread
PYFLAGS += -DUSE_GCC -DUSE_SFMT 
ifeq ($(DEBUG_MODE), 0)
	PYFLAGS += -DNDEBUG
endif
