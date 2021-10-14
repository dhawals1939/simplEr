CC = g++
#GENERALFLAGS = -fPIC -W -Wall -Wextra -g -pedantic -std=c++14
GENERALFLAGS = -fPIC -g -std=c++14
OPTIMFLAGS = -march=native -O0 -ffast-math -fopenmp -pthread '-pipe' '-march=native' '-msse2' '-ftree-vectorize' '-mfpmath=sse' '-funsafe-math-optimizations' '-fno-rounding-math' '-fno-signaling-nans' '-fno-math-errno' '-fomit-frame-pointer'
#REPORTSFLAGS = -Winline -Wimplicit
REPORTSFLAGS =
DEBUGFLAG = -g
CFLAGS += $(DEBUGFLAG) $(GENERALFLAGS) $(OPTIMFLAGS)
ifeq ($(DEBUG_MODE), 0)
	CFLAGS += -DNDEBUG
endif
ifeq ($(PRODUCE_REPORTS), 1)
	CFLAGS += $(REPORTSFLAGS) 
endif
