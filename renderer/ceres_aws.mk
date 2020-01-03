CERESINCLUDE= -I/home/ubuntu/ceres-bin/config -I/home/ubuntu/ceres-solver-1.14.0/include -isystem /usr/local/eigen3 
INCLUDES += $(CERESINCLUDE)

CERESFLAGS= -Wmissing-declarations -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers -DNDEBUG -DCERES_GFLAGS_NAMESPACE=google 
CFLAGS += $(CERESFLAGS)

LIBS += -rdynamic /home/ubuntu/ceres-bin/lib/libceres.a -lglog -lgflags -lpthread -lspqr -lcholmod -lccolamd -lcamd -lcolamd -lamd -llapack -lf77blas -latlas -lsuitesparseconfig -lrt -lcxsparse -lgomp -lpthread -lspqr -lcholmod -lccolamd -lcamd -lcolamd -lamd -llapack -lf77blas -latlas -lsuitesparseconfig -lrt -lcxsparse -lgomp

