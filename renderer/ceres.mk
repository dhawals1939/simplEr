#CERESINCLUDE= -I/home/apedired/Dropbox/ceres/ceres-solver/ceres-bin/config -I/home/apedired/Dropbox/ceres/ceres-solver/include -isystem /usr/local/eigen3
CERESINCLUDE= -isystem /usr/local/eigen3
INCLUDES += $(CERESINCLUDE)

CERESFLAGS= -Wmissing-declarations -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers -DNDEBUG -DCERES_GFLAGS_NAMESPACE=google 
CFLAGS += $(CERESFLAGS)

#LIBS += /home/apedired/Dropbox/ceres/ceres-solver/ceres-bin/lib/libceres.a /usr/lib/x86_64-linux-gnu/libglog.so /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2 -lpthread /usr/lib/x86_64-linux-gnu/libspqr.so /usr/lib/x86_64-linux-gnu/libcholmod.so /usr/lib/x86_64-linux-gnu/libccolamd.so /usr/lib/x86_64-linux-gnu/libcamd.so /usr/lib/x86_64-linux-gnu/libcolamd.so /usr/lib/x86_64-linux-gnu/libamd.so /usr/lib/x86_64-linux-gnu/liblapack.so /usr/lib/x86_64-linux-gnu/libf77blas.so /usr/lib/x86_64-linux-gnu/libatlas.so /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so /usr/lib/x86_64-linux-gnu/librt.so /usr/lib/x86_64-linux-gnu/libcxsparse.so /usr/lib/x86_64-linux-gnu/liblapack.so /usr/lib/x86_64-linux-gnu/libf77blas.so /usr/lib/x86_64-linux-gnu/libatlas.so /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so /usr/lib/x86_64-linux-gnu/librt.so /usr/lib/x86_64-linux-gnu/libcxsparse.so
LIBS += /lib/libceres.so /lib/libglog.so /lib/libgflags.so.2.2.2 -lpthread /lib/libspqr.so /lib/libcholmod.so /lib/libccolamd.so /lib/libcamd.so /lib/libcolamd.so /lib/libamd.so
LIBS += /lib/liblapack.so /opt/ATLAS/lib/libf77blas.a /opt/ATLAS/lib/libatlas.a /lib/libsuitesparseconfig.so /lib/librt.so /lib/libcxsparse.so /lib/liblapack.so /opt/ATLAS/lib/libf77blas.a /lib/libsuitesparseconfig.so /lib/librt.so /lib/libcxsparse.so
