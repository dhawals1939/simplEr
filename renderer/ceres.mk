CERESINCLUDE= -I/home/igkiou/Dropbox/ceres/ceres-solver/ceres-bin/config -I/home/igkiou/Dropbox/ceres/ceres-solver/include -isystem /usr/local/eigen3 
INCLUDES += $(CERESINCLUDE)

CERESFLAGS= -Wmissing-declarations -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-missing-field-initializers -DNDEBUG -DCERES_GFLAGS_NAMESPACE=google 
CFLAGS += $(CERESFLAGS)

LIBS += /mnt/da64b98f-9fd9-4b2c-994e-ca7276846901/Dropbox/ceres/ceres-solver/ceres-bin/lib/libceres.a /usr/lib/x86_64-linux-gnu/libglog.so /usr/lib/x86_64-linux-gnu/libgflags.so -lpthread /usr/lib/x86_64-linux-gnu/libspqr.so /usr/lib/x86_64-linux-gnu/libtbbmalloc.so /usr/lib/x86_64-linux-gnu/libtbb.so /usr/lib/x86_64-linux-gnu/libcholmod.so /usr/lib/x86_64-linux-gnu/libccolamd.so /usr/lib/x86_64-linux-gnu/libcamd.so /usr/lib/x86_64-linux-gnu/libcolamd.so /usr/lib/x86_64-linux-gnu/libamd.so /usr/lib/liblapack.so /usr/lib/libf77blas.so /usr/lib/libatlas.so /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so /usr/lib/x86_64-linux-gnu/librt.so /usr/lib/x86_64-linux-gnu/libcxsparse.so /usr/lib/liblapack.so /usr/lib/libf77blas.so /usr/lib/libatlas.so /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so /usr/lib/x86_64-linux-gnu/librt.so /usr/lib/x86_64-linux-gnu/libcxsparse.so
