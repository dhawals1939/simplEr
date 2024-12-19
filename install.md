# Installation Instructions

Install [vcpkg](https://github.com/microsoft/vcpkg.git)
Install [cmake](https://cmake.org/)
Install [matlab](https://www.mathworks.com/downloads/)

Set the `VCPKG_ROOT` to the directory with `vcpkg` binary. Also add this director to PATH.

My `vcpkg` is installed in `/opt/vcpkg` with write permisions
My `matlab` is installed in `/opt/matlab/R2024b`


Add these lines to `~/.bashrc`

```
# VCPKG
export VCPKG_ROOT="/opt/vcpkg"

# MATLAB
export MATLAB_DIR="/opt/matlab/R2024b"

export PATH="$VCPKG_ROOT:$MATLAB_DIR:$PATH"
```

## Linux

In linux machines

```
sudo apt-get update

sudo apt-get install gfortran pkg-config libatlas-base-dev

vcpkg install --triplet=x64-linux-dynamic

cmake --preset linux
```

If the build fails while building the CERES, you might have to link the `libopenblas.so` correctly.

`ln -s build/vcpkg_installed/x64-linux-dynamic/debug/lib/libopenblas_d.so build/vcpkg_installed/x64-linux-dynamic/debug/lib/libopenblas.so`



