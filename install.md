# Installation Instructions

Install [vcpkg](https://github.com/microsoft/vcpkg.git)
Install [cmake](https://cmake.org/)

Set the `VCPKG_ROOT` to the directory with `vcpkg` binary. Also add this director to PATH.

My `vcpkg` is installed in `/opt` with write permisions

```
# VCPKG
export VCPKG_ROOT="/opt/vcpkg"
export PATH="$VCPKG_ROOT:$PATH"
```

## Linux

In linux machines

```
sudo apt-get update

sudo apt-get install gfortran pkg-config

vcpkg install --triplet=x64-linux-dynamic

cmake --preset linux
```



