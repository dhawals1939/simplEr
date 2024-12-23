# simplEr: simple renderer for ultrasonically sculpted gradient-index waveguides inside scattering volumes

This renderer was developed as part of the following three projects:
- [Path tracing estimators for refractive radiative transfer](https://imaging.cs.cmu.edu/rrte/)
- [Optimized virtual optical waveguides enhance light throughput in scattering media](https://imaging.cs.cmu.edu/optimized_virtual_optical_waveguides/)
- [Megahertz light steering without moving parts](https://imaging.cs.cmu.edu/ultrafast_steering/)

The renderer has the following capabilities:
1. Cylindrical transducer: The renderer can simulate arbitrary ultrasound (US) amplitude, frequency, mode (0, 1, 2 ....); light source duty cycle (0 to 100%), input pattern (collimated, collimated and textured, diffused, diffused and textured); transducer length; medium optical properties (transparent, or scattering with varying albedo, anisotropy, scattering coefficient); thin lenses for sensor or light source. 
2. Arbitrarily shaped transducer: The renderer can simulate transducers of arbitrary shapes though at much slower speed.

## Code structure

The renderer is written in C++ and uses multiple open sources libraries. There is also a CUDA implementation for GPU deployment. Lastly, there a python bindings using `pybind`, alongside examples in python scripts and jupyter notebooks. 

## Compile-time build options

There are a few compile-time options to specify before building:
- Whether the build should parallelize across CPU cores or a GPU.
- Whether the build should run as a stand-alone executable or through the python interface.
- Whether the build should simulate a cylindrical transducer (fastest), or a sculpted refractive-index field specified as a 3D spline (slower), or a travelling-wave ultrasound (slowest).

## Installation notes

The renderer has been tested on Ubuntu 20.04 and 22.04 and you should be able to install and build it on those systems without issues.

If you are not a Linux user, you can use Docker to run the renderer from any operating system.

## #Installation instructions

Make sure you have installed `python3`, `python3-pybind11`, `g++11`, and that your python installation has the packages `matplotlib`, `numpy`, and `jupyterlab`.

Additionally, you need to install CUDA (including the driver) at version at least 10.6 or newer.

Edit the `Makefile` under `<simplEr_location>/renderer/` to specify build options:
- Setting `USE_CUDA = 1` will enable GPU rendering.
- Setting `USE_PYBIND = 1` will enable the python API.
- Setting `USE_FUS_RIF = 0` and `USE_SPLINE_RIF = 0` will enable the cylindrical transducer.

Then you can build the renderer by running `make`.

## Examples

### Using jupyter notebooks to simulate a cylindrical transducer
Build the renderer with options `USE_PYBIND = 1`, `USE_FUS_RIF = 0`, and `USE_SPLINE_RIF = 0`. Then use:
```
cd <simplEr_location>/renderer/pybindFiles/tests/
jupyter-notebook FUS_configurations.ipynb
```
Run all the blocks of the notebook to see an image rendered with ultrasound generated by a cylindrical transducer. 

### Using jupyter notebooks to simulate a traveling-wave ultrasound
Build the renderer with options `USE_PYBIND = 1`, `USE_FUS_RIF = 1`, and `USE_SPLINE_RIF=0`. Then use:
```
cd <simplEr_location>/renderer/pybindFiles/tests/
jupyter-notebook FUS_configurations.ipynb
```
Run the first seven blocks of the notebook to see an image rendered with two traveling wave transducer elements

### Using command-line executables to simulate a cylindrical transducer
Build the renderer with options `USE_PYBIND = 0`, `USE_FUS_RIF = 0`, and `USE_SPLINE_RIF = 0`. Then use:
```
cd <simplEr_location>/renderings/
sh testing.sh
```
You can view the output file (`PFM3D` format) using the `readpfm3d.m` script under `helperScripts`. 

### Using command-line executables to simulate a traveling-wave transducer
Build the renderer with options `USE_PYBIND = 0`, `USE_FUS_RIF = 1`, and `USE_SPLINE_RIF = 0`. Then use:
```
cd <simplEr_location>/renderings/
sh fus_rif.sh
```
You can view the output file (`PFM3D` format) using the `readpfm3d.m` script under `helperScripts`. 

## Cheat sheet

Setting `CUDA_VISIBLE_DEVICES = k` deploys the renderer on the `k + 1` available CUDA device. This is helpful for deployment on a GPU that is not used by the display.
