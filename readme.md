# Illcrawl - a high performance reconstruction engine for data from the Illustris simulation

## Features
illcrawl (the Illustris data crawling and reconstruction engine) is a comprehensive software package for the reconstruction and visualization of datasets from the illustris simulation. Its features include
  * Generating projections (i.e. images) of Dark Matter and baryonic material
  * Generating volumetric slices and tomographies
  * Render animations
  * Calculate spectra (currently, X-ray spectra with and without Chandra's instrumental response are supported)
  * Calculating radial profiles (e.g densities, mass, luminosity, ...)
  * Filter Illustris snapshots based on spatial selection criteria

These features can be used to visualize (in principle) arbitrary quantities based on the quantities that are saved in illustris data for each gas cell or dark matter particle. Illcrawl has built-in support for the most common quantities, and it is easy to add support for new ones.

It is possible to access the illcrawl algorithms either through easy-to-use command line tools provided by illcrawl itself, or by writing custom C++ code that makes use of the libillcrawl_core static library.

## High performance
Illcrawl takes advantage of modern parallel hardware. All computationally intensive code is parallelized for GPUs using OpenCL. Since OpenCL is vendor neutral, in principle any GPU from NVIDIA, AMD or even integrated GPUs from Intel should work (although the latter is not recommended for good performance).
However, your GPU must support at least OpenCL 1.2, which is typically well supported by any GPU which is not truly ancient.

Additionally, some parts of illcrawl are parallelized on a coarser level using MPI as well, allowing some very computationally intensive tasks to be accelerated by GPU clusters. This generally applies to tasks which involve calculating several images (animations, tomographies, spectra).
For example, for the calculation of an animation, the individual frames of the animation are distributed among the parallel MPI processes, which then in turn parallelize the rendering of their assigned frames on the GPU. Each illcrawl MPI process binds to one GPU,
hence you need as many illcrawl processes as you have GPUs to take advantage of all computational resources.

## Building
### Requirements
Hardware requirements:
  * A GPU supporting at least OpenCL 1.2. It is advised to use a recent GPU with a decent amount of VRAM.

Software requirements:
  * A C++11 compliant compiler
  * CMake >= 3.1
  * An MPI implementation. Illcrawl is tested with OpenMPI 2.1.
  * An OpenCL implementation (at least OpenCL 1.2 compliant)
  * The HDF5 libraries and their official C++ wrappers
  * The Boost C++ libraries >= 1.61 (in particular, boost.mpi, boost.serialization, boost.program_options, boost.compute)
  * cfitsio

Illcrawl itself should be fairly platform independent, but it is only tested on Unix-like (in particular Linux) systems. If you want to try it on Windows,
feel free to do so, but you will be pretty much on your own. In the following, I will assume that you are using a Unix-like system.

### Building illcrawl
Once all requirements are satisfied, building should be straightforward.
  1. Create a build directory:

  `$ mkdir build`
  2. Execute cmake. This will generate a Makefile taylored for your specific system.

  `$ cmake <Path to Illcrawl source directory>`
  3. Compile:

  `$ make`

This should create the following executables and libraries in your build directory:
  * libillcrawl_core.a
  * illcrawl_render
  * illcrawl_cluster_analysis
  * illcrawl_extract_fits_slice
  * illcrawl_filter_snapshot
  * illcrawl_sum_pixels

Additionally, a number of files ending with *.cl will also appear in your build directory. These are the OpenCL source files (i.e., the computationally intensive parts of illcrawl that will run on the GPU) which will be compiled at runtime by your OpenCL runtime environment.
It is therefore necessary that these files are in your current working directory when executing an illcrawl program.

