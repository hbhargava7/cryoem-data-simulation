# cryoem-data-simulation

Python3 tools for simulating cryoEM particle data.

Built with code from https://github.com/mbrubake/cryoem-cvpr2015

### Usage

Note: The `pyx` modules in `cryoem-cython-pyx` need to be compiled for your system. See below.

*Simulate data*
```
$ python3 simulate_particles.py <input_mrc_path> <output_directory> --n_particles n --snr s
```
Other Flags:
* `--snr`: Specify a signal to noise ratio for the particles
* `--sigma_noise`: Specify a numeric standard deviation for the noise to be added (will override `--snr`)
* `--cpus`: Specify the number of processors to use (otherwise will use (n_cpus_available - 1)
* The microscope and CTF parameters can be changed in the `params` dict in main().

*Inputs*
* `input_mrc`: mrc volume from which the volume data, pixel size, and box size will be read

*Outputs (in output_directory)*
* `simulated_particles.mrcs`: mrcs particle stack containing the particle image data
* `simulated_particles.star`: Relion star file describing the simulated particles
* `plot.png`: Visualization of the first 8 particles

### To compile the pyx files (in cryoem-cython-pyx)
Compile using `clang` or `gcc` and then either move or symlink the `.so` files into the `cryoem` directory.
*For macOS*
```
$ export C_INCLUDE_PATH=/usr/local/Cellar/python/3.7.4/Frameworks/Python.framework/Versions/3.7/Headers:/usr/local/lib/python3.7/site-packages/numpy/core/include

$ python3 setup.py build_ext --inplace
```
*For Linux*
```
$ export C_INCLUDE_PATH=/home/<username>/.local/lib/python3.7/site-packages/numpy/core/include
```

*Symlink the compiled modules*
```
$ cd cryoem
$ ln -s ../cryoem-cython-pyx <result_so_file> <symlink_name>
```

### Dependencies
* Python3
* numpy
* scipy
* cython
* pyfftw (optional, but creates significant speedup)
* matplotlib

