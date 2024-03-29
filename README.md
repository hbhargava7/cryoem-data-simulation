# cryoem-data-simulation

Python3 tools for simulating cryoEM particle data, analyzing alignments, perturbing data, and more.

Built with help from https://github.com/mbrubake/cryoem-cvpr2015 and https://github.com/asarnow/pyem.

### Features
**Data Simulation**
* `simulate_particles.py`: Core script for multithreaded simulation of cryoEM particle stacks.
* `simulate_dual_stacks.py`: Simulate two stacks from two different volumes but match noise, ctf, and projection angle per particle between the stacks.
* `simulate_dual_stacks_rv.py`: Like `simulate_dual_stacks` but specify two **sets** of input volumes and probability weights (e.g. to model particle heterogeneity or classes).

**Alignment Analysis**
* `analyze_alignment.py`: Given experimental alignment parameters (e.g. cisTEM .par file) and theoretical parameters (e.g. from `simulate_dual_stacks.py`) from two different reconstructions, compare the distributions of angular and positional  errors.

**Tellurification**
* `tellurify.py`: Replace methionines with telluromethionines or phenylalanines with tellurienylalanines in a PDB file.

**Data Perturbation**
* `perturb_alignment.py`: Given a star file with some Euler angles, perturb these orientations by a characteristic overall angular error.


### Simulating Data

Note: The `pyx` modules in `cryoem-cython-pyx` need to be compiled for your system. See below.

*Simulate data*
```
$ python3 simulate_particles.py <input_mrc_path> <output_directory> --n_particles n --snr s
```
Other Flags:
* `--snr`: Specify a signal to noise ratio for the particles
* `--sigma_noise`: Specify a numeric standard deviation for the noise to be added (will override `--snr`)
* `--cpus`: Specify the number of processors to use (otherwise will use (n_cpus_available - 1)
* `--overwrite`: Don't prompt user for confirmation if the output directory exists and will be overwritten
* The microscope and CTF parameters can be changed in the `params` dict in main().

*Inputs*
* `input_mrc`: mrc volume from which the volume data, pixel size, and box size will be read

*Outputs (in output_directory)*
* `simulated_particles.mrcs`: mrcs particle stack containing the particle image data
* `simulated_particles.star`: Relion star file describing the simulated particles
* `simulation_metadata.txt`: Log of the simulation parameters and performance
* `plot.png`: Visualization of the first 8 particles

![plot example](reference/plot.png)

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

