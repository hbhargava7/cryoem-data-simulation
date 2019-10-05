# cryoem-data-simulation

Python3 tools for simulating cryoEM particle data.

Built on top of https://github.com/mbrubake/cryoem-cvpr2015

*Simulating Data*

```
$ python3 simulate_particles.py <input_mrc_path> <output_directory> --n_particles n
```
Flags:
`--snr`: Specify a signal to noise ratio for the particles
`--sigma_noise`: Specify a numeric standard deviation for the noise to be added (will override `--snr`)

### To compile the pyx files (in cryoem-cython-pyx)

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

