# cryoem-data-simulation

Simulate cryoEM particle data.

Built on top of https://github.com/mbrubake/cryoem-cvpr2015

### To compile the pyx files (in cryoem-cython-pyx)

```
$ export C_INCLUDE_PATH=/usr/local/Cellar/python/3.7.4/Frameworks/Python.framework/Versions/3.7/Headers:/usr/local/lib/python3.7/site-packages/numpy/core/include

$ python3 setup.py build_ext --inplace

```
