To compile pyx files:

```
$ export C_INCLUDE_PATH=/usr/local/Cellar/python/3.7.4/Frameworks/Python.framework/Versions/3.7/Headers:/usr/local/lib/python3.7/site-packages/numpy/core/include

$ python3 setup.py build_ext --inplace

```