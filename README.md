![Zhetapi Logo](zhetapi_logo.svg)

Zhetapi is a modern C++ machine learning and numerical analysis library with an
emphasis on intuitive usage.

# Building

Zhetapi is primarily a header-only library, but for now there are some examples
that one can play around with in the `experimental` directory.

This project is developed using C++ 20. Additional dependenies include PNG
(`libpng-dev` on Ubuntu systems), OpenMP (Optional) and CUDA (Optional).

Generate the build configuration using CMake as follows:

```
$ cd zhetapi
$ mkdir build && cd build
$ cmake -DZHETAPI_ENABLE_CUDA=<ON|OFF> # ON by default
```

And build the targets as one would usually do (e.g. `make` or `ninja`).
