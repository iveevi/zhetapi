![Zhetapi Logo](zhetapi_logo.svg)

![build badge](https://github.com/vedavamadathil/zhetapi/actions/workflows/cmake.yml/badge.svg)

Zhetapi is a modern C++ machine learning and numerical analysis library with an
emphasis on intuitive usage.

# Linear Algebra

Zhetapi provides a basic linear algebra interface using C++ 20 concepts. The
`Field` structure enforces a contraint on types to ensure they behave like
algebraic fields. As a result these structures support basic arithmetic along
with other useful methods.

For now, the following template classes are provided: `Tensor`, `Matrix`,
`Vector`.

# Auto Differentiation

The auto differentiation facilities in Zhetapi belong in the `zhetapi::autograd`
namespace. All operations that depend on autodiff use `float`s as the underlying
type; in particular `Constant` is a `Tensor <float>` and is the basis of all
numerical values in this module.

To provide a seamless, operator based interface into the autodiff facilities,
two notable classes are provided, `Variable` and `Function`. As one can expect,
`Variable`s can store arbitrary `Constant` values, and `Function`s are
compositions of `Variables` under varying operations. For example:

```cpp
Function f = x + y;
Function g = x * y;

// f and g are now functions of *two* variables
Constant a = f(1, 2);
Constant b = g(1, 2);

// Composition of functions is done likewise
Function h = f(x, g(x, y)); // NOTE: h is still a function of two variables
```

## Symbolic Differentiation

## Backward Pass

Currently, only backward mode is enabled for autograd.

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
