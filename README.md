![alt text][logo]

# Zhetapi ℤHΠ

Zhetapi (ℤHΠ) is a C++ computation library that was built in the hopes of
making mathematical computation and algorithmic research more convenient to the
users.

This project is a work in progress.

## Usage

The Zhetapi library comes with many abilities. They are listed below:

* **Evaluation of complex expressions:**

The library can evaluate complex expressions, which have operands of various
types, such as integers, rational numbers, complex numbers, vectors and
matrices.

The framework of library allows the evaluate to be sensetive to certain types of
operations and their corresponding operations. For example, multiplication of
two integers yields and integers, the division of two rational numbers stays
rational, and the product of an matrix with integer compenents with a rational
scalar yields a matrix with rational components.

* **Customization of operands:**

As mentioned in above, the engine is sensetive to overloads of certain
operations. In addition, users can create their own set of operations and
corresponding overloads or add more overloads.

As of right now, however, due to the way in which the engine parses expressions,
one cannot add new symbols for operations.

* **Usage and Declaration of Variables:**

The library provides tools which allow the user to store variables and retrieve
them in the scope of a certain setting space. Users can then refer to these
variables, and their values can be retrieved or changed.

* **User Defined Functions:**

Users can create their own mathematical functions, which can then be used as any
other C++ functor object.

* **Linear Algbebra:**

The library also provides way in which the user can do linear algebra. The
classes `Vector` and `Matrix` come with a variety of methods on their own, which
include performing computation as well as manipulation of their representations.

In addition to these classes, the library provides standard algorithms such Gram
Shmidt and LU Factorization (see below).

### Overview of Usable Classes

Below are the currently usable classes.

| Class Name	| Description			|
| :----------:	| -----------------------------	| 
| `Vector`	| A vector in linear algebra	|
| `Matrix`	| A matrix in linear algebra	|
| `Polynomial`	| A polynomial in algebra	|

## Most Recent Stable Commit

https://github.com/vedavamadathil/zhetapi/tree/2d9112b98cf730239396f125b4f1f0680d5021c0

[logo]: https://github.com/vedavamadathil/zhetapi/blob/master/zhetapi-logo.png
