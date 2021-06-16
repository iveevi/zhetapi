The ZHP Scripting Language
==========================

The ZHP scripting language is a dynamically typed language, and as of writing
this documentation, it is an interpreted language (for increased performace, it
will later use JIT compilation).

Philosophy
----------

This scripting language was designed to be used for mathematical purposes. It is
not a general purpose language, but rather a language created for young
mathematicians and professionals alike to experiment with numerical and machine
learning algorithms, or to solve arbitrary math problems. The syntax resembles
that of Python to keep simplicity, and uses certain syntax from C and C++ to
keep some structure.

Variables
---------

Treat variables simply as value-holders, similar to Python. However, there are
two main types for these values: operands and operators.

An operand is anything that can be operated on: integers, rational numbers,
reals, etc., are all valid operands. Assigning operands to variables is shown
below.

.. code-block::

        # Integer
        x = 10

        # Rational
        y = 5/3

        # Real
        z = 1.66

        # Vector with complex components
        v = [i, 0.16i, 5i/3]

Operators are objects which take in a set of operands and output an operand; it
is the generalization of functions, lamdas, etc. The following are the specific
ways to define and assign operators.

Functions
~~~~~~~~~

**Functions in ZHP are NOT the same as functions in Python, C, etc.** This
distinction is made to keep consistent with mathematics. Functions in ZHP are
like lambdas in Python, and represent mathematical functions.

One defines functions as follows:

.. code-block::

        f(x) = x^2 - x * sin(x)

        # g and f are the same function
        g = f

Algorithms
~~~~~~~~~~

Algorithms in ZHP are like functions in Python and similar languages. For now
consider this simple example.

.. code-block::

        # An overly simplified example
        alg foo()
        {
                println("Computing the answer to life...")

                return 42
        }

For more details on algorithms, see [].

Registrables
~~~~~~~~~~~~

Registrables are the last of the operators. They are ZHP's built-in method of
allowing users to create bindings for ZHP operators in other languages (C/C++,
etc.).

Branching
---------

Branching is done using the same keywords as Python: ``if``, ``elif`` and ``elif``.
The corresponding block goes in the line after the declaration of the branch, or
in a block enclosed by braces.

.. code-block::

        if ([condition1])
                [block1]        // Single line block
        
        if ([condition2]) {
                [block2]        // Can be multiline
        } elif ([condition3]) {
                [block3]
        } else {
                [block4]
        }

Conditions must evaluate to boolean expressions (``true`` or ``false``) or otherwise
must be of the form ``[variable1] in [variable2]``, where ``[variable2]`` must be a
Set primitive type. Implicit conditions from C/C++ (ie. ``if (1)`` or ``if([pointer])``)
may be added in the future.

.. toctree::
        :maxdepth: 2
        :hidden: