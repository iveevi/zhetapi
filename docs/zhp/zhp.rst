.. _zhp_page:

The ZHP Scripting Language
==========================

.. toctree::
        :maxdepth: 2
        :hidden:

        ..

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

        // Integer
        x = 10

        // Rational
        y = 5/3

        // Real
        z = 1.66

        // Vector with complex components
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

        // g and f are the same function
        g = f

Algorithms
~~~~~~~~~~

Algorithms in ZHP are like functions in Python and similar languages. For now
consider this simple example.

.. code-block::

        // An overly simplified example
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

Loops
-----

The ZHP scripting language supports two styles of looping: ``while`` looping and
``for`` looping. These are distinct and are generally not exchangable.

While Loops
~~~~~~~~~~~

While loops follow the same principles as from other languages. The body of the
loops is repeatedly executed as long as the condition is evaluate to ``true``.

.. code-block::


        // Loops can be single line...
        x = 0
        while (x < 10)
                println("x = ", x)
        
        // Or multiline, for larger bodies
        x = 256
        while (x > 0) {
                println("x = ", x)

                x = x / 2
        }

In the future, warning against possible infinite loops may be displayed for
monotonic inequalities (such as ``x > 0`` or ``x < 100``).

For Loops
~~~~~~~~~

All for loops must consist of a "generator" clause of the form ``[identifier] in
[expression]``, which is then followed by the body of the loop. The
``identifier`` in the generator clause must be a valid identifier, and
``expression`` must evaluate to a Generator type (such as a Collection). The
loop then iterates over the Generator, assigns the values to ``[indentifier]``
and executes the body of the loop.

.. code-block::

        // l is a collection of values
        l = {1, 2, "three", 4.0}

        for (x in l)
                println("x = ", l)
        
        // A more common and useful example
        for (x in range(4))
                println("l[x] = ", l[x])

Modules
-------

There are currently two ways of organizing source code into separate files and
libraries, both of which revolve around the principle of Modules.

Modules as Source Files
~~~~~~~~~~~~~~~~~~~~~~~

Modules can be created simply as source files. The only difference is that by
default, only algorithms are available from imported modules. To make regular
variables available, the ``global`` keyword must be used. An example source
module is shown below. Note that the values are retrieved from the module using
the member operator, ``[module].[member]``.

.. code-block::
        :caption: module.zhp

        // Defining a global variable
        global x = 100

        // Defining a file-local variable
        y = 10

        // Simple function
        alg foo()
                println("Printing from foo!");
        
        // Another function
        alg bar() {
                println("Printing a file-local variable...")

                // Should result in an error when called from
                // the module as y is a file-local variable
                println("\ty = ", y)
        }

Then, importing and using the module can be done as follows.

.. code-block::
        :caption: main.zhp

        // Import module.zhp
        import module

        // Print the variables from the module:
        // there should be an error with the second
        // println because y is file-local in module.zhp
        println("x = ", module.x)
        println("y = ", module.y)

        // Call both algorithms from the module:
        // there should be an error with the second call,
        // as per its comment
        module.foo()
        module.bar()

Modules as Shared Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aside from source files, shared libraries compiled with a specific set of
functions can also be used as modules. See more :ref:`here <libs_page>`.

Include Directories
~~~~~~~~~~~~~~~~~~~

By default, any modules imported with the ``import`` keyword must be located in
either the current directory or ``/usr/local/include/zhp``. To add a search
path, use the ``include`` keyword. For example, ``include bin/libs/`` adds
``bin/libs`` (from the current directory) to the include paths for the execution
of the program.
