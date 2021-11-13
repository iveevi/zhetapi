.. _libs_page:

Compiling Libraries as Modules
==============================

.. toctree::
        :maxdepth: 2
        :hidden:

        ..

Of the two methods to create modules, through source files and shared libraries,
the later is the more powerful and generic approach. One can create ZHP bindings
for functions in C++ that are much faster than they would be if natively
implemented in ZHP.

The Process of Loading Shared Libraries as Modules
--------------------------------------------------

When the interpreter executes an ``import`` statement, it searches through all
the include directories (defaulted to ``.`` and ``/usr/local/include/zhp``) and
tries to find a unique library (or source file).

The shared library is then loaded into execution, and then the interpreter
searches for the specific function ``zhetapi_export_symbols``. This function
loads the requested symbols into a Module object, and this object is then loaded
into the execution of the main file.

Impementing the Exporter function
---------------------------------

Each library must have exactly one function, ``zhetapi_export_symbols``, that
acts as an "exporter" for the library. This function is used to load necessary
symbols into a module and additionally to do any preprocessing or setup. The
exporter has a signature of ``void (Module *)`` and it must preserve its name
during linkage: the ``extern "C"`` keyword must be used if compiling as a C++
library.

To load regular variables into Modules, use its ``add`` method. This works for
loading C++ functions, but instead of simply passing the name of the function,
the function must be wrapped in a Registrable object. An example single-file
library is shown below:

.. code-block:: cpp

        // Targs is an alias for std::vector <Token *>
        Token *reg1(const Targs &targs)
        {
                // some code...
        }

        Token *my_registrable2(const Targs &targs)
        {
                // some more code...
        }

        // This is the exporter function (which should only
        // appear once in the complation unit)
        extern "C" void zhetapi_export_symbols(Module *module)
        {
                // Add the above functions as registrables
                module->add("reg1", new Registrable("reg1", reg1));
                module->add("reg2", new Registrable("reg2", my_registrable2));

                // Adding operand values (R is an alias for long double)
                module->add("rounded_pi", long double, 3.14);
                module->add("real_pi", R, acos(-1));

                // and more symbols...
        }

The API provides macros to make all of this more convenient. The above source
code can be rewritten as follows:

.. code-block:: cpp

        ZHETAPI_REGISTER(reg1)
        {
                // code for reg1...
        }

        ZHETAPI_REGISTER(my_registrable2)
        {
                // code for my_registrable2...
        }

        ZHETAPI_LIBRARY()
        {
                ZHETAPI_EXPORT(reg1);
                ZHETAPI_EXPORT_SYMBOL(reg2, my_registrable2);

                ZHETAPI_EXPORT_CONSTANT(rounded_pi, long double, 3.14);
                ZHETAPI_EXPORT_CONSTANT(real_pi, Z, acos(-1));
        }

These macros are defined as follows:

.. literalinclude:: ../../engine/module.hpp
        :lines: 54-97
        :linenos:
        :caption: engine/module.hpp

A Canonical Organization for a Library
--------------------------------------

A good way to organize the source code for each library, in my experience, is to
have a global header file, a file containing just the exporter function, and
then other source files containing the implementations of the Registrables,
separated in any logical way.

[TODO add image]

To see examples of actual library implementations, see the directories in
``libs`` of the `GitHub repository <https://github.com/vedavamadathil/zhetapi>`_
(each of the subdirectories corresponds to an individual library).
