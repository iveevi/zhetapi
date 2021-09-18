Variables
=========

Definition
----------

All programming languages must have a way for the user to store and retrieve
values. The objects that do this are called *variables*.

These are not the same as variables in mathematics, however. Consider the
statement :math:`x = y^2 + 5`. In mathematics, this is an equality: it states a
relationship between :math:`x` and :math:`y`. In programming, however, this is
an *assignment* statement: assuming that the value of :math:`y` had been
previously been defined, it sets the value of (refered to by) :math:`x` to the
value of the expression :math:`y^2 + 5`. For example, if :math:`y = 5`, then
:math:`x = 30`.

Using Variables in ZHP
----------------------

In ZHP, variables are created when they are assigned (and initialized).
Assignment statements are of the form :code:`[identifier] = [value]` (identifier
is a string of alphabets, digits and underscores which does not start with a
digit). Examples are below:

.. code_block::

       x = 10
       y = 10.10
       z = 4/5

       I = [[1, 0], [0, 1]]
       X = [4, 5]

       Y = IX // or Y = I * X

Dynamically vs Statically Typed Languages
-----------------------------------------

ZHP is a *dynamically typed* language. This means that the type of object
refered by variables is not set until the script is running. In contrast, in
*statically typed* languages like Java, C, C++, etc., variables must adhere to a
specific type.

As a mathematical analogy, think of the type of a variable to be
the mathematical set to which it belongs. For example, when we write :code:`x =
5`, we expect that the type of :code:`x` is that of an integer: :code:`x`
belongs to the set of integers. Hence, with statically typed languages, the
domain of a variable is static, whereas in dynamically typed languages, the
domain of a variable is dynamic.
