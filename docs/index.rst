Zhetapi Docs
============

Zhetapi is a *C++14* machine learning and numerical analysis API that was
built in the hopes of making mathematical computation and algorithmic research
more convenient to users. In the name of convenience and beginner-friendliness,
the API comes with a scripting language, ZHP.

The project is composed to two main parts:

Scripting Language
------------------

While C++ overloading makes the API convenient for C++ users, the :ref:`ZHP
scripting language <zhp_page>` language does better through notational
convenience. The idea is that the code should read with elegance alike to a
mathematical proof; hence why a lot of "mathy" terminolgy is used (ie. functions
vs. algorithms).

.. toctree::
        :caption: ZHP Scripting Language
        :maxdepth: 2
        :hidden:

        zhp/zhp.rst
        zhp/libs.rst

Application Interface
---------------------

On one hand, Zhetapi is a library of many C++ functions for applications in
numerical analysis and machine learning. See the full index of classes and
namespaces :ref:`here <api_page>` or in the sidebar.

.. toctree::
        :caption: Lessons
        :maxdepth: 2
        :hidden:

        lessons/architecture/index.rst
        lessons/dnn/index.rst
        lessons/intro/index.rst

.. toctree::
        :caption: API
        :maxdepth: 2
        :hidden:

        api/activation
        api/cast
        api/collection
        api/generator
        api/gnn
        api/gradient
        api/image
        api/indexable
        api/interval
        api/iterator
        api/layer
        api/linalg
        api/matrix
        api/module
        api/nvarena
        api/parametrization
        api/plot
        api/polynomial
        api/tensor
        api/token
        api/vector

Lessons
-------

This site also provides :ref:`lessons <lessons_page>` on various things related
to computer science.  Check them out (using the link or the sidebar) to learn
more about computer science and gain practice with programming.
