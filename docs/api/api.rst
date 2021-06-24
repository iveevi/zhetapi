The Zhetapi API
===============

Zhetapi's API can be further divided into four branches:

Core API
--------

The ZHP scripting language revolves around the core API. This branches focuses
on symbolic manipulation, and making the language easy to use for beginners.

See more about the core API from ZHP's persepective:

.. toctree::
        :maxdepth: 2

        ../zhp/zhp

Linear Algebra
---------------

Crucial to any numerical analysis (and machine learning) library is the
framework to perform basic operations with matrices and vectors. This is possible
through Zhetapi's Matric and Vector classes and ``C++`` overloading.

Machine Learning
-----------------

Machine learning can be done with the API, currently through the ``DNN`` class.
I originally implemented such facilities to the API for personal usage, but
seeing as ML is so ubiquitous and that many students are getting involved in ML,
I decided to keep it in the API.

Miscellaneous
-------------

The Zhetapi API also comes with many other facilities that I believe would make
algorithm design and testing easier.

Full List
---------

.. toctree::
        :maxdepth: 2
     
        activation
        collection
        generator
        image
        indexable
        iterator
        layer
        linalg
        matrix
        module
        nvarena
        parametrization
        plot
        polynomial
        tensor
        token
        vector