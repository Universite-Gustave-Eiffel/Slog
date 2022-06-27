Quick Start
===========

This is meant as a quickstart guide to get an insight into the slog library
Start by installing the slog library. For instructions please refer to readme.md.

First import all classes from slog:

>>> from slog import *

Defining the geometries
-----------------------

In order to define a geometry we first need to define a set of :class:`geometry_module.Node`.

>>> P0 = Node(0, 0)
>>> P1 = Node(1, 0)
>>> P2 = Node(1, 0.8)
>>> P3 = Node(0.7, 0.8)
>>> P4 = Node(0.6, 0.5)
>>> P5 = Node(0.4, 0.5)
>>> P6 = Node(0.3, 0.8)
>>> P7 = Node(0, 0.8)
>>> P8 = Node(0.5, 0.5)

Using these nodes we define two geometries.

>>> geom = Geometry([P0, P1, P2, P3, P4, P5, P6, P7])
>>> geom2 = Geometry([P0, P1, P2, P3, P4])

.. Note::
    ceci est une note

.. Warning::
    ceci est un warning

