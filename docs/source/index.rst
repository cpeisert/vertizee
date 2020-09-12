.. _contents:

===============================
Vertizee Graph Library
===============================

Vertizee is an object-oriented, typed, graph library for the analysis and study of graphs.

Features
=======================

* Object-oriented API: vertices and edges are first-class objects
* Graph theory greatest-hits including:

  * Breadth-first-search (BFS) and depth-first search (DFS)
  * Cuts: Karger minimum cut
  * All-pairs-shortest paths: the Floyd-Warshall algorithm and Johnson's algorithm
  * Shortest-paths: the Bellman-Ford algorithm and Dijkstra's algorithm
  * Spanning trees: Kruskal's and Prim's algorithms
  * Strongly-connected components: Kosaraju's algorithm

* Input/output routines for reading and writing unweighted and weighted graphs
* Data structures:

  * `BK Tree <https://github.com/cpeisert/vertizee/blob/master/vertizee/classes/collections/bk_tree.py>`_
  * `Fibonacci Heap <https://github.com/cpeisert/vertizee/blob/master/vertizee/classes/collections/fibonacci_heap.py>`_
  * `Priority Queue <https://github.com/cpeisert/vertizee/blob/master/vertizee/classes/collections/priority_queue.py>`_
  * `Union Find (a.k.a. Disjoint Set) <https://github.com/cpeisert/vertizee/blob/master/vertizee/classes/collections/union_find.py>`_


Python
=======================

Python is a powerful programming language that enables graphs, and graph algorithms, to be
expressed in a natural, concise format reminiscent of pseudocode. Vertizee takes advantage of
`type hints <https://docs.python.org/3/library/typing.html>`_, and requires Python 3.6+.


Free software
=======================

Vertizee is free software; you can redistribute it and/or modify it under the
terms of the :doc:`Apache 2.0 License </license>`.  We welcome contributions.
Join us on `GitHub <https://github.com/cpeisert/vertizee>`_.


History
=======================

Vertizee was originally developed by Christopher Peisert during the pandemic of 2020 and grew out of
programming assignments for Professor Tim Roughgarden's Stanford Algorithms courses
(`available on Coursera <https://www.coursera.org/specializations/algorithms>`_). In addition to
Roughgarden's excellent series *Algorithms Illuminated*, many of the Vertizee implementations
are based on the modern-classic
*Introduction to Algorithms: Third Edition* by Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
The `NetworkX <https://networkx.github.io/>`_ library was used as a template for the package
structure and inspired many ideas for improved algorithms and documentation.


Documentation
=======================

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/getting_started
   tutorials/bfs_dfs
   tutorials/connected_components
   tutorials/shortest_paths

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/classes/index
   api/algorithms/index
   api/io/index
   api/exceptions

.. toctree::
   :maxdepth: 1
   :caption: Developer Documentation

   developer/contributor_intro
   developer/contributor_guide
   developer/core_dev

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   license
   release_log


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
