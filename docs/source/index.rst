===============================
Vertizee Graph Library
===============================

Vertizee is an object-oriented, typed, graph library for the analysis and study of graphs.

Features
=======================

* Object-oriented API: vertices and edges are first-class objects
* Graph theory greatest-hits including:

  * :doc:`Breadth-first-search (BFS) and depth-first search (DFS) <api/algorithms/search>`
  * :doc:`Karger and Karger-Stein minimum cuts <api/algorithms/cuts>`
  * All-pairs-shortest paths: the :func:`Floyd-Warshall algorithm
    <vertizee.algorithms.shortest_paths.weighted.all_pairs_shortest_paths_floyd_warshall>` and
    :func:`Johnson's algorithm
    <vertizee.algorithms.shortest_paths.weighted.all_pairs_shortest_paths_johnson>`
  * Shortest-paths: the :func:`Bellman-Ford algorithm
    <vertizee.algorithms.shortest_paths.weighted.shortest_paths_bellman_ford>` and :func:`Dijkstra's
    algorithm <vertizee.algorithms.shortest_paths.weighted.shortest_paths_dijkstra>`
  * Spanning trees: :func:`Kruskal's algorithm
    <vertizee.algorithms.tree.spanning.spanning_tree_kruskal>` and :func:`Prim's algorithm
    <vertizee.algorithms.tree.spanning.spanning_tree_prim>`
  * Strongly-connected components: :func:`Kosaraju's algorithm
    <vertizee.algorithms.components.strongly_connected.kosaraju_strongly_connected_components>`

* Input/output routines for reading and writing unweighted and weighted graphs
* Data structures:

  * :doc:`api/classes/bk_tree`
  * :doc:`api/classes/fibonacci_heap`
  * :doc:`api/classes/priority_queue`
  * :doc:`api/classes/union_find`


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
structure and inspired many ideas for improved documentation and algorithms.


=======================
Documentation
=======================

.. toctree::
   :maxdepth: 1
   :caption: General Information

   installation
   license
   release_log

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/getting_started
   tutorials/search
   tutorials/connected_components
   tutorials/shortest_paths

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/algorithms/index
   api/classes/index
   api/exceptions
   api/io/index

.. toctree::
   :maxdepth: 1
   :caption: Developer Documentation

   developer/contributor_intro
   developer/contributor_guide
   developer/core_dev


Indices
==================

* :ref:`genindex`
