===============================
Vertizee Graph Library
===============================

Vertizee is an object-oriented, typed, graph library for the analysis and study of graphs.

Features
=======================

* Object-oriented API: vertices and edges are first-class objects
* Graph theory greatest-hits including:

  * :doc:`Breadth-first-search (BFS) <api/algorithms/search_breadth_first>` and
    :doc:`depth-first search (DFS) <api/algorithms/search_depth_first>`
  * :doc:`Karger and Karger-Stein minimum cuts <api/algorithms/cuts>`
  * All-pairs-shortest paths: the :func:`Floyd-Warshall algorithm
    <vertizee.algorithms.paths.all_pairs.floyd_warshall>` and
    :func:`Johnson's algorithm <vertizee.algorithms.paths.all_pairs.johnson>`
  * Shortest-paths: the :func:`Bellman-Ford algorithm
    <vertizee.algorithms.paths.single_source.bellman_ford>` and :func:`Dijkstra's
    algorithm <vertizee.algorithms.paths.single_source.dijkstra>`
  * Spanning trees: :func:`Kruskal's algorithm
    <vertizee.algorithms.spanning.undirected.kruskal_spanning_tree>` and :func:`Prim's algorithm
    <vertizee.algorithms.spanning.undirected.prim_spanning_tree>`
  * Strongly-connected components: :func:`Kosaraju's algorithm
    <vertizee.algorithms.connectivity.components.strongly_connected_components>`

* Input/output routines for reading and writing unweighted and weighted graphs
* Graph theory :doc:`glossary <glossary>`, include special terminology specific to Vertizee
* Data structures:

  * :doc:`api/classes/data_structures/fibonacci_heap`
  * :doc:`api/classes/data_structures/priority_queue`
  * :doc:`api/classes/data_structures/tree`
  * :doc:`api/classes/data_structures/union_find`


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

Vertizee was originally developed by Christopher Peisert during the pandemic of 2020 and was
inspired by assignments for Professor Tim Roughgarden's Stanford Algorithms courses
(`available on Coursera <https://www.coursera.org/specializations/algorithms>`_). In addition to
Roughgarden's excellent series *Algorithms Illuminated*, many of the Vertizee algorithm
implementations are based on the modern-classic *Introduction to Algorithms: Third Edition* by
Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. The Vertizee package
structure and various aspects of the API were influenced by the excellent `NetworkX
<https://networkx.github.io/>`_ library.


=======================
Documentation
=======================

.. toctree::
   :maxdepth: 5
   :caption: General Information

   installation
   glossary
   bibliography
   license
   release_log

.. toctree::
   :maxdepth: 5
   :caption: Tutorials

   tutorials/getting_started
   tutorials/connected_components
   tutorials/cuts
   tutorials/paths
   tutorials/search
   tutorials/spanning_tree_arborescence

.. toctree::
   :maxdepth: 5
   :caption: API Reference

   api/algorithms/index
   api/classes/index
   api/exceptions
   api/io/index

.. toctree::
   :maxdepth: 5
   :caption: Developer Documentation

   developer/contributor_intro
   developer/contributor_guide
   developer/core_dev


Indices
==================

* :ref:`genindex`
