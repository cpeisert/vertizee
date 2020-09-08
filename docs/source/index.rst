.. _contents:

Vertizee Graph Library
===============================

Vertizee is an object-oriented, typed, graph library for the analysis and study of graphs.

Vertizee provides:

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


Audience
--------

Vertizee is geared toward computer science students. Many of the basic graph
algorithms in the library are based on the treatment from *Introduction to Algorithms: Third Edition* by
Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.

`Professor Tim Roughgarden's Coursera Algorithms Specialization <https://www.coursera.org/specializations/algorithms>`_
also provided inspiration for several algorithms and examples.


Python
------

Python is a powerful programming language that enables graphs, and graph algorithms, to be
expressed in a natural, concise format reminiscent of pseudocode. Vertizee takes advantage of
`type hints <https://docs.python.org/3/library/typing.html>`_, and requires Python 3.6+.


Free software
-------------

Vertizee is free software; you can redistribute it and/or modify it under the
terms of the :doc:`Apache 2.0 License </license>`.  We welcome contributions.
Join us on `GitHub <https://github.com/cpeisert/vertizee>`_.


History
-------

Vertizee was born in September 2020. The original version was designed and written by
Christopher Peisert during the pandemic of 2020 and drew inspiration from
`NetworkX <https://networkx.github.io/>`_.


| Section | Description |
|-|-|
| [Installation](#installation) | How to install the package |
| [Quick Tour](#quick-tour) | Working with graphs |
| [Tutorial: The Essentials](essentials.ipynb) | Essential concepts to get started quickly |
| [Tutorial: BFS and DFS traversals](#quick-tour-bfs-dfs) | Traversing graphs with breadth-first-search and depth-first-search |
| [Tutorial: Shortest paths](#quick-tour-shortest-paths) | Finding the shortest paths through weighted and unweighted graphs |
| [Tutorial: All-pairs-shortest-paths](#quick-tour-all-pairs-shortest-paths) | Finding all of the shortest paths between every pair of vertices |
| [Tutorial: Strongly-connected-components](#quick-tour-strongly-connected) | Finding the strongly-connected components of a directed graph |


Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/getting_started

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

   developer/contributing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
