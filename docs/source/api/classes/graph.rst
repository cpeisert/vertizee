============================================================================
Graph Classes
============================================================================

All Vertizee graph classes inherit from :class:`GraphBase <vertizee.classes.graph_base.GraphBase>`.
Only *subclasses* of :class:`GraphBase <vertizee.classes.graph_base.GraphBase>` should be used in
code.

The most flexible classes are prefixed ``Multi``, since they allow multiple parallel edges
between vertices. The ``DiGraph`` classes are for graphs with directed edges.

Which graph class should I use?
===============================

+-----------------------------------------+------------+--------------------+------------------------+
| Vertizee Class                          | Type       | Parallel edges allowed | Self loops allowed |
+=========================================+============+========================+====================+
| :class:`Graph                           | undirected | No                     | Yes                |
| <vertizee.classes.graph.Graph>`         |            |                        |                    |
+-----------------------------------------+------------+------------------------+--------------------+
| :class:`MultiGraph                      | undirected | Yes                    | Yes                |
| <vertizee.classes.graph.MultiGraph>`    |            |                        |                    |
+-----------------------------------------+------------+------------------------+--------------------+
| :class:`DiGraph                         | directed   | No                     | Yes                |
| <vertizee.classes.digraph.DiGraph>`     |            |                        |                    |
+-----------------------------------------+------------+------------------------+--------------------+
| :class:`MultiDiGraph                    | directed   | Yes                    | Yes                |
| <vertizee.classes.digraph.MultiDiGraph>`|            |                        |                    |
+-----------------------------------------+------------+------------------------+--------------------+


Recommended Tutorial
====================

:doc:`Getting Started <../../tutorials/getting_started>`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/cpeisert/vertizee/blob/master/docs/source/tutorials/getting_started.ipynb


Graph Module
===============================

.. automodule:: vertizee.classes.graph


DiGraph Module
===============================

.. automodule:: vertizee.classes.digraph


Graph Base Module
===============================

.. automodule:: vertizee.classes.graph_base
