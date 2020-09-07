Graph Classes
============================================================================

All Vertizee graph classes inherit from `GraphBase`. Only *subclasses* of `GraphBase`
should be used in code.

The most flexible classes are prefixed `Multi`, since they allow multiple parallel edges
between vertices. The `DiGraph` classes are for graphs with directed edges.

Which graph class should I use?
-------------------------------

+----------------+------------+--------------------+------------------------+
| Networkx Class | Type       | Self-loops allowed | Parallel edges allowed |
+================+============+====================+========================+
| SimpleGraph    | undirected | No                 | No                     |
+----------------+------------+--------------------+------------------------+
| Graph          | undirected | Yes                | No                     |
+----------------+------------+--------------------+------------------------+
| MultiGraph     | undirected | Yes                | Yes                    |
+----------------+------------+--------------------+------------------------+
| DiGraph        | directed   | Yes                | No                     |
+----------------+------------+--------------------+------------------------+
| MultiDiGraph   | directed   | Yes                | Yes                    |
+----------------+------------+--------------------+------------------------+


.. autosummary::
     :toctree: _stubs

     vertizee.classes.graph_base
     vertizee.classes.graph
     vertizee.classes.digraph
