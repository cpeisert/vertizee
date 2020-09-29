# Vertizee

[![PyPI version](https://badge.fury.io/py/vertizee.svg)](https://pypi.python.org/pypi/vertizee/)
[![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![Build Status](https://img.shields.io/travis/dmlc/vertizee.svg?label=build&logo=travis&branch=master)](https://travis-ci.org/dmlc/vertizee)
[![Code Coverage](https://codecov.io/github/vertizee/vertizee/badge.svg?branch=master&service=github)](https://codecov.io/github/vertizee/vertizee?branch=master)
[![GitHub License](https://img.shields.io/badge/license-Apache%202-blue.svg?style=flat)](./LICENSE)

Vertizee is an object-oriented, typed, graph library for the analysis and study of graphs.

## Basic Example

```python
>>> import vertizee as vz
>>> g = vz.DiGraph()
>>> g.add_edge('s', 't', weight=10)
>>> g.add_edge('s', 'y', weight=5)
>>> g.add_edge('y', 't', weight=3)
>>> g.vertex_count
3
>>> g['s'].adjacent_vertices
{y, t}
>>> g['s'].degree
2
>>> g['s'].edges_incoming
set()
>>> g['s'].edges_outgoing
{(s, t), (s, y)}
>>> g['s', 'y'].weight
5.0
>>> from vertizee.algorithms import shortest_paths_dijkstra
>>> # `s_paths` is a dictionary mapping vertices to ShortestPath objects. In
>>> # this case, the ShortestPath objects contain information about shortest
>>> # paths from source vertex `s` to all other vertices in the graph.
>>> s_paths: vz.VertexDict[vz.ShortestPath] = shortest_paths_dijkstra(g, source='s', save_paths=True)
>>> s_paths['t'].path
[s, y, t]
>>> s_paths['t'].length
8.0
>>> s_paths['y'].path
[s, y]
>>> s_paths['y'].length
5.0
```


## Features

* Object-oriented API: vertices and edges are first-class objects
* Graph theory greatest-hits including:
  * Breadth-first-search (BFS) and depth-first search (DFS)
  * Cuts: Karger and Karger-Stein algorithms
  * Shortest-paths: the Bellman-Ford algorithm and Dijkstra's algorithm
  * All-pairs-shortest paths: the Floyd-Warshall algorithm and Johnson's algorithm
  * Spanning trees: Kruskal's and Prim's algorithms
  * Strongly-connected components: Kosaraju's algorithm
* Input/output routines for reading and writing unweighted and weighted graphs
* Data structures:
  * BK Tree
  * Fibonacci Heap
  * Priority Queue
  * Union Find (a.k.a. Disjoint Set)


## Installation

For detailed instructions, see the [Installation documentation]().


### With pip

```bash
pip install vertizee
```

### With conda

```bash
conda install --channel conda-forge vertizee
```


## Documentation

Vertizee documentation: https://cpeisert.github.io/vertizee


# TODO(cpeisert): Update tutorial hyperlinks to online documentation.

## Tutorials

| Notebook     |      Description      |   |
|:----------|:-------------|------:|
| [Getting Started](https://github.com/cpeisert/vertizee/blob/master/docs/source/tutorials/getting_started.ipynb)  | How to create and work with graphs  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cpeisert/vertizee/blob/master/docs/source/tutorials/getting_started.ipynb) |
| [Breadth-First and Depth-First Search](https://github.com/cpeisert/vertizee/blob/master/docs/source/tutorials/search.ipynb)  | BFS and DFS graph search and traversal  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cpeisert/vertizee/blob/master/docs/source/tutorials/search.ipynb) |
| [Shortest paths](https://github.com/cpeisert/vertizee/blob/master/docs/source/tutorials/shortest_paths.ipynb)  | Finding shortest paths and all-pairs shortest paths  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cpeisert/vertizee/blob/master/docs/source/tutorials/shortest_paths.ipynb) |
| [Connected Components](https://github.com/cpeisert/vertizee/blob/master/docs/source/tutorials/connected_components.ipynb)  | Finding strongly-connected components  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cpeisert/vertizee/blob/master/docs/source/tutorials/connected_components.ipynb) |


## Want to help?

Want to [file a bug](https://github.com/cpeisert/vertizee/issues), contribute some code, or improve documentation? Excellent!
Read up on our guidelines for [contributing](https://github.com/cpeisert/vertizee/blob/master/CONTRIBUTING.rst).
