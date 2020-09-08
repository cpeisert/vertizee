# Vertizee

[![PyPI version](https://badge.fury.io/py/vertizee.svg)](https://pypi.python.org/pypi/vertizee/)
[![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![Build Status](https://img.shields.io/travis/dmlc/vertizee.svg?label=build&logo=travis&branch=master)](https://travis-ci.org/dmlc/vertizee)
[![Code Coverage](https://codecov.io/github/vertizee/vertizee/badge.svg?branch=master&service=github)](https://codecov.io/github/vertizee/vertizee?branch=master)
[![GitHub License](https://img.shields.io/badge/license-Apache%202-blue.svg?style=flat)](./LICENSE)

Vertizee is an object-oriented, typed, graph library for the analysis and study of graphs.

## Basic Example

```
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
{<vertizee.classes.edge.DiEdge object at 0x7f9f35c4b070>, <vertizee.classes.edge.DiEdge object at 0x7f9f35bbc9d0>}
>>> g['s']['y'].weight
5.0
>>> from vertizee.algorithms import shortest_paths_dijkstra
>>> s_paths: vz.VertexDict[vz.ShortestPath] = shortest_paths_dijkstra(g, source='s', find_path_lengths_only=False)
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
  * Cuts: Karger minimum cut
  * Shortest-paths: the Bellman-Ford algorithm and Dijkstra's algorithm
  * All-pairs-shortest paths: the Floyd-Warshall algorithm and Johnson's algorithm
  * Spanning trees: Kruskal's and Prim's algorithms
  * Strongly-connected components: Kosaraju's algorithm
* Input/output routines for reading and writing unweighted and weighted graphs
* Data structures:
  * BK Tree ([source](https://github.com/cpeisert/vertizee/blob/master/vertizee/classes/collections/bk_tree.py))
  * Fibonacci Heap ([source](https://github.com/cpeisert/vertizee/blob/master/vertizee/classes/collections/fibonacci_heap.py))
  * Priority Queue ([source](https://github.com/cpeisert/vertizee/blob/master/vertizee/classes/collections/priority_queue.py))
  * Union Find (a.k.a. Disjoint Set) ([source](https://github.com/cpeisert/vertizee/blob/master/vertizee/classes/collections/union_find.py))


## Installation

You should install Vertizee in a [virtual environment](https://docs.python.org/3/library/venv.html)
or a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).


### With pip

```bash
pip install vertizee
```

### With conda

```bash
conda install --channel conda-forge vertizee
```


## Documentation

Vertizee API documentation: https://cpeisert.github.io/vertizee


## Tutorials

- [The Essentials](https://colab.research.google.com/drive/) - Essential concepts to get started quickly
- [Breadth-first and Depth-first Search](https://colab.research.google.com/drive/) - Graph search and traversal using BFS and DFS
- [Shortest paths](https://colab.research.google.com/drive/) - Finding the shortest paths through weighted and unweighted graphs
- [All-pairs-shortest-paths](https://colab.research.google.com/drive/) - Finding all of the shortest paths between every pair of vertices
- [Strongly-connected-components](https://colab.research.google.com/drive/) - Finding the strongly-connected components of a directed graph


## Want to help?

Want to [file a bug](https://github.com/cpeisert/vertizee/issues), contribute some code, or improve documentation? Excellent! Read up on our
guidelines for [contributing](https://github.com/cpeisert/vertizee/blob/master/CONTRIBUTING.md).
