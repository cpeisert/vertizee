# Vertizee
Vertizee is an object-oriented, typed, graph library for the analysis and study of graphs.

[![PyPI version](https://badge.fury.io/py/vertizee.svg)](https://pypi.python.org/pypi/vertizee/)
[![Python Versions](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![Build Status](https://img.shields.io/travis/dmlc/vertizee.svg?label=build&logo=travis&branch=master)](https://travis-ci.org/dmlc/vertizee)
[![Code Coverage](https://codecov.io/github/vertizee/vertizee/badge.svg?branch=master&service=github)](https://codecov.io/github/vertizee/vertizee?branch=master)
[![GitHub License](https://img.shields.io/badge/license-Apache%202-blue.svg?style=flat)](./LICENSE)


## Features

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
  * BK Tree ([source](https://github.com/cpeisert/vertizee/blob/master/vertizee/classes/collections/bk_tree.py))
  * Fibonacci Heap ([source](https://github.com/cpeisert/vertizee/blob/master/vertizee/classes/collections/fibonacci_heap.py))
  * Priority Queue ([source](https://github.com/cpeisert/vertizee/blob/master/vertizee/classes/collections/priority_queue.py))
  * Union Find (a.k.a. Disjoint Set) ([source](https://github.com/cpeisert/vertizee/blob/master/vertizee/classes/collections/union_find.py))


| Section | Description |
|-|-|
| [Installation](#installation) | How to install the package |
| [Quick Tour](#quick-tour) | Working with graphs |
| [Tutorial: The Essentials](essentials.ipynb) | Essential concepts to get started quickly |
| [Tutorial: BFS and DFS traversals](#quick-tour-bfs-dfs) | Traversing graphs with breadth-first-search and depth-first-search |
| [Tutorial: Shortest paths](#quick-tour-shortest-paths) | Finding the shortest paths through weighted and unweighted graphs |
| [Tutorial: All-pairs-shortest-paths](#quick-tour-all-pairs-shortest-paths) | Finding all of the shortest paths between every pair of vertices |
| [Tutorial: Strongly-connected-components](#quick-tour-strongly-connected) | Finding the strongly-connected components of a directed graph |


## Installation

This repository is tested on Python 3.6+.

You should install Vertizee in a [virtual environment](https://docs.python.org/3/library/venv.html)
or a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).


### With pip

```bash
pip install vertizee
```

### With conda

```bash
conda install vertizee
```

## Want to help?

Want to file a bug, contribute some code, or improve documentation? Excellent! Read up on our
guidelines for [contributing](https://github.com/cpeisert/vertizee/blob/master/CONTRIBUTING.md).


## Disclaimer
This library is in alpha and will be going through breaking changes. Releases will be
stable enough for research and learning, but it is recommended not to use in a production
environment yet.
