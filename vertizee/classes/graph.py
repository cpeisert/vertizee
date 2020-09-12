# Copyright 2020 The Vertizee Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data types to support undirected graphs.

Note:
    All graph types except SimpleGraph allow self loops.

Undirected Graph Types:
    * Graph - Undirected graph without parallel edges.
    * MultiGraph - Undirected graph that allows parallel edges.
    * SimpleGraph - Undirected graph containing no parallel edges and no self loops.

Supporting Types for Undirected Graphs:
    * Edge - A undirected connection between two vertices. The order of the vertices does not
        matter.


Example::

    >>> g = MultiGraph()
    >>> edge01 = g.add_edge(0, 1)
    >>> edge12 = g.add_edge(1, 2)
    >>> edge20 = g.add_edge(2, 0)
    >>> assert g[0].degree == 2
"""

# Note: In Python < 3.10, in order to prevent Sphinx from unfolding type aliases, future
# annotations must be imported and type aliases that should not be unfolded must be quoted.
from __future__ import annotations

from vertizee.classes.graph_base import GraphBase
from vertizee.classes.graph_primitives import GraphPrimitive


class Graph(GraphBase):
    """A graph is an undirected graph without parallel edges. Self loops are allowed."""
    def __init__(self, *args: 'GraphPrimitive'):
        super().__init__(
            GraphBase._create_key, is_directed_graph=False, is_multigraph=False,
            is_simple_graph=False)
        super().add_edges_from(*args)

    def deepcopy(self) -> 'Graph':
        graph_copy = Graph()
        super()._deepcopy_into(graph_copy)
        return graph_copy


class MultiGraph(GraphBase):
    """A multigraph is an undirected graph that allows parallel edges and self loops."""
    def __init__(self, *args: 'GraphPrimitive'):
        super().__init__(
            GraphBase._create_key, is_directed_graph=False, is_multigraph=True,
            is_simple_graph=False)
        super().add_edges_from(*args)

    def deepcopy(self) -> 'MultiGraph':
        graph_copy = MultiGraph()
        super()._deepcopy_into(graph_copy)
        return graph_copy

    def edge_count_ignoring_parallel_edges(self) -> int:
        return len(self._edges)


class SimpleGraph(GraphBase):
    """A simple graph is an undirected graph with no parallel edges and no self loops."""
    def __init__(self, *args: 'GraphPrimitive'):
        super().__init__(
            GraphBase._create_key, is_directed_graph=False, is_multigraph=False,
            is_simple_graph=True)
        super().add_edges_from(*args)

    def deepcopy(self) -> 'SimpleGraph':
        graph_copy = SimpleGraph()
        super()._deepcopy_into(graph_copy)
        return graph_copy
