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

"""Data types for undirected graphs.

Undirected graph types:
    * :class:`Graph` - Undirected graph without parallel edges.
    * :class:`MultiGraph` - Undirected graph that allows parallel edges.
    * :class:`SimpleGraph` - Undirected graph containing no parallel edges and no self loops.

See Also:
    * :class:`Edge <vertizee.classes.edge.Edge>`
    * :class:`GraphBase <vertizee.classes.graph_base.GraphBase>`
    * :mod:`GraphPrimitive <vertizee.classes.parsed_primitives>`
    * :class:`Vertex <vertizee.classes.vertex.Vertex>`

Note:
    All graph types except :class:`SimpleGraph` allow self loops.

Example:
    >>> g = MultiGraph()
    >>> edge01 = g.add_edge(0, 1)
    >>> edge12 = g.add_edge(1, 2)
    >>> edge20 = g.add_edge(2, 0)
    >>> g[0].degree
    2
"""

# Note: In Python < 3.10, in order to prevent Sphinx from unfolding type aliases, future
# annotations must be imported and type aliases that should not be unfolded must be quoted.
from __future__ import annotations
from typing import TYPE_CHECKING

from vertizee.classes.graph_base import GraphBase

if TYPE_CHECKING:
    from vertizee.classes.parsed_primitives import GraphPrimitive


class Graph(GraphBase):
    """An undirected graph without parallel edges. Self loops are allowed."""

    def __init__(self, *args: "GraphPrimitive") -> None:
        super().__init__(
            GraphBase._create_key,
            is_directed_graph=False,
            is_multigraph=False,
            is_simple_graph=False,
        )
        super().add_edges_from(*args)

    def deepcopy(self) -> "Graph":
        """Returns a deep copy of this graph."""
        graph_copy = Graph()
        super()._deepcopy_into(graph_copy)
        return graph_copy


class MultiGraph(GraphBase):
    """An undirected graph that allows parallel edges and self loops."""

    def __init__(self, *args: "GraphPrimitive") -> None:
        super().__init__(
            GraphBase._create_key,
            is_directed_graph=False,
            is_multigraph=True,
            is_simple_graph=False,
        )
        super().add_edges_from(*args)

    def deepcopy(self) -> "MultiGraph":
        """Returns a deep copy of this graph."""
        graph_copy = MultiGraph()
        super()._deepcopy_into(graph_copy)
        return graph_copy

    def edge_count_ignoring_parallel_edges(self) -> int:
        """The number of edges excluding parallel edges."""
        return len(self._edges)


class SimpleGraph(GraphBase):
    """An undirected graph with no parallel edges and no self loops."""

    def __init__(self, *args: "GraphPrimitive") -> None:
        super().__init__(
            GraphBase._create_key,
            is_directed_graph=False,
            is_multigraph=False,
            is_simple_graph=True,
        )
        super().add_edges_from(*args)

    def deepcopy(self) -> "SimpleGraph":
        """Returns a deep copy of this graph."""
        graph_copy = SimpleGraph()
        super()._deepcopy_into(graph_copy)
        return graph_copy
