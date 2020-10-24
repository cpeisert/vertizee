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

"""Data types for directed graphs.

Directed graph types:
    * :class:`DiGraph` - Directed graph without parallel edges.
    * :class:`MultiDiGraph` - Directed graph that allows parallel edges.

See Also:
    * :class:`DiEdge <vertizee.classes.edge.DiEdge>`
    * :class:`GraphBase <vertizee.classes.graph_base.GraphBase>`
    * :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
    * :class:`Vertex <vertizee.classes.vertex.Vertex>`

Example:
    >>> import vertizee as vz
    >>> g = vz.MultiDiGraph()
    >>> edge01 = g.add_edge(0, 1)
    >>> g.add_edge(0, 1)
    >>> g.add_edge(1, 0)
    >>> g.add_edge(2, 0)
    >>> g[0].degree
    4
    >>> edge01.parallel_edge_count
    1
    >>> g[1, 0].parallel_edge_count
    0
"""
# Note: In Python < 3.10, in order to prevent Sphinx from unfolding type aliases, future
# annotations must be imported and type aliases that should not be unfolded must be quoted.
from __future__ import annotations

from typing import TYPE_CHECKING

from vertizee.classes import edge
from vertizee.classes.graph_base import GraphBase

if TYPE_CHECKING:
    from vertizee.classes.edge import DiEdge, Edge
    from vertizee.classes.primitives_parsing import GraphPrimitive
    from vertizee.classes.vertex import VertexType


# pylint: disable=useless-super-delegation


class DiGraph(GraphBase):
    """A digraph is a directed graph without parallel edges. Self loops are allowed."""

    def __init__(self, *args: "GraphPrimitive") -> None:
        super().__init__(
            GraphBase._create_key,
            is_directed_graph=True,
            is_multigraph=False,
            is_simple_graph=False,
        )
        super().add_edges_from(*args)

    # pylint: disable=arguments-differ
    def add_edge(
        self,
        tail: "VertexType",
        head: "VertexType",
        weight: float = edge.DEFAULT_WEIGHT,
        **attr
    ) -> "Edge":
        """Adds a new directed edge to the graph.

        If there is already an edge with matching vertices, then the internal :class:`DiEdge
        <vertizee.classes.edge.DiEdge>` object is modified by incrementing the parallel edge count.

        Args:
            tail: The starting vertex. This is a synonym for ``vertex1``.
            head: The destination vertex to which the ``tail`` points. This is a synonym for
                ``vertex2``.
            weight: Optional; The edge weight. Defaults to 1.
            **attr: Optional; Keyword arguments to be added to the ``attr`` dictionary.

        Returns:
            DiEdge: The newly added directed edge (or pre-existing edge if a parallel edge was
            added).
        """
        return super().add_edge(tail, head, weight, **attr)

    def get_reverse_graph(self) -> "DiGraph":
        """Creates a new graph that is the reverse of this graph (i.e. all directed edges
        pointing in the opposite direction).

        The reverse of a directed graph is also called the transpose or the converse. See
        https://en.wikipedia.org/wiki/Transpose_graph.

        Returns:
            DiGraph: The reverse of this graph.
        """
        reverse_graph = DiGraph()
        self._reverse_graph_into(reverse_graph)
        return reverse_graph


class MultiDiGraph(GraphBase):
    """A multidigraph is a directed graph that allows parallel edges and self loops."""

    def __init__(self, *args: "GraphPrimitive") -> None:
        super().__init__(
            GraphBase._create_key, is_directed_graph=True, is_multigraph=True, is_simple_graph=False
        )
        super().add_edges_from(*args)

    # pylint: disable=arguments-differ
    def add_edge(
        self,
        tail: "VertexType",
        head: "VertexType",
        weight: float = edge.DEFAULT_WEIGHT,
        **attr
    ) -> "Edge":
        """Adds a new directed edge to the graph.

        If there is already an edge with matching vertices, then the internal :class:`DiEdge
        <vertizee.classes.edge.DiEdge>` object is modified by incrementing the parallel edge count.

        Args:
            tail: The starting vertex. This is a synonym for ``vertex1``.
            head: The destination vertex to which the ``tail`` points. This is a synonym for
                ``vertex2``.
            weight: Optional; The edge weight. Defaults to 1.
            **attr: Optional; Keyword arguments to be added to the ``attr`` dictionary.

        Returns:
            DiEdge: The newly added directed edge (or pre-existing edge if a parallel edge was
            added).
        """
        return super().add_edge(tail, head, weight, **attr)

    @property
    def edge_count_ignoring_parallel_edges(self) -> int:
        """The number of edges excluding parallel edges."""
        edge_count = len(self._edges)
        if self._is_simple_graph:
            return edge_count

        edges_inverted_tails_and_heads_count = 0
        for edge in self.edges:
            parallel = self._get_edge(edge.vertex2, edge.vertex1)
            if parallel is not None and not parallel.is_loop():
                edges_inverted_tails_and_heads_count += 1
        edge_count -= int(edges_inverted_tails_and_heads_count / 2)
        return edge_count

    def get_reverse_graph(self) -> "MultiDiGraph":
        """Creates a new graph that is the reverse of this graph (i.e. all directed edges
        pointing in the opposite direction).

        The reverse of a directed graph is also called the transpose or the converse. See
        https://en.wikipedia.org/wiki/Transpose_graph.

        Returns:
            MultiDiGraph: The reverse of this graph.
        """
        reverse_graph = MultiDiGraph()
        self._reverse_graph_into(reverse_graph)
        return reverse_graph
