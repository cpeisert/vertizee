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

"""Data types to support directed graphs.

Note:
    All graph types allow self loops.

Directed Graph Types:
    * DiGraph - Directed graph without parallel edges.
    * MultiDiGraph - Directed graph that allows parallel edges.

Supporting Type(s) for Directed Graphs:
    * DiEdge - A directed connection between two vertices. The starting vertex is called the tail
        and the destination vertex is called the head. The edge points from the tail to the head.

Example:
    g = MultiDiGraph()
    edge01 = g.add_edge(0, 1)
    edge12 = g.add_edge(1, 0)
    edge20 = g.add_edge(2, 0)
    assert g[0].degree == 3
"""
# Note: In Python < 3.10, in order to prevent Sphinx from unfolding type aliases, future
# annotations must be imported and type aliases that should not be unfolded must be quoted.
from __future__ import annotations

from typing import List, Optional, Set

from vertizee.classes.edge import DEFAULT_WEIGHT, DiEdge
from vertizee.classes.graph_base import GraphBase
from vertizee.classes.graph_primitives import GraphPrimitive
from vertizee.classes.graph_primitives import ParsedPrimitives
from vertizee.classes.vertex import VertexKeyType

# pylint: disable=useless-super-delegation


class DiGraph(GraphBase):
    """A digraph is a directed graph without parallel edges. Self loops are allowed."""

    def __init__(self, *args: "GraphPrimitive"):
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
        tail: VertexKeyType,
        head: Optional[VertexKeyType] = None,
        weight: Optional[float] = DEFAULT_WEIGHT,
        parallel_edge_count: Optional[int] = 0,
        parallel_edge_weights: Optional[List[float]] = None,
    ) -> DiEdge:
        return super().add_edge(tail, head, weight, parallel_edge_count, parallel_edge_weights)

    def deepcopy(self) -> "DiGraph":
        graph_copy = DiGraph()
        super()._deepcopy_into(graph_copy)
        return graph_copy

    @property
    def edges(self) -> Set[DiEdge]:
        return super().edges

    def get_all_graph_edges_from_parsed_primitives(
        self, parsed_primitives: ParsedPrimitives
    ) -> List[DiEdge]:
        return super().get_all_graph_edges_from_parsed_primitives(parsed_primitives)

    def get_edge(self, *args: GraphPrimitive) -> Optional[DiEdge]:
        return super().get_edge(*args)

    def get_random_edge(self) -> Optional[DiEdge]:
        return super().get_random_edge()

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

    def __init__(self, *args: "GraphPrimitive"):
        super().__init__(
            GraphBase._create_key, is_directed_graph=True, is_multigraph=True, is_simple_graph=False
        )
        super().add_edges_from(*args)

    # pylint: disable=arguments-differ
    def add_edge(
        self,
        tail: VertexKeyType,
        head: Optional[VertexKeyType] = None,
        weight: Optional[float] = DEFAULT_WEIGHT,
        parallel_edge_count: Optional[int] = 0,
        parallel_edge_weights: Optional[List[float]] = None,
    ) -> DiEdge:
        return super().add_edge(tail, head, weight, parallel_edge_count, parallel_edge_weights)

    def deepcopy(self) -> "MultiDiGraph":
        graph_copy = MultiDiGraph()
        super()._deepcopy_into(graph_copy)
        return graph_copy

    @property
    def edges(self) -> Set[DiEdge]:
        return super().edges

    def edge_count_ignoring_parallel_edges(self) -> int:
        edge_count = len(self._edges)
        if self._is_simple_graph:
            return edge_count

        edges_inverted_tails_and_heads_count = 0
        for edge in self.edges:
            parallel = self.get_edge(edge.head, edge.tail)
            if parallel is not None and not parallel.is_loop():
                edges_inverted_tails_and_heads_count += 1
        edge_count -= int(edges_inverted_tails_and_heads_count / 2)
        return edge_count

    def get_all_graph_edges_from_parsed_primitives(
        self, parsed_primitives: ParsedPrimitives
    ) -> List[DiEdge]:
        return super().get_all_graph_edges_from_parsed_primitives(parsed_primitives)

    def get_edge(self, *args: GraphPrimitive) -> Optional[DiEdge]:
        return super().get_edge(*args)

    def get_random_edge(self) -> Optional[DiEdge]:
        return super().get_random_edge()

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
