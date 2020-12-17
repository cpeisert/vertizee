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

"""Utility classes supporting algorithms for findining :term:`spanning trees <spanning tree>`,
:term:`spanning arborescenses <spanning arborescence>`, :term:`spanning forests <spanning forest>`,
and spanning :term:`branchings <branching>`.

* :class:`Cycle` - A :term:`cycle` in a graph.
* :class:`PseudoEdge` - A :term:`directed edge` that has :class:`PseudoVertex` endpoints.
* :class:`PseudoGraph` - A :term:`digraph` that is comprised of :class:`pseudovertices
  <PseudoVertex>` and :class:`pseudoedges <PseudoEdge>`.
* :class:`PseudoVertex` - A :term:`vertex` that may either represent a regular vertex or be a new
  vertex formed by contracting a :term:`cycle`.
* :class:`UndirectedPseudoVertex` - A :term:`vertex` in an term:`undirected graph` that supports
  additional properties for spanning algorithms.
"""

from __future__ import annotations
from typing import Dict, Final, Generic, List, Optional, Set

from vertizee.classes.graph import DiGraph
from vertizee.classes.edge import _DiEdge, E
from vertizee.classes.vertex import _DiVertex, V


class Cycle(Generic[V, E]):
    """A :term:`cycle` in a graph.

    Args:
        initial_vertex: The first vertex added to the pseudovertex.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        """The label of the cycle matches the label of the vertex that is the contraction of the
        cycle."""

        self.edges: Set[E] = set()
        self.vertices: Set[V] = set()

        self.incoming_edges: Set[E] = set()
        """The set of edges whose tail is outside the cycle and whose head is part of the cycle."""

        self.outgoing_edges: Set[E] = set()
        """The set of edges whose head is outside the cycle and whose tail is part of the cycle."""

        self.outgoing_edge_head_previous_parent: Dict[E, V] = dict()
        """A dictionary mapping the head vertices of outgoing edges to their previous parent
        vertices prior to this cycle being contracted."""

        self.min_weight_edge: Optional[E] = None
        """The edge of the cycle that has the min weight."""

    def __hash__(self) -> int:
        return hash(self.label)

    def add_cycle_edge(self, edge: E) -> None:
        """Adds an edge (and associated vertices) to the cycle."""
        if not self.min_weight_edge:
            self.min_weight_edge = edge

        self.edges.add(edge)
        self.vertices.add(edge.vertex1)
        self.vertices.add(edge.vertex2)


class PseudoVertex(_DiVertex):
    """Pseudovertices are used to support algorithms that :term:`contract` vertices and edges from
    some graph :math:`G`, where the vertices and edges to be contracted comprise a :term:`cycle` in
    :math:`G`. The contracted vertices and edges form a new vertex (the pseudovertex)
    :math:`v_1^{i + 1} in a :term:`subgraph` :math:`H`. Any pseudovertex that is not the
    contraction of a cycle in :math:`G` is the same as the vertex in :math:`G` with a matching
    label.

    If a pseudovertex represents a contracted cycle from :math:`G`, then ``cycle`` will
    refer to the cycle that was contracted to form the new vertex.

    Pseudovertices may also store a reference to an :term:`arborescence` in :math:`G` containing
    the vertex.

    Args:
        label: The label for this vertex. Must be unique to the graph.
        parent_graph: The parent graph to which this vertex belongs.
        incident_edges: Optional; The edges that are incident on this vertex.
    """

    __slots__ = ("cycle", "parent_vertex")

    def __init__(
        self, label: str, parent_graph: PseudoGraph, incident_edge_labels: Set[str] = None
    ) -> None:
        super().__init__(label, parent_graph)
        self._incident_edges._incident_edge_labels = incident_edge_labels

        self.cycle: Optional[Cycle] = None
        self.parent_vertex: Optional[PseudoVertex] = None
        """The parent vertex in the arborescence. If this vertex (self) is the root of an
        arborescence, then parent will be None."""

    def contains_cycle(self) -> bool:
        """Returns True if the pseudovertex represents the contraction of a cycle."""
        return self.cycle is not None

    def get_an_original_vertex(self) -> PseudoVertex:
        """If this vertex does not contain a cycle, then it is an original vertex. If this vertex
        does contain a cycle, then an arbitrary vertex is chosen from the cycle, and its
        ``get_an_original_vertex`` method is called.
        """
        if not self.contains_cycle():
            return self
        cycle_vertex = next(iter(self.cycle.vertices))
        return cycle_vertex.get_an_original_vertex()


class PseudoEdge(_DiEdge):
    """PseudoEdge is a directed edge that supports algorithms that may contract vertices into a new
    vertex. If one of the pseudovertex endpoints represents a contracted cycle, then a reference to
    the previous version of the edge prior to the vertex contraction is stored in
    ``previous_version.``

    Args:
        initial_vertex: The first vertex added to the pseudovertex.
    """

    __slots__ = ("previous_version",)

    def __init__(self, vertex1: PseudoVertex, vertex2: PseudoVertex, weight: float) -> None:
        super().__init__(vertex1, vertex2, weight)

        self.previous_version: Optional[PseudoEdge] = None
        """The previous version (if one exists), is the version of the edge before one of its
        endpoints was replaced with a vertex representing a circuit."""

    def is_original_edge(self) -> bool:
        """Returns True if the pseudoedge represents an original edge, i.e., and edge with no
        previous versions."""
        return self.previous_version is None

    def get_original_edge(self) -> PseudoEdge:
        """If this edge does not contain a previous version, then it is an original edge. If this
        edge does contain a previous version, then the previous version's ``get_original_edge``
        method is called.
        """
        if self.previous_version:
            return self.previous_version.get_original_edge()
        return self


class PseudoGraph(DiGraph):
    """PseudoGraph is a graph that supports algorithms that may :term:`contract` vertices and edges,
    where the vertices and edges to be contracted comprise a :term:`cycle`.
    """
    def __init__(self):
        super().__init__()
        self.cycle_stack: List[Cycle[PseudoVertex, PseudoEdge]] = list()
        """The cycle stack is a first-in-last-out (FILO) sequence of cycles found in the graph.
        Each cycle on the stack corresponds to a :class:`PseudoVertex` with the same label that
        was formed by contracting the vertices and edges comprising the cycle."""

        self.cycle_label_count = 0
        self._CYCLE_LABEL_PREFIX: Final = "__cycle_label_"

    def add_edge_object(self, edge: PseudoEdge):
        """Adds a PseudoEdge to the graph."""
        self._edges[edge.label] = edge
        edge.vertex1._add_edge(edge)
        edge.vertex2._add_edge(edge)

    def add_vertex_object(self, vertex: PseudoVertex):
        """Adds a PseudoVertex to the graph."""
        self._vertices[vertex.label] = vertex

    def create_cycle_label(self) -> str:
        """Creates a cycle label that is unique to this graph instance."""
        self.cycle_label_count += 1
        return f"{self._CYCLE_LABEL_PREFIX}{self.cycle_label_count}__"
