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

"""
============================
Spanning algorithm utilities
============================

Utility classes supporting algorithms for findining :term:`spanning trees <spanning tree>`,
:term:`spanning arborescenses <spanning arborescence>`, :term:`spanning forests <spanning forest>`,
and spanning :term:`branchings <branching>`.

Class summary
=============

* :class:`Cycle` - A :term:`cycle` in a graph.
* :class:`PseudoEdge` - A :term:`directed edge` that has :class:`PseudoVertex` endpoints.
* :class:`PseudoGraph` - A :term:`digraph` that is comprised of :class:`pseudovertices
  <PseudoVertex>` and :class:`pseudoedges <PseudoEdge>`.
* :class:`PseudoVertex` - A :term:`vertex` that may either represent a regular vertex or be a new
  vertex formed by contracting a :term:`cycle`.

Function summary
================

* :func:`get_weight_function` - Returns a function that accepts an edge and returns the
  corresponding edge weight.

See Also:
    * :func:`edmonds <vertizee.algorithms.spanning.directed.edmonds>`
    * :func:`kruskal_optimum_forest
      <vertizee.algorithms.spanning.undirected.kruskal_optimum_forest>`
    * :func:`kruskal_spanning_tree <vertizee.algorithms.spanning.undirected.kruskal_spanning_tree>`
    * :func:`optimum_directed_forest
      <vertizee.algorithms.spanning.directed.optimum_directed_forest>`
    * :func:`optimum_forest <vertizee.algorithms.spanning.undirected.optimum_forest>`
    * :func:`prim_spanning_tree <vertizee.algorithms.spanning.undirected.prim_spanning_tree>`
    * :func:`prim_fibonacci <vertizee.algorithms.spanning.undirected.prim_fibonacci>`
    * :func:`spanning_arborescence <vertizee.algorithms.spanning.directed.spanning_arborescence>`
    * :func:`spanning_tree <vertizee.algorithms.spanning.undirected.spanning_tree>`

Detailed documentation
======================
"""

from __future__ import annotations
import collections.abc
import copy
import random
from typing import Any, Callable, cast, Dict, Final, Iterable, List, Optional
from typing import Set, TYPE_CHECKING, Union, ValuesView

from vertizee.classes import edge as edge_module
from vertizee.classes import graph as graph_module
from vertizee.classes.collection_views import SetView
from vertizee.classes.graph import GraphBase
from vertizee.classes.edge import _DiEdge, Attributes, MultiEdge, EdgeBase, MultiEdgeBase
from vertizee.classes.vertex import _DiVertex, _IncidentEdges, VertexBase

if TYPE_CHECKING:
    from vertizee.classes.edge import E_co, EdgeType
    from vertizee.classes.vertex import V_co, VertexLabel, VertexType


def get_weight_function(
    weight: str = "Edge__weight", minimum: bool = True
) -> Callable[[EdgeBase[VertexBase]], float]:
    """Returns a function that accepts an edge and returns the corresponding edge weight.

    If there is no edge weight, then the edge weight is assumed to be one.

    Note:
        For multigraphs, the minimum (or maximum) edge weight among the parallel edge connections
        is returned.

    Args:
        weight: Optional; The key to use to retrieve the weight from the edge ``attr``
            dictionary. The default value ("Edge__weight") uses the edge property ``weight``.
        minimum: Optional; For multigraphs, if True, then the minimum weight from the parallel edge
            connections is returned, otherwise the maximum weight. Defaults to True.

    Returns:
        Callable[[MutableE_co], float]: A function that accepts an edge and returns the
        corresponding edge weight.
    """

    def default_weight_function(edge: EdgeBase[VertexBase]) -> float:
        if edge._parent_graph.is_multigraph():
            if minimum:
                return min(c.weight for c in cast(MultiEdgeBase[VertexBase], edge).connections())
            return max(c.weight for c in cast(MultiEdge, edge).connections())
        return edge.weight

    def attr_weight_function(edge: EdgeBase[VertexBase]) -> float:
        if edge._parent_graph.is_multigraph():
            assert isinstance(edge, MultiEdgeBase)
            if minimum:
                return float(min(c.attr.get(weight, 1.0) for c in edge.connections()))
            return float(max(c.attr.get(weight, 1.0) for c in edge.connections()))
        return cast(Attributes, edge).attr.get(weight, 1.0)

    if weight == "Edge__weight":
        return default_weight_function
    return attr_weight_function


class Cycle:
    """A :term:`cycle` in a graph.

    Args:
        label: The label of the cycle. To generate a new cycle label, use
            :meth:`PseudoGraph.create_cycle_label`. If there is a pseudovertex that is the
            contraction of this cycle, then ``label`` should match the label of the pseudovertex.
    """

    def __init__(self, label: str) -> None:

        self.edges: Set[PseudoEdge] = set()
        """The set of edges comprising the cycle."""

        self.incoming_edges: Set[PseudoEdge] = set()
        """The set of edges whose tail is outside the cycle and whose head is part of the cycle."""

        self.label = label
        """The label of the cycle."""

        self.outgoing_edges: Set[PseudoEdge] = set()
        """The set of edges whose head is outside the cycle and whose tail is part of the cycle."""

        self.outgoing_edge_head_previous_parent: Dict[PseudoEdge, Optional[PseudoVertex]] = dict()
        """A dictionary mapping outgoing edges to their previous parent vertices in a search
        arborescence (prior to this cycle being contracted)."""

        self.min_weight_edge: Optional[PseudoEdge] = None
        """The edge of the cycle that has the minimum weight."""

        self.vertices: Set[PseudoVertex] = set()
        """The set of vertices comprising the cycle."""

    def __hash__(self) -> int:
        return hash(self.label)

    def add_cycle_edge(self, edge: PseudoEdge) -> None:
        """Adds an edge (and associated vertices) to the cycle."""
        if not self.min_weight_edge:
            self.min_weight_edge = edge

        self.edges.add(edge)
        self.vertices.add(edge.vertex1)
        self.vertices.add(edge.vertex2)


class PseudoVertex(_DiVertex):
    """Pseudovertices are used to support algorithms that :term:`contract <contraction>` vertices
    and edges from some graph :math:`G`, where the vertices and edges to be contracted comprise a
    :term:`cycle` in :math:`G`. The contracted vertices and edges form a new vertex (the
    pseudovertex) :math:`v_1^{i + 1}` in a :term:`subgraph` :math:`H`. Any pseudovertex that is not
    the contraction of a cycle in :math:`G` is the same as the vertex in :math:`G` with a matching
    label.

    If a pseudovertex represents a contracted cycle from :math:`G`, then ``cycle`` will
    refer to the cycle that was contracted to form the new vertex.

    Args:
        label: The label for this vertex. Must be unique to the graph.
        parent_graph: The parent graph to which this vertex belongs.
        incident_edge_labels: Optional; The labels of the edges that are incident on this vertex.

    Note:
        :class:`PseudoVertex`, :class:`PseudoEdge`, and :class:`PseudoGraph` were designed to
        provide suitable data structures to implement Edmonds' algorithm as presented in his 1967
        paper "Optimum Branchings". :cite:`1967:edmonds`
    """

    __slots__ = ("cycle", "parent_vertex")

    def __init__(
        self,
        label: str,
        parent_graph: "PseudoGraph",
        incident_edge_labels: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(label, parent_graph)
        self._incident_edges._incident_edge_labels = incident_edge_labels

        self._incident_edges: _IncidentEdges[
            PseudoVertex, PseudoEdge
        ] = self._incident_edges  # type: ignore

        self.cycle: Optional[Cycle] = None
        """The cycle that was contracted to form this vertex. If this vertex is not the contraction
        of a cycle, then ``cycle`` is None."""

        self.parent_vertex: Optional[PseudoVertex] = None
        """The parent vertex in the arborescence formed by searching the graph. If this vertex
        (self) is the root of an arborescence, then parent will be None."""

    def contains_cycle(self) -> bool:
        """Returns True if the pseudovertex represents the contraction of a cycle."""
        return self.cycle is not None

    def incident_edges_incoming(self) -> SetView[PseudoEdge]:
        return SetView(self._incident_edges.incoming)

    def incident_edges_outgoing(self) -> SetView[PseudoEdge]:
        return SetView(self._incident_edges.outgoing)


class PseudoEdge(_DiEdge):
    """PseudoEdge is a directed edge that supports algorithms that may contract vertices into a new
    vertex. If one of the pseudovertex endpoints represents a contracted cycle, then a reference to
    the previous version of the edge prior to the vertex contraction is stored in
    ``previous_version.``

    Args:
        initial_vertex: The first vertex added to the pseudovertex.

    Note:
        :class:`PseudoVertex`, :class:`PseudoEdge`, and :class:`PseudoGraph` were designed to
        provide suitable data structures to implement Edmonds' algorithm as presented in his 1967
        paper "Optimum Branchings". :cite:`1967:edmonds`
    """

    __slots__ = ("previous_version",)

    def __init__(self, vertex1: "PseudoVertex", vertex2: "PseudoVertex", weight: float) -> None:
        super().__init__(vertex1, vertex2, weight)

        self.previous_version: Optional[PseudoEdge] = None
        """The previous version (if one exists), is the version of the edge before one of its
        endpoints was replaced with a vertex representing a circuit."""

    def is_original_edge(self) -> bool:
        """Returns True if the pseudoedge represents an original edge, i.e., and edge with no
        previous versions."""
        return self.previous_version is None

    def get_original_edge(self) -> "PseudoEdge":
        """If this edge does not contain a previous version, then it is an original edge. If this
        edge does contain a previous version, then the previous version's ``get_original_edge``
        method is called.
        """
        if self.previous_version:
            return self.previous_version.get_original_edge()
        return self

    @property
    def vertex1(self) -> PseudoVertex:
        """The tail vertex (type :class:`DiVertex <vertizee.classes.vertex.DiVertex>`), which is
        the origin of the :term:`directed edge`."""
        return cast(PseudoVertex, self._vertex1)

    @property
    def vertex2(self) -> PseudoVertex:
        """The head vertex (type :class:`DiVertex <vertizee.classes.vertex.DiVertex>`), which is
        the destination of the :term:`directed edge`."""
        return cast(PseudoVertex, self._vertex2)


class PseudoGraph(GraphBase[PseudoVertex, PseudoEdge]):
    """PseudoGraph is a graph that supports algorithms that may :term:`contract <contraction>`
    vertices and edges, where the vertices and edges to be contracted comprise a :term:`cycle`.

    Note:
        :class:`PseudoVertex`, :class:`PseudoEdge`, and :class:`PseudoGraph` were designed to
        provide suitable data structures to implement Edmonds' algorithm as presented in his 1967
        paper "Optimum Branchings". :cite:`1967:edmonds`
    """

    def __init__(
        self,
        edges_or_graph: Optional[Union[Iterable["EdgeType"], "GraphBase[V_co, E_co]"]] = None,
        **attr: Any,
    ) -> None:
        super().__init__(
            allow_self_loops=True,
            is_directed=True,
            is_multigraph=False,
            is_weighted_graph=False,
            **attr,
        )

        if edges_or_graph and isinstance(edges_or_graph, GraphBase):
            graph_module._init_graph_from_graph(self, edges_or_graph)

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(cast(Iterable["EdgeType"], edges_or_graph))

        elif edges_or_graph:
            raise TypeError(
                f"expected GraphBase or Iterable instance; {type(edges_or_graph).__name__} found"
            )

        self.cycle_stack: List["Cycle"] = list()
        """The cycle stack is a first-in-last-out (FILO) sequence of cycles found in the graph.
        Each cycle on the stack corresponds to a :class:`PseudoVertex` with the same label that
        was formed by contracting the vertices and edges comprising the cycle."""

        self.cycle_label_count = 0
        self._CYCLE_LABEL_PREFIX: Final[str] = "__cycle_label_"

    def add_edge(
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        **attr: Any,
    ) -> "PseudoEdge":
        """Adds a new directed edge to the graph.

        If an existing edge matches the vertices, the existing edge is returned.

        Args:
            tail: The starting vertex. This is a synonym for ``vertex1``.
            head: The destination vertex to which the ``tail`` points. This is a synonym for
                ``vertex2``.
            weight: Optional; The edge weight. Defaults to ``edge.DEFAULT_WEIGHT`` (1.0).
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

        Returns:
            DiEdge: The newly added edge, or an existing edge with matching vertices.
        """
        return graph_module._add_edge_to_graph(
            self, PseudoEdge, vertex1=vertex1, vertex2=vertex2, weight=weight, **attr
        )

    def add_edge_object(self, edge: "PseudoEdge") -> None:
        """Adds a PseudoEdge to the graph."""
        self._edges[edge.label] = edge
        edge.vertex1._add_edge(edge)
        edge.vertex2._add_edge(edge)

    def add_vertex(self, label: "VertexLabel", **attr: Any) -> "PseudoVertex":
        """Adds a vertex to the graph and returns the new Vertex object. If an existing vertex
        matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex.
            **attr: Optional; Keyword arguments to add to the vertex ``attr`` dictionary.

        Returns:
            DiVertex: The new vertex (or an existing vertex matching the vertex label).
        """
        return graph_module._add_vertex_to_graph(self, PseudoVertex, label=label, **attr)

    def add_vertex_object(self, vertex: "PseudoVertex") -> None:
        """Adds a PseudoVertex to the graph."""
        self._vertices[vertex.label] = vertex

    def clear(self) -> None:
        """Removes all edges and vertices from the graph."""
        self._edges.clear()
        self._vertices.clear()

    def create_cycle_label(self) -> str:
        """Creates a cycle label that is unique to this graph instance."""
        self.cycle_label_count += 1
        return f"{self._CYCLE_LABEL_PREFIX}{self.cycle_label_count}__"

    def deepcopy(self) -> "PseudoGraph":
        """Returns a deep copy of this graph."""
        return copy.deepcopy(self)

    @property
    def edge_count(self) -> int:
        """The number of edges."""
        return len(self._edges)

    def edges(self) -> ValuesView["PseudoEdge"]:
        """A view of graph edges."""
        return self._edges.values()

    def get_edge(self, vertex1: "VertexType", vertex2: "VertexType") -> "PseudoEdge":
        """Returns the :term:`diedge` specified by the vertices, or None if no such edge exists.

        Args:
            vertex1: The first vertex (the :term:`tail` in :term:`directed graphs
                <directed graph>`).
            vertex2: The second vertex (the :term:`head` in directed graphs).

        Returns:
            PseudoEdge: The specified edge.

        Raises:
            KeyError: If the graph does not contain an edge with the specified vertex endpoints.
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        return self._edges[edge_label]

    def get_random_edge(self) -> Optional["PseudoEdge"]:
        """Returns a randomly selected edge from the graph, or None if there are no edges."""
        if self._edges:
            return random.choice(list(self._edges.values()))
        return None

    def vertices(self) -> ValuesView["PseudoVertex"]:
        """A view of the graph vertices."""
        return self._vertices.values()
