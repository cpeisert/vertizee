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
=============
Vertex module
=============

Data types supporting :term:`vertices <vertex>` (also called nodes or points) in a :term:`graph`.

Class and type-alias summary
============================

* :class:`V` - a generic type parameter defined as ``TypeVar("V", bound="VertexBase")``. See
  :class:`VertexBase <vertizee.classes.vertex.VertexBase>`.
* :class:`V_co` - a generic covariant type parameter defined as
  ``TypeVar("V_co", bound="VertexBase", covariant=True)``. See
  :class:`VertexBase <vertizee.classes.vertex.VertexBase>`.
* :class:`VertexType` - A type alias defined as ``Union[VertexBase, VertexLabel, VertexTuple]``,
  where ``VertexLabel`` is either an ``int`` or ``str`` and ``VertexTuple`` is a type alias defined
  as ``Tuple[VertexLabel, AttributesDict]``. An ``AttributesDict`` is a ``dict``.
* :class:`VertexBase` - An abstract base class that defines the minimum API for a :term:`vertex`.
  All other vertex classes derive from ``VertexBase``.
* :class:`Vertex` - A vertex in an :term:`undirected graph`. This is an abstract class that should
  be used for type hints.
* :class:`DiVertex` - A vertex in a :term:`directed graph <digraph>`. This is an abstract class
  that should be used for type hints.
* :class:`MultiVertex` - A vertex in an undirected :term:`multigraph`. This is an abstract class
  that should be used for type hints.
* :class:`MultiDiVertex` - A vertex in a directed :term:`multigraph`. This is an abstract class
  that should be used for type hints.

Function summary
================

* :func:`get_vertex_label` - Returns the vertex label string for the specified vertex.
* :func:`is_vertex_type` - Helper function to determine if a variable is a ``VertexType``.

Detailed documentation
======================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import collections.abc
from typing import Any, cast, Dict, Generic, Optional, Set, Tuple, TYPE_CHECKING, TypeVar, Union

from vertizee.classes.comparable import Comparable
from vertizee.classes.collection_views import SetView

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.edge import DiEdge, Edge, EdgeBase, MultiDiEdge, MultiEdge
    from vertizee.classes.graph import G, GraphBase

# Type aliases
AttributesDict = dict
VertexLabel = Union[int, str]
VertexTuple = Tuple[VertexLabel, AttributesDict]

#: VertexType: A type alias defined as ``Union[V, VertexLabel, VertexTuple]``, where
#: ``VertexLabel`` is either an ``int`` or ``str`` and ``VertexTuple`` is a type alias defined
#: as ``Tuple[VertexLabel, AttributesDict]``. An ``AttributesDict`` is a ``dict``.
VertexType = Union["VertexBase", VertexLabel, VertexTuple]

#: **V** - A generic vertex type parameter.
#: ``V = TypeVar("V", bound=VertexBase)``
V = TypeVar("V", bound="VertexBase")

#: **V_co** - A generic vertex type parameter for use with type-covariant containers.
#: ``V_co = TypeVar("V_co", bound="VertexBase", covariant=True)``
V_co = TypeVar("V_co", bound="VertexBase", covariant=True)

#: **E** - A generic edge type parameter defined in "vertex.py" to avoid circular imports.
#: ``E = TypeVar("E")``
E = TypeVar("E")


def get_vertex_label(other: "VertexType") -> str:
    """Returns the vertex label string for the specified vertex."""
    if isinstance(other, VertexBase):
        return other.label
    if isinstance(other, tuple):
        return str(other[0])
    return str(other)


def is_vertex_type(var: Any) -> bool:
    """Helper function to determine if a variable is a ``VertexType``.

    The ``VertexType`` type alias is defined as ``Union[V, VertexLabel, VertexTuple]``, where
    ``VertexLabel`` is either an ``int`` or ``str`` and ``VertexTuple`` is a type alias defined
    as ``Tuple[VertexLabel, AttributesDict]``. An ``AttributesDict`` is a ``dict``.

    Args:
        var: The variable to test.

    Returns:
        bool: True if the variable is a vertex type as defined by the type alias ``VertexType``.
    """
    if isinstance(var, (int, str, VertexBase)):
        return True
    if isinstance(var, tuple) and len(var) == 2 and isinstance(var[1], collections.abc.Mapping):
        return True
    return False


class VertexBase(ABC, Comparable):
    """Abstract base class from which all vertex classes inherit.

    Args:
        label: The label for this vertex. Must be unique to the graph.
        parent_graph: The parent graph to which this vertex belongs.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.
    """

    __slots__ = ("_attr", "_incident_edges", "_label", "_parent_graph")

    def __init__(
        self,
        label: VertexLabel,
        parent_graph: "GraphBase[VertexBase, EdgeBase[VertexBase]]",
        **attr: Any,
    ) -> None:
        super().__init__()
        self._label = str(label)

        self._attr: Optional[Dict[Any, Any]] = None  # Initialized lazily using property getter.
        for k, v in attr.items():
            self.attr[k] = v

        self._parent_graph = parent_graph
        self._incident_edges: _IncidentEdges[VertexBase, EdgeBase[VertexBase]] = _IncidentEdges(
            self._label, parent_graph
        )

    def __eq__(self, other: object) -> bool:
        try:
            other_label = get_vertex_label(cast(VertexType, other))
        except TypeError:
            return False
        return self._label == other_label

    def __lt__(self, other: object) -> bool:
        try:
            other_label = get_vertex_label(cast(VertexType, other))
        except TypeError:
            return False
        return self._label < other_label

    def __gt__(self, other: object) -> bool:
        return (not self < other) and self != other

    def __le__(self, other: object) -> bool:
        return self < other or self == other

    def __ge__(self, other: object) -> bool:
        return not self < other

    def __getitem__(self, key: Any) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""
        return self.attr[key]

    def __hash__(self) -> int:
        """Supports adding vertices to set-like collections."""
        return hash(self._label)

    def __repr__(self) -> str:
        return self.__str__()

    def __setitem__(self, key: Any, value: Any) -> None:
        """Supports index accessor notation to set values in the ``attr`` dictionary."""
        self.attr[key] = value

    def __str__(self) -> str:
        return self._label

    @abstractmethod
    def adj_vertices(self) -> "SetView[VertexBase]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all adjacent
        vertices."""

    @property
    def attr(self) -> Dict[str, Any]:
        """Attribute dictionary to store optional data associated with a vertex."""
        if not self._attr:
            self._attr = dict()
        return self._attr

    @property
    @abstractmethod
    def degree(self) -> int:
        """The degree (or valance) of this vertex. The degree is the number of incident edges.
        Self-loops are counted twice."""

    def has_attributes_dict(self) -> bool:
        """Returns True if this vertex has an instantiated ``attr`` dictionary. Since the
        ``attr`` dictionary is only created as needed, this method can be used to save memory.
        Calling an ``attr`` accessor (such as the ``attr`` property), results in automatic
        dictionary instantiation."""
        return self._attr is not None

    @abstractmethod
    def incident_edges(self) -> "SetView[EdgeBase[VertexBase]]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all incident
        edges (including self-loops)."""

    def is_isolated(self, ignore_self_loops: bool = False) -> bool:
        """Returns True if this vertex has degree zero, that is, no incident edges.

        Args:
            ignore_self_loops: If True, then self-loops are ignored, meaning that a vertex whose
                only incident edges are self-loops will be considered isolated. Defaults to False.
        """
        if self.degree == 0:
            return True
        if ignore_self_loops and self._incident_edges.loop_edge:
            adjacent = self._incident_edges.adj_vertices.difference({self})
            return len(adjacent) == 0
        return False

    @property
    def label(self) -> str:
        """The vertex label string."""
        return self._label

    @property
    @abstractmethod
    def loop_edge(self) -> Optional[EdgeBase[VertexBase]]:
        """Returns the loop edge if this vertex has a self loop, otherwise None."""

    def remove(self) -> None:
        """Removes this vertex.

        For a vertex to be removed, it must be :term:`semi-isolated`, meaning that it has no
        incident edges except for :term:`self-loops <self-loop>`. Any non-loop incident edges must
        be deleted prior to vertex removal.

        Raises:
            VertizeeException: If the vertex has non-loop incident edges.

        See Also:
            :meth:`remove_incident_edges`
        """
        self._parent_graph.remove_vertex(self._label)

    def remove_incident_edges(self) -> int:
        """Removes all edges from the graph that are incident on this vertex.

        Returns:
            int: The number of edges deleted. In multigraphs, this includes the count of parallel
            edge connections.

        See Also:
            :meth:`remove`
        """
        deletion_count = 0
        for edge in self._incident_edges.incident_edges:
            if self._parent_graph.is_multigraph():
                deletion_count += edge.multiplicity  # type: ignore
            else:
                deletion_count += 1
            self._incident_edges.remove_edge(edge)
        return deletion_count

    def remove_loops(self) -> int:
        """Removes all edges that are loops on this vertex.

        Returns:
            int: The number of loops deleted. In multigraphs, this includes the count of parallel
            loop connections.
        """
        deletion_count = 0
        loops = self._incident_edges.loop_edge
        if loops:
            if self._parent_graph.is_multigraph():
                deletion_count = loops.multiplicity  # type: ignore
            else:
                deletion_count = 1
            self._incident_edges.remove_edge(loops)
            self._parent_graph.remove_edge(loops.vertex1, loops.vertex2)
        return deletion_count

    def _add_edge(self, edge: EdgeBase[VertexBase]) -> None:
        """Adds an edge.

        If an incident edge already exists with the same vertices, it is overwritten.

        Raises:
            ValueError: If the new edge does not include this vertex.
        """
        self._incident_edges.add_edge(edge)

    def _remove_edge(self, edge: EdgeBase[VertexBase]) -> None:
        """Removes an incident edge.

        Args:
            edge: The edge to remove.
        """
        self._incident_edges.remove_edge(edge)


class Vertex(VertexBase):
    """A graph primitive representing a vertex (also called a node) that may be connected to other
    vertices via undirected edges.

    To help ensure the integrity of graphs, the ``Vertex`` class is abstract and cannot be
    instantiated directly. To create vertices, use :meth:`Graph.add_vertex
    <vertizee.classes.graph.Graph.add_vertex>` and :meth:`Graph.add_vertices_from
    <vertizee.classes.graph.G.add_vertices_from>`.

    No two vertices within a graph may share the same label. Labels may be strings or integers, but
    internally, they are always stored as strings. The following statements are equivalent::

        graph.add_vertex(1)
        graph.add_vertex("1")

    Any context that accepts a ``VertexType`` (defined as
    ``Union[V, VertexLabel, VertexTuple]``) permits specifying vertices using labels
    (i.e. strings or integers), ``Vertex`` objects, or a tuple of the form
    ``Tuple[VertexLabel, AttributesDict]``. For example, the following statements are equivalent
    and return the vertex object with label "1"::

        obj_one: Vertex = graph.add_vertex(1)
        graph[obj_one]
        graph[1]
        graph["1"]

    Args:
        label: The label for this vertex. Must be unique to the graph.
        parent_graph: The parent graph to which this vertex belongs.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

    See Also:
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :class:`Graph <vertizee.classes.graph.Graph>`
    """

    __slots__ = ()

    def __init__(
        self, label: VertexLabel, parent_graph: "GraphBase[Vertex, Edge]", **attr: Any
    ) -> None:
        super().__init__(label, parent_graph, **attr)

        self._incident_edges: _IncidentEdges[Vertex, Edge] = self._incident_edges  # type: ignore

    @abstractmethod
    def adj_vertices(self) -> "SetView[Vertex]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all adjacent
        vertices."""

    @abstractmethod
    def incident_edges(self) -> "SetView[Edge]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all incident
        edges (including self-loops)."""

    @property
    @abstractmethod
    def loop_edge(self) -> Optional[Edge]:
        """Returns the loop edge if this vertex has a self loop, otherwise None."""


class DiVertex(VertexBase):
    """A graph primitive representing a vertex (also called a node) in a digraph that may be
    connected to other vertices via directed edges.

    To help ensure the integrity of graphs, the ``DiVertex`` class is abstract and cannot be
    instantiated directly. To create divertices, use :meth:`DiGraph.add_vertex
    <vertizee.classes.graph.DiGraph.add_vertex>` and :meth:`DiGraph.add_vertices_from
    <vertizee.classes.graph.G.add_vertices_from>`.

    No two vertices within a graph may share the same label. Labels may be strings or integers, but
    internally, they are always stored as strings. The following statements are equivalent::

        digraph.add_vertex(1)
        digraph.add_vertex("1")

    Any context that accepts a ``VertexType`` (defined as
    ``Union[V, VertexLabel, VertexTuple]``) permits specifying vertices using labels
    (i.e. strings or integers), ``DiVertex`` objects, or a tuple of the form
    ``Tuple[VertexLabel, AttributesDict]``. For example, the following statements are equivalent
    and return the vertex object with label "1"::

        obj_one: DiVertex = digraph.add_vertex(1)
        digraph[obj_one]
        digraph[1]
        digraph["1"]

    Args:
        label: The label for this vertex. Must be unique to the graph.
        parent_graph: The parent graph to which this vertex belongs.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

    See Also:
        * :class:`DiEdge <vertizee.classes.edge.DiEdge>`
        * :class:`DiGraph <vertizee.classes.graph.DiGraph>`
    """

    __slots__ = ()

    def __init__(
        self, label: VertexLabel, parent_graph: "GraphBase[DiVertex, DiEdge]", **attr: Any
    ) -> None:
        super().__init__(label, parent_graph, **attr)

        self._incident_edges: _IncidentEdges[
            DiVertex, DiEdge
        ] = self._incident_edges  # type: ignore

    @abstractmethod
    def adj_vertices(self) -> "SetView[DiVertex]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all adjacent
        vertices."""

    @abstractmethod
    def adj_vertices_incoming(self) -> "SetView[DiVertex]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all adjacent
        vertices from incoming edges."""

    @abstractmethod
    def adj_vertices_outgoing(self) -> "SetView[DiVertex]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all adjacent
        vertices from outgoing edges."""

    @abstractmethod
    def incident_edges(self) -> "SetView[DiEdge]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all incident
        edges (incoming, outgoing, and self-loops)."""

    @abstractmethod
    def incident_edges_incoming(self) -> "SetView[DiEdge]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of incoming
        incident edges (i.e. edges where this vertex is the head)."""

    @abstractmethod
    def incident_edges_outgoing(self) -> "SetView[DiEdge]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of outgoing
        incident edges (i.e. edges where this vertex is the tail)."""

    @property
    @abstractmethod
    def indegree(self) -> int:
        """The indegree of this vertex, which is the number of incoming incident edges."""

    @property
    @abstractmethod
    def loop_edge(self) -> Optional[DiEdge]:
        """The loop edge if this vertex has a self loop."""

    @property
    @abstractmethod
    def outdegree(self) -> int:
        """The outdegree of this vertex, which is the number of outgoing incident edges."""


class MultiVertex(VertexBase):
    """A graph primitive representing a vertex (also called a node) that may be connected to other
    vertices via undirected edges in a multigraph.

    To help ensure the integrity of graphs, the ``MultiVertex`` class is abstract and cannot be
    instantiated directly. To create vertices, use :meth:`MultiGraph.add_vertex
    <vertizee.classes.graph.MultiGraph.add_vertex>` and :meth:`MultiGraph.add_vertices_from
    <vertizee.classes.graph.G.add_vertices_from>`.

    Args:
        label: The label for this vertex. Must be unique to the graph.
        parent_graph: The parent graph to which this vertex belongs.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

    See Also:
        * :class:`MultiEdge <vertizee.classes.edge.MultiEdge>`
        * :class:`MultiGraph <vertizee.classes.graph.MultiGraph>`
        * :class:`Vertex`
    """

    __slots__ = ()

    def __init__(
        self, label: VertexLabel, parent_graph: "GraphBase[MultiVertex, MultiEdge]", **attr: Any
    ) -> None:
        super().__init__(label, parent_graph, **attr)

        self._incident_edges: _IncidentEdges[
            MultiVertex, MultiEdge
        ] = self._incident_edges  # type: ignore

    @abstractmethod
    def adj_vertices(self) -> "SetView[MultiVertex]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all adjacent
        vertices."""

    @abstractmethod
    def incident_edges(self) -> "SetView[MultiEdge]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all incident
        edges (including self-loops)."""

    @property
    @abstractmethod
    def loop_edge(self) -> Optional[MultiEdge]:
        """The loop multiedge if this vertex has one or more self loops."""


class MultiDiVertex(VertexBase):
    """A graph primitive representing a vertex (also called a node) in a directed graph that may be
    connected to other vertices via directed edges in a multigraph.

    To help ensure the integrity of graphs, the ``MultiDiVertex`` class is abstract and cannot be
    instantiated directly. To create divertices, use :meth:`MultiDiGraph.add_vertex
    <vertizee.classes.graph.MultiDiGraph.add_vertex>` and :meth:`MultiDiGraph.add_vertices_from
    <vertizee.classes.graph.G.add_vertices_from>`.

    Args:
        label: The label for this vertex. Must be unique to the graph.
        parent_graph: The parent graph to which this vertex belongs.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

    See Also:
        * :class:`MultiDiEdge <vertizee.classes.edge.MultiDiEdge>`
        * :class:`MultiDiGraph <vertizee.classes.graph.MultiDiGraph>`
        * :class:`DiVertex`
    """

    __slots__ = ()

    def __init__(
        self, label: VertexLabel, parent_graph: "GraphBase[MultiDiVertex, MultiDiEdge]", **attr: Any
    ) -> None:
        super().__init__(label, parent_graph, **attr)

        self._incident_edges: _IncidentEdges[
            MultiDiVertex, MultiDiEdge
        ] = self._incident_edges  # type: ignore

    @abstractmethod
    def adj_vertices(self) -> "SetView[MultiDiVertex]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all adjacent
        vertices."""

    @abstractmethod
    def adj_vertices_incoming(self) -> "SetView[MultiDiVertex]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all adjacent
        vertices from incoming edges."""

    @abstractmethod
    def adj_vertices_outgoing(self) -> "SetView[MultiDiVertex]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all adjacent
        vertices from outgoing edges."""

    @abstractmethod
    def incident_edges(self) -> "SetView[MultiDiEdge]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of all incident
        edges (incoming, outgoing, and self-loops)."""

    @abstractmethod
    def incident_edges_incoming(self) -> "SetView[MultiDiEdge]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of incoming
        incident edges (i.e. edges where this vertex is the head)."""

    @abstractmethod
    def incident_edges_outgoing(self) -> "SetView[MultiDiEdge]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of outgoing
        incident edges (i.e. edges where this vertex is the tail)."""

    @property
    @abstractmethod
    def indegree(self) -> int:
        """The indegree of this vertex, which is the number of incoming incident edges."""

    @property
    @abstractmethod
    def loop_edge(self) -> Optional["MultiDiEdge"]:
        """The loop edge if this vertex has a self loop."""

    @property
    @abstractmethod
    def outdegree(self) -> int:
        """The outdegree of this vertex, which is the number of outgoing incident edges."""


class _Vertex(Vertex):
    """Concrete implementation of the abstract :class:`Vertex` class."""

    __slots__ = ()

    def adj_vertices(self) -> SetView[Vertex]:
        return SetView(self._incident_edges.adj_vertices)

    @property
    def degree(self) -> int:
        """The degree (or valance) of this vertex. The degree is the number of incident edges.
        Self-loops are counted twice."""
        total = len(self.incident_edges())
        if self.loop_edge:
            total += 1
        return total

    def incident_edges(self) -> SetView[Edge]:
        return SetView(self._incident_edges.incident_edges)

    @property
    def loop_edge(self) -> Optional[Edge]:
        return self._incident_edges.loop_edge


class _DiVertex(DiVertex):
    """Concrete implementation of the abstract :class:`DiVertex` class."""

    __slots__ = ()

    def adj_vertices(self) -> SetView[DiVertex]:
        return SetView(self._incident_edges.adj_vertices)

    def adj_vertices_incoming(self) -> SetView[DiVertex]:
        return SetView(self._incident_edges.adj_vertices_incoming)

    def adj_vertices_outgoing(self) -> SetView[DiVertex]:
        return SetView(self._incident_edges.adj_vertices_outgoing)

    @property
    def degree(self) -> int:
        """The degree (or valance) of this vertex. The degree is the number of incident edges.
        Self-loops are counted twice."""
        total = len(self.incident_edges())
        if self.loop_edge:
            total += 1
        return total

    def incident_edges(self) -> SetView[DiEdge]:
        return SetView(self._incident_edges.incident_edges)

    def incident_edges_incoming(self) -> SetView[DiEdge]:
        return SetView(self._incident_edges.incoming)

    def incident_edges_outgoing(self) -> SetView[DiEdge]:
        return SetView(self._incident_edges.outgoing)

    @property
    def indegree(self) -> int:
        return len(self._incident_edges.incoming)

    @property
    def loop_edge(self) -> Optional[DiEdge]:
        return self._incident_edges.loop_edge

    @property
    def outdegree(self) -> int:
        return len(self._incident_edges.outgoing)


class _MultiVertex(MultiVertex):
    """Concrete implementation of the abstract :class:`MultiVertex` class."""

    __slots__ = ()

    def adj_vertices(self) -> SetView[MultiVertex]:
        return SetView(self._incident_edges.adj_vertices)

    @property
    def degree(self) -> int:
        """The degree (or valance) of this vertex. The degree is the number of incident edges.
        Self-loops are counted twice."""
        total = sum(e.multiplicity for e in self.incident_edges())
        if self.loop_edge:
            total += self.loop_edge.multiplicity
        return total

    def incident_edges(self) -> SetView[MultiEdge]:
        return SetView(self._incident_edges.incident_edges)

    @property
    def loop_edge(self) -> Optional[MultiEdge]:
        return self._incident_edges.loop_edge


class _MultiDiVertex(MultiDiVertex):
    """Concrete implementation of the abstract :class:`MultiDiVertex` class."""

    __slots__ = ()

    def adj_vertices(self) -> SetView[MultiDiVertex]:
        return SetView(self._incident_edges.adj_vertices)

    def adj_vertices_incoming(self) -> SetView[MultiDiVertex]:
        return SetView(self._incident_edges.adj_vertices_incoming)

    def adj_vertices_outgoing(self) -> SetView[MultiDiVertex]:
        return SetView(self._incident_edges.adj_vertices_outgoing)

    @property
    def degree(self) -> int:
        """The degree (or valance) of this vertex. The degree is the number of incident edges.
        Self-loops are counted twice."""
        total = sum(e.multiplicity for e in self.incident_edges())
        if self.loop_edge:
            total += self.loop_edge.multiplicity
        return total

    def incident_edges(self) -> SetView[MultiDiEdge]:
        return SetView(self._incident_edges.incident_edges)

    def incident_edges_incoming(self) -> SetView[MultiDiEdge]:
        return SetView(self._incident_edges.incoming)

    def incident_edges_outgoing(self) -> SetView[MultiDiEdge]:
        return SetView(self._incident_edges.outgoing)

    @property
    def indegree(self) -> int:
        indegree_total = 0
        for e in self._incident_edges.incoming:
            indegree_total += e.multiplicity
        return indegree_total

    @property
    def loop_edge(self) -> Optional[MultiDiEdge]:
        return self._incident_edges.loop_edge

    @property
    def outdegree(self) -> int:
        outdegree_total = 0
        for e in self._incident_edges.outgoing:
            outdegree_total += e.multiplicity
        return outdegree_total


def _create_edge_label(v1_label: str, v2_label: str, is_directed: bool) -> str:
    """Creates a consistent string representation of an :term:`edge`.

    This function is used instead of `edge.create_edge_label` to avoid circular dependencies.

    Args:
        v1_label: The first vertex label of the edge.
        v2_label: The second vertex label of the edge.
        is_directed (bool): True indicates a directed edge, False an undirected edge.

    Returns:
        str: The edge label.
    """
    if not is_directed and v1_label > v2_label:
        return f"({v2_label}, {v1_label})"
    return f"({v1_label}, {v2_label})"


# pylint: disable=used-before-assignment
class _IncidentEdges(Generic[V, E]):
    """Collection of :term:`edges <edge>` that are incident on a shared :term:`vertex`.

    Attempting to add an edge that does not have the shared vertex raises an error.
    :term:`Self loops <loop>` are tracked and may be accessed with the ``loop_edge`` property. Loops
    cause a vertex to be adjacent to itself and the loop is an :term:`incident edge <incident>`.

    In the case of directed edges, incident edges may be also be filtered as ``incoming`` or
    ``outgoing``.

    Args:
        shared_vertex_label: The vertex label of the vertex shared by the incident edges.
        parent_graph: The graph to which the incident edges belong.
    """

    __slots__ = ("_incident_edge_labels", "has_loop", "parent_graph", "shared_vertex_label")

    def __init__(self, shared_vertex_label: str, parent_graph: "G") -> None:
        self._incident_edge_labels: Optional[Set[str]] = None
        """The set of edge labels for edges adjacent to the shared vertex."""

        self.has_loop = False
        self.parent_graph = parent_graph

        self.shared_vertex_label: str = shared_vertex_label
        """The label of the vertex common between all of the incident edges."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _IncidentEdges):
            return False
        if self.shared_vertex_label != other.shared_vertex_label or len(self.incident_edges) != len(
            other.incident_edges
        ):
            return False
        if self.incident_edges != other.incident_edges:
            return False
        return True

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        str_edges = ", ".join(self.incident_edge_labels)
        return f"_IncidentEdges: {str_edges}"

    def add_edge(self, edge: EdgeBase[V]) -> None:
        """Adds an edge incident on the vertex specified by ``shared_vertex_label``.

        Args:
            edge: The edge to add.
        """
        shared = self.shared_vertex_label
        if shared not in (edge.vertex1.label, edge.vertex2.label):
            raise ValueError(
                f"cannot add edge {edge} since it is not incident on vertex " f"'{shared}'"
            )

        self.incident_edge_labels.add(edge.label)
        if edge.is_loop():
            self.has_loop = True

    @property
    def adj_vertices(self) -> Set[V]:
        """The set of all vertices adjacent to the shared vertex."""
        vertex_labels: Set[str] = set()
        for label in self.incident_edge_labels:
            comma_pos = label.find(",")
            v1_label = label[1:comma_pos]
            v2_label = label[comma_pos + 1 : -1].strip()
            vertex_labels.add(v1_label)
            vertex_labels.add(v2_label)

        if not self.has_loop and self.shared_vertex_label in vertex_labels:
            vertex_labels.remove(self.shared_vertex_label)

        vertex_set = set()
        for vertex_label in vertex_labels:
            vertex_set.add(self.parent_graph._vertices[vertex_label])
        return vertex_set

    @property
    def adj_vertices_incoming(self) -> Set[V]:
        """The set of all vertices adjacent to the shared vertex from incoming edges. For undirected
        graphs, this is an empty set."""
        if not self.parent_graph.is_directed():
            return set()

        vertex_labels: Set[str] = set()
        for label in self.incident_edge_labels:
            comma_pos = label.find(",")
            v1_label = label[1:comma_pos]
            if v1_label != self.shared_vertex_label:
                vertex_labels.add(v1_label)

        if self.has_loop:
            vertex_labels.add(self.shared_vertex_label)

        vertex_set = set()
        for vertex_label in vertex_labels:
            vertex_set.add(self.parent_graph._vertices[vertex_label])
        return vertex_set

    @property
    def adj_vertices_outgoing(self) -> Set[V]:
        """Set of all vertices adjacent to the shared vertex from outgoing edges. For undirected
        graphs, this is an empty set."""
        vertex_labels: Set[str] = set()
        for label in self.incident_edge_labels:
            comma_pos = label.find(",")
            v2_label = label[comma_pos + 1 : -1].strip()
            if v2_label != self.shared_vertex_label:
                vertex_labels.add(v2_label)

        if self.has_loop:
            vertex_labels.add(self.shared_vertex_label)

        vertex_set = set()
        for vertex_label in vertex_labels:
            vertex_set.add(self.parent_graph._vertices[vertex_label])
        return vertex_set

    @property
    def incident_edge_labels(self) -> Set[str]:
        """Property to handle lazy initialization of incident edge label set."""
        if not self._incident_edge_labels:
            self._incident_edge_labels = set()
        return self._incident_edge_labels

    @property
    def incident_edges(self) -> Set[E]:
        """The set of all incident edges: parallel, self loops, incoming, and outgoing."""
        edge_set = set()
        for edge_label in self.incident_edge_labels:
            edge_set.add(self.parent_graph._edges[edge_label])
        return edge_set

    @property
    def incoming(self) -> Set[E]:
        """Edges whose head vertex is ``shared_vertex_label``. For undirected graphs, this is an
        empty set."""
        if not self.parent_graph.is_directed():
            return set()

        edge_labels = set()
        for label in self.incident_edge_labels:
            comma_pos = label.find(",")
            v1_label = label[1:comma_pos]
            if v1_label != self.shared_vertex_label:
                edge_labels.add(label)

        if self.has_loop:
            shared = self.shared_vertex_label
            edge_labels.add(_create_edge_label(shared, shared, self.parent_graph.is_directed()))

        edge_set = set()
        for edge_label in edge_labels:
            edge_set.add(self.parent_graph._edges[edge_label])
        return edge_set

    @property
    def loop_edge(self) -> Optional[E]:
        """Loops on ``shared_vertex_label``.

        Since all loops are parallel to each other, only one Edge object is needed.
        """
        if not self.has_loop:
            return None
        return cast(
            E, self.parent_graph.get_edge(self.shared_vertex_label, self.shared_vertex_label)
        )

    @property
    def outgoing(self) -> Set[E]:
        """Edges whose tail vertex is ``shared_vertex_label``. For undirected graphs, this is an
        empty set."""
        if not self.parent_graph.is_directed():
            return set()

        edge_labels = set()
        for label in self.incident_edge_labels:
            comma_pos = label.find(",")
            v2_label = label[comma_pos + 1 : -1].strip()
            if v2_label != self.shared_vertex_label:
                edge_labels.add(label)

        if self.has_loop:
            shared = self.shared_vertex_label
            edge_labels.add(_create_edge_label(shared, shared, self.parent_graph.is_directed()))

        edge_set = set()
        for edge_label in edge_labels:
            edge_set.add(self.parent_graph._edges[edge_label])
        return edge_set

    def remove_edge(self, edge: EdgeBase[V]) -> None:
        """Removes an incident edge."""
        if edge.label in self.incident_edge_labels:
            self.incident_edge_labels.remove(edge.label)
            if edge.is_loop():
                self.has_loop = False
