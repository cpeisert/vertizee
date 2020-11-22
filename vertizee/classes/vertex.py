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

"""Data types supporting vertices (also called nodes or points) in a graph.

Classes and type aliases:

* :class:`VertexBase(ABC, Generic[E]) <VertexBase>` - A generic abstract base class that defines
  the minimum API for a vertex in a graph. The vertex classes inherit from ``VertexBase``.
* :class:`Vertex(VertexBase[Edge]) <Vertex>` - A vertex in an undirected graph. This is an abstract
  class that should be used for type hints.
* :class:`DiVertex(VertexBase[DiEdge]) <DiVertex>` - A vertex in a directed graph. This is an
  abstract class that should be used for type hints.
* :class:`MultiVertex(VertexBase[MultiEdge]) <MultiVertex>` - A vertex in an undirected multigraph.
  This is an abstract class that should be used for type hints.
* :class:`MultiDiVertex(VertexBase[MultiDiEdge]) <MultiDiVertex>` - A vertex in a directed
  multigraph. This is an abstract class that should be used for type hints.
* :class:`VertexType` - A type alias defined as ``Union[VertexLabel, VertexTupleAttr, Vertex]``.
  Any context that accepts ``VertexType`` permits referring to vertices by label (``str`` or
  ``int``), by vertex object, or by tuple (``Tuple[VertexLabel, AttributesDict]``).

Functions:

* :func:`get_vertex_label` - Returns the vertex label string for the specified vertex.
* :func:`is_vertex_type` - Helper function to determine if a variable is a ``VertexType``.
"""
# pylint: disable=unsubscriptable-object
# See pylint issue #2822: https://github.com/PyCQA/pylint/issues/2822

from __future__ import annotations
from abc import ABC, abstractmethod
import collections.abc
from typing import Any, Generic, Optional, Set, Tuple, TYPE_CHECKING, TypeVar, Union

from vertizee.classes.collection_views import SetView

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.edge import (
        Connection, DiEdge, Edge, EdgeClass, MultiConnection, MultiDiEdge, MultiEdge
    )
    from vertizee.classes.graph import G
    from vertizee.classes.primitives_parsing import GraphPrimitive

# Type aliases
AttributesDict = dict
VertexClass = Union["DiVertex", "MultiDiVertex", "MultiVertex", "Vertex", "VertexBase"]
VertexLabel = Union[int, str]
VertexTupleAttr = Tuple[VertexLabel, AttributesDict]

#: VertexType: A type alias defined as ``Union[VertexClass, VertexLabel, VertexTupleAttr]``,
# where ``VertexClass`` is an alias for various the vertex classes, ``VertexLabel`` is either
# an ``int`` or ``str``, and ``VertexTupleAttr`` is a type alias defined as
# ``Tuple[VertexLabel, AttributesDict]``.
VertexType = Union[VertexClass, VertexLabel, VertexTupleAttr]

# Generic type parameters.

#: E: A generic type parameter that represents an edge (for example, DiEdge, Edge, MultiDiEdge,
# MultiEdge).
E = TypeVar("E", bound=Union["Connection", "MultiConnection"])

#: V: A generic type parameter that represents a vertex (for example, DiVertex, MultiDiVertex,
# MultiVertex, Vertex).
V = TypeVar("V", bound="VertexBase")


def get_vertex_label(other: "VertexType") -> str:
    """Returns the vertex label string for the specified vertex."""
    if isinstance(other, VertexBase):
        return other.label
    if isinstance(other, tuple):
        return other[0]
    return str(other)


def is_vertex_type(var: Any) -> bool:
    """Helper function to determine if a variable is a ``VertexType``.

    The ``VertexType`` type alias is defined as ``Union[VertexClass, VertexLabel, VertexTupleAttr]``
    and ``VertexTupleAttr`` is defined as ``Tuple[VertexLabel, AttributesDict]``. A vertex label may
    be a string or integer.

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


class VertexBase(ABC, Generic[E]):
    """Abstract base class from which all vertex classes inherit.

    Args:
        label: The label for this vertex. Must be unique to the graph.
        parent_graph: The parent graph to which this vertex belongs.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.
    """

    __slots__ = ('_attr', '_incident_edges', '_label', '_parent_graph')

    def __init__(self, label: VertexLabel, parent_graph: G, **attr) -> None:
        self._label = str(label)

        self._attr: Optional[dict] = None  # Initialized lazily using property getter.
        for k, v in attr.items():
            self.attr[k] = v

        self._incident_edges: _IncidentEdges[E] = \
            _IncidentEdges(self.label, parent_graph)
        self._parent_graph = parent_graph

    def __compare(self, other: VertexType, operator: str) -> bool:
        try:
            other_label = get_vertex_label(other)
        except TypeError:
            return False
        compare = False
        if operator == "==":
            if self.label == other_label:
                compare = True
        elif operator == "<":
            if self.label < other_label:
                compare = True
        elif operator == "<=":
            if self.label <= other_label:
                compare = True
        elif operator == ">":
            if self.label > other_label:
                compare = True
        elif operator == ">=":
            if self.label >= other_label:
                compare = True
        return compare

    # mypy: See https://github.com/python/mypy/issues/2783#issuecomment-579868936
    def __eq__(self, other: VertexType) -> bool:  # type: ignore[override]
        return self.__compare(other, "==")

    def __getitem__(self, key: Any) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""
        return self.attr[key]

    def __ge__(self, other: VertexType) -> bool:
        return self.__compare(other, ">=")

    def __gt__(self, other: VertexType) -> bool:
        return self.__compare(other, ">")

    def __hash__(self) -> int:
        return hash(self.label)

    def __le__(self, other: VertexType) -> bool:
        return self.__compare(other, "<=")

    def __lt__(self, other: VertexType) -> bool:
        return self.__compare(other, "<")

    def __repr__(self) -> str:
        return self.__str__()

    def __setitem__(self, key: Any, value: Any) -> None:
        """Supports index accessor notation to set values in the ``attr`` dictionary."""
        self.attr[key] = value

    def __str__(self) -> str:
        return self.label

    @abstractmethod
    def adj_vertices(self) -> SetView[VertexBase]:
        """Returns a dynamic, set-like view of all adjacent vertices."""

    @property
    def attr(self) -> dict:
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
    def incident_edges(self) -> SetView[E]:
        """Returns a dynamic, set-like view of all incident edges (incoming, outgoing, and
        self-loops)."""

    def is_isolated(self) -> bool:
        """Returns True if this vertex has no incident edges other than self loops."""
        if self.degree == 0:
            return True
        if self._incident_edges.loop_edge:
            adjacent = self._incident_edges.adj_vertices.difference({self})
            return len(adjacent) == 0
        return False

    @property
    def label(self) -> str:
        """The vertex label string."""
        return self._label

    @property
    @abstractmethod
    def loop_edge(self) -> Optional[E]:
        """The loop edge (or edges) if this vertex has one or more self loops."""

    def remove(self) -> None:
        """Removes this vertex.

        For a vertex to be removed, it must be isolated. That means that the vertex has no incident
        edges (except self loops). Any incident edges must be deleted prior to vertex removal.

        Raises:
            VertizeeException: If the vertex has non-loop incident edges.

        See Also:
            :meth:`remove_incident_edges`
        """
        self._parent_graph.remove_vertex(self)

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
                deletion_count += edge.multiplicity
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
                deletion_count = loops.multiplicity
            else:
                deletion_count = 1
            self._incident_edges.remove_edge(loops)
            self._parent_graph.remove_edge(loops.vertex1, loops.vertex2)
        return deletion_count

    def _add_edge(self, edge: E) -> None:
        """Adds an edge.

        If an incident edge already exists with the same vertices, it is overwritten.

        Raises:
            ValueError: If the new edge does not include this vertex.
        """
        self._incident_edges.add_edge(edge)

    def _remove_edge(self, edge: E) -> None:
        """Removes an incident edge.

        Args:
            edge: The edge to remove.
        """
        self._incident_edges.remove_edge(edge)


class Vertex(VertexBase["Edge"]):
    """A graph primitive representing a vertex (also called a node) that may be connected to other
    vertices via undirected edges.

    To help ensure the integrity of graphs, the ``Vertex`` class is abstract and cannot be
    instantiated directly. To create vertices, use :meth:`Graph.add_vertex
    <vertizee.classes.graph.Graph.add_vertex>` and :meth:`Graph.add_vertices_from
    <vertizee.classes.graph.Graph.add_vertices_from>`.

    No two vertices within a graph may share the same label. Labels may be strings or integers, but
    internally, they are always stored as strings. The following statements are equivalent::

        graph.add_vertex(1)
        graph.add_vertex("1")

    Any context that accepts a ``VertexType`` (defined as
    ``Union[VertexClass, VertexLabel, VertexTupleAttr]``) permits specifying vertices using labels
    (i.e. strings or integers), ``Vertex`` objects, or a tuple of the form
    ``Tuple[VertexLabel, AttributesDict]``. For example, the following statements are equivalent
    and return the vertex object with label "1"::

        obj_one: Vertex = graph.add_vertex(1)
        graph[obj_one]
        graph[1]
        graph["1"]

    See Also:
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :class:`Graph <vertizee.classes.graph.Graph>`
    """

    __slots__ = ()

    @abstractmethod
    def adj_vertices(self) -> SetView[Vertex]:
        """Returns a dynamic, set-like view of all adjacent vertices."""

    @abstractmethod
    def incident_edges(self) -> SetView[Edge]:
        """Returns a dynamic, set-like view of all incident edges (including self-loops)."""

    @property
    @abstractmethod
    def loop_edge(self) -> Optional[Edge]:
        """The loop edge if this vertex has a self loop."""


class DiVertex(VertexBase["DiEdge"]):
    """A graph primitive representing a vertex (also called a node) in a directed graph that may be
    connected to other vertices via directed edges.

    To help ensure the integrity of graphs, the ``DiVertex`` class is abstract and cannot be
    instantiated directly. To create divertices, use :meth:`DiGraph.add_vertex
    <vertizee.classes.graph.DiGraph.add_vertex>` and :meth:`DiGraph.add_vertices_from
    <vertizee.classes.graph.DiGraph.add_vertices_from>`.

    No two vertices within a graph may share the same label. Labels may be strings or integers, but
    internally, they are always stored as strings. The following statements are equivalent::

        digraph.add_vertex(1)
        digraph.add_vertex("1")

    Any context that accepts a ``VertexType`` (defined as
    ``Union[VertexClass, VertexLabel, VertexTupleAttr]``) permits specifying vertices using labels
    (i.e. strings or integers), ``DiVertex`` objects, or a tuple of the form
    ``Tuple[VertexLabel, AttributesDict]``. For example, the following statements are equivalent
    and return the vertex object with label "1"::

        obj_one: DiVertex = digraph.add_vertex(1)
        digraph[obj_one]
        digraph[1]
        digraph["1"]

    See Also:
        * :class:`DiEdge <vertizee.classes.edge.DiEdge>`
        * :class:`DiGraph <vertizee.classes.graph.DiGraph>`
    """

    __slots__ = ()

    @abstractmethod
    def adj_vertices(self) -> SetView[DiVertex]:
        """Returns a dynamic, set-like view of all adjacent vertices."""

    @abstractmethod
    def adj_vertices_incoming(self) -> SetView[DiVertex]:
        """Returns a dynamic, set-like view of all adjacent vertices from incoming edges."""

    @abstractmethod
    def adj_vertices_outgoing(self) -> SetView[DiVertex]:
        """Returns a dynamic, set-like view of all adjacent vertices from outgoing edges."""

    @abstractmethod
    def incident_edges(self) -> SetView[DiEdge]:
        """Returns a dynamic, set-like view of all incident edges (incoming, outgoing, and
        self-loops)."""

    @abstractmethod
    def incident_edges_incoming(self) -> SetView[DiEdge]:
        """Returns a dynamic, set-like view of incoming incident edges (i.e. edges where this
        vertex is the head)."""

    @abstractmethod
    def incident_edges_outgoing(self) -> SetView[DiEdge]:
        """Returns a dynamic, set-like view of outgoing incident edges (i.e. edges where this
        vertex is the tail)."""

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


class MultiVertex(VertexBase["MultiEdge"]):
    """A graph primitive representing a vertex (also called a node) that may be connected to other
    vertices via undirected edges in a multigraph.

    To help ensure the integrity of graphs, the ``MultiVertex`` class is abstract and cannot be
    instantiated directly. To create vertices, use :meth:`MultiGraph.add_vertex
    <vertizee.classes.graph.MultiGraph.add_vertex>` and :meth:`MultiGraph.add_vertices_from
    <vertizee.classes.graph.MultiGraph.add_vertices_from>`.

    See Also:
        * :class:`MultiEdge <vertizee.classes.edge.MultiEdge>`
        * :class:`MultiGraph <vertizee.classes.graph.MultiGraph>`
        * :class:`Vertex`
    """

    __slots__ = ()

    @abstractmethod
    def adj_vertices(self) -> SetView[MultiVertex]:
        """Returns a dynamic, set-like view of all adjacent vertices."""

    @abstractmethod
    def incident_edges(self) -> SetView[MultiEdge]:
        """Returns a dynamic, set-like view of all incident edges (including self-loops)."""

    @property
    @abstractmethod
    def loop_edge(self) -> Optional[MultiEdge]:
        """The loop multiedge if this vertex has one or more self loops."""


class MultiDiVertex(VertexBase["MultiDiEdge"]):
    """A graph primitive representing a vertex (also called a node) in a directed graph that may be
    connected to other vertices via directed edges in a multigraph.

    To help ensure the integrity of graphs, the ``MultiDiVertex`` class is abstract and cannot be
    instantiated directly. To create divertices, use :meth:`MultiDiGraph.add_vertex
    <vertizee.classes.graph.MultiDiGraph.add_vertex>` and :meth:`MultiDiGraph.add_vertices_from
    <vertizee.classes.graph.MultiDiGraph.add_vertices_from>`.

    See Also:
        * :class:`MultiDiEdge <vertizee.classes.edge.MultiDiEdge>`
        * :class:`MultiDiGraph <vertizee.classes.graph.MultiDiGraph>`
        * :class:`DiVertex`
    """

    __slots__ = ()

    @abstractmethod
    def adj_vertices(self) -> SetView[MultiDiVertex]:
        """Returns a dynamic, set-like view of all adjacent vertices."""

    @abstractmethod
    def adj_vertices_incoming(self) -> SetView[MultiDiVertex]:
        """Returns a dynamic, set-like view of all adjacent vertices from incoming edges."""

    @abstractmethod
    def adj_vertices_outgoing(self) -> SetView[MultiDiVertex]:
        """Returns a dynamic, set-like view of all adjacent vertices from outgoing edges."""

    @abstractmethod
    def incident_edges(self) -> SetView[MultiDiEdge]:
        """Returns a dynamic, set-like view of all incident edges (incoming, outgoing, and
        self-loops)."""

    @abstractmethod
    def incident_edges_incoming(self) -> SetView[MultiDiEdge]:
        """Returns a dynamic, set-like view of incoming incident edges (i.e. edges where this
        vertex is the head)."""

    @abstractmethod
    def incident_edges_outgoing(self) -> SetView[MultiDiEdge]:
        """Returns a dynamic, set-like view of outgoing incident edges (i.e. edges where this
        vertex is the tail)."""

    @property
    @abstractmethod
    def indegree(self) -> int:
        """The indegree of this vertex, which is the number of incoming incident edges."""

    @property
    @abstractmethod
    def loop_edge(self) -> Optional[MultiDiEdge]:
        """The loop edge if this vertex has a self loop."""

    @property
    @abstractmethod
    def outdegree(self) -> int:
        """The outdegree of this vertex, which is the number of outgoing incident edges."""


class _Vertex(Vertex, VertexBase["Edge"]):
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


class _DiVertex(DiVertex, VertexBase["DiEdge"]):
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


class _MultiVertex(MultiVertex, VertexBase["MultiEdge"]):
    """Concrete implementation of the abstract :class:`MultiVertex` class."""

    __slots__ = ()

    def adj_vertices(self) -> SetView[MultiEdge]:
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


class _MultiDiVertex(MultiDiVertex, VertexBase["MultiDiEdge"]):
    """Concrete implementation of the abstract :class:`MultiDiVertex` class."""

    __slots__ = ()

    def adj_vertices(self) -> SetView[MultiDiEdge]:
        return SetView(self._incident_edges.adj_vertices.copy())

    def adj_vertices_incoming(self) -> SetView[MultiDiEdge]:
        return SetView(self._incident_edges.adj_vertices_incoming)

    def adj_vertices_outgoing(self) -> SetView[MultiDiEdge]:
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
        return sum(e.multiplicity for e in self._incident_edges.incoming)

    @property
    def loop_edge(self) -> Optional[MultiDiEdge]:
        return self._incident_edges.loop_edge

    @property
    def outdegree(self) -> int:
        return sum(e.multiplicity for e in self._incident_edges.outgoing)


class _IncidentEdges(Generic[E]):
    """Collection of edges that are incident on a shared vertex.

    Attempting to add an edge that does not have the shared vertex raises an error. Self loops are
    tracked and may be accessed with the ``loop_edge`` property. Loops cause a vertex to be adjacent
    to itself and the loop is an incident edge. [LOOP2020]_

    In the case of directed edges, incident edges may be also be filtered as ``incoming`` or
    ``outgoing``.

    Args:
        shared_vertex_label: The vertex label of the vertex shared by the incident edges.
        parent_graph: The graph to which the incident edges belong.

    References:
     .. [LOOP2020] Wikipedia contributors. "Loop (graph theory)." Wikipedia, The Free
                   Encyclopedia. Available from: https://en.wikipedia.org/wiki/Loop_(graph_theory).
                   Accessed 29 September 2020.
    """

    __slots__ = ("_incident_edge_labels", "has_loop", "parent_graph", "shared_vertex_label")

    def __init__(self, shared_vertex_label: str, parent_graph: G) -> None:
        self._incident_edge_labels: Set[str] = None
        """The set of edge labels for edges adjacent to the shared vertex."""

        self.has_loop = False
        self.parent_graph = parent_graph

        self.shared_vertex_label: str = shared_vertex_label
        """The label of the vertex common between all of the incident edges."""

    def __eq__(self, other) -> bool:
        if not isinstance(other, _IncidentEdges):
            return False
        if self.shared_vertex_label != other.shared_vertex_label or len(
            self.incident_edges
        ) != len(other.incident_edges):
            return False
        if self.incident_edges != other.incident_edges:
            return False
        return True

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        str_edges = ", ".join(self.incident_edge_labels)
        return f"_IncidentEdges: {str_edges}"

    def add_edge(self, edge: E) -> None:
        """Adds an edge incident on the vertex specified by ``shared_vertex_label``.

        Args:
            edge: The edge to add.
        """
        shared = self.shared_vertex_label
        if shared not in (edge.vertex1.label, edge.vertex2.label):
            raise ValueError(f"cannot add edge {edge} since it is not incident on vertex "
                f"'{shared}'")

        self.incident_edge_labels.add(edge.label)
        if edge.is_loop():
            self.has_loop = True

    @property
    def adj_vertices(self) -> Set[VertexClass]:
        """The set of all vertices adjacent to the shared vertex."""
        vertices = set()
        for label in self.incident_edge_labels:
            comma_pos = label.find(',')
            v1_label = label[1:comma_pos]
            v2_label = label[comma_pos + 1:-1].strip()
            vertices.add(v1_label)
            vertices.add(v2_label)

        if not self.has_loop and self.shared_vertex_label in vertices:
            vertices.remove(self.shared_vertex_label)
        return set(self.parent_graph._vertices[v] for v in vertices)

    @property
    def adj_vertices_incoming(self) -> Set[VertexClass]:
        """The set of all vertices adjacent to the shared vertex from incoming edges. For undirected
        graphs, this is an empty set."""
        if not self.parent_graph.is_directed():
            return set()

        vertices = set()
        for label in self.incident_edge_labels:
            comma_pos = label.find(',')
            v1_label = label[1:comma_pos]
            if v1_label != self.shared_vertex_label:
                vertices.add(v1_label)

        if self.has_loop:
            vertices.add(self.shared_vertex_label)
        return set(self.parent_graph._vertices[v] for v in vertices)

    @property
    def adj_vertices_outgoing(self) -> Set[VertexClass]:
        """Set of all vertices adjacent to the shared vertex from outgoing edges. For undirected
        graphs, this is an empty set."""
        vertices = set()
        for label in self.incident_edge_labels:
            comma_pos = label.find(',')
            v2_label = label[comma_pos + 1:-1].strip()
            if v2_label != self.shared_vertex_label:
                vertices.add(v2_label)

        if self.has_loop:
            vertices.add(self.shared_vertex_label)
        return set(self.parent_graph._vertices[v] for v in vertices)

    @property
    def incident_edge_labels(self) -> Set[str]:
        """Property to handle lazy initialization of incident edge label set."""
        if not self._incident_edge_labels:
            self._incident_edge_labels = set()
        return self._incident_edge_labels

    @property
    def incident_edges(self) -> Set[E]:
        """The set of all incident edges: parallel, self loops, incoming, and outgoing."""
        return set(self.parent_graph._edges[e] for e in self.incident_edge_labels)

    @property
    def incoming(self) -> Set[E]:
        """Edges whose head vertex is ``shared_vertex_label``. For undirected graphs, this is an
        empty set."""
        if not self.parent_graph.is_directed():
            return set()

        edges = set()
        for label in self.incident_edge_labels:
            comma_pos = label.find(',')
            v1_label = label[1:comma_pos]
            if v1_label != self.shared_vertex_label:
                edges.add(label)

        if self.has_loop:
            shared = self.shared_vertex_label
            edges.add(__create_edge_label(shared, shared, self.parent_graph.is_directed()))
        return set(self.parent_graph._edges[e] for e in edges)

    @property
    def loop_edge(self) -> Optional[E]:
        """Loops on ``shared_vertex_label``.

        Since all loops are parallel to each other, only one Edge object is needed.
        """
        if not self.has_loop:
            return None

        return self.parent_graph[self.shared_vertex_label, self.shared_vertex_label]

    @property
    def outgoing(self) -> Set[E]:
        """Edges whose tail vertex is ``shared_vertex_label``. For undirected graphs, this is an
        empty set."""
        edges = set()
        for label in self.incident_edge_labels:
            comma_pos = label.find(',')
            v2_label = label[comma_pos + 1:-1].strip()
            if v2_label != self.shared_vertex_label:
                edges.add(label)

        if self.has_loop:
            shared = self.shared_vertex_label
            edges.add(__create_edge_label(shared, shared, self.parent_graph.is_directed()))
        return set(self.parent_graph._edges[e] for e in edges)

    def remove_edge(self, edge: E) -> None:
        """Removes an incident edge."""
        if edge.label in self.incident_edge_labels:
            self.incident_edge_labels.remove(edge.label)
            if edge.is_loop():
                self.has_loop = False


def __create_edge_label(v1_label: str, v2_label: str, is_directed: bool) -> str:
    """Creates a consistent string representation of an edge.

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
