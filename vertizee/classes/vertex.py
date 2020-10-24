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

"""Classes representing vertices (also called nodes or points) in a graph.

Classes and type aliases:
    * :class:`DiVertex` - A vertex in a directed graph.
    * :class:`Vertex` - A vertex in an undirected graph.
    * :class:`VertexType` - A type alias defined as ``Union[VertexLabel, VertexTupleAttr, Vertex]``.
      Any context that accepts ``VertexType`` permits referring to vertices by label (``str`` or
      ``int``), by object (``Vertex`` or ``DiVertex``), or by tuple
      (``Tuple[VertexLabel, AttributesDict]``) or

Function summary:
    * :func:`get_vertex_label` - Returns the vertex label string for the specified vertex.
    * :func:`is_vertex_type` - Helper function to determine if a variable is a ``VertexType``.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import collections.abc
from typing import Any, Generic, Iterator, Optional, Set, Tuple, TYPE_CHECKING, TypeVar, Union

from vertizee.classes import primitives_parsing
from vertizee.utils import abc_utils

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.edge import DiEdge, Edge, EdgeClass, MultiDiEdge, MultiEdge
    from vertizee.classes.graph import GraphType
    from vertizee.classes.primitives_parsing import GraphPrimitive

# Type aliases
AttributesDict = dict
VertexClass = Union["DiVertex", "Vertex"]
VertexLabel = Union[int, str]
VertexTupleAttr = Tuple[VertexLabel, AttributesDict]
VertexType = Union[VertexClass, VertexLabel, VertexTupleAttr]

# Generic type parameters
E = TypeVar("E", "DiEdge", "Edge", "MultiDiEdge", "MultiEdge")
E_DIRECTED = TypeVar("E_DIRECTED", "DiEdge", "MultiDiEdge")
E_UNDIRECTED = TypeVar("E_UNDIRECTED", "Edge", "MultiEdge")


def get_vertex_label(other: "VertexType") -> str:
    """Returns the vertex label string for the specified vertex."""
    if not is_vertex_type(other):
        raise TypeError(f"VertexType expected; found {type(other).__name__}")

    if isinstance(other, Vertex):
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
    if isinstance(var, (int, str, Vertex)):
        return True
    if isinstance(var, tuple) and len(var) == 2 and isinstance(var[1], collections.abc.Mapping):
        return True
    return False


class _VertexBase(ABC, Generic[E]):
    """Abstract base class from which all vertex classes inherit.

    Args:
        label: The label for this vertex. Must be unique to the graph.
        parent_graph: The parent graph to which this vertex belongs.
    Attributes:
        attr: Attribute dictionary to store ad hoc data associated with the vertex.
    """

    __slots__ = ('_attr', '_incident_edges', '_label', '_parent_graph')

    def __init__(self, label: VertexLabel, parent_graph: GraphType, **attr) -> None:
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
        """Supports index accessor notation to retrieve values from the `attr` dictionary.

        Example:
            >>> import vertizee as vz
            >>> g = vz.Graph()
            >>> g.add_vertex(1)
            1
            >>> g[1]["color"] = "blue"
            >>> g[1]["color"]  # <== calls __getitem__("color")
            'blue'

        Args:
            key: The `attr` dictionary key.

        Returns:
            Any: The value indexed by `key`.
        """
        return self.attr[key]

    def __ge__(self, other: VertexType) -> bool:
        return self.__compare(other, ">=")

    def __gt__(self, other: VertexType) -> bool:
        return self.__compare(other, ">")

    def __hash__(self) -> int:
        return hash(self.label)

    def __iter__(self) -> Iterator[E]:
        return iter(self._incident_edges.incident_edges)

    def __le__(self, other: VertexType) -> bool:
        return self.__compare(other, "<=")

    def __lt__(self, other: VertexType) -> bool:
        return self.__compare(other, "<")

    def __repr__(self) -> str:
        return f"{self.label}"

    def __setitem__(self, key: Any, value: Any) -> None:
        """Supports index accessor notation to set values in the `attr` dictionary.

        Example:
            >>> import vertizee as vz
            >>> g = vz.Graph()
            >>> g.add_vertex(1)
            1
            >>> g[1]["color"] = "blue"  # <== calls __setitem__("color", "blue")
            >>> g[1]["color"]
            'blue'

        Args:
            key: The `attr` dictionary key.
            value: The value to assign to `key` in the `attr` dictionary.
        """
        self.attr[key] = value

    def __str__(self) -> str:
        return f"{self.label}"

    @classmethod
    def __subclasshook__(cls, C):
        if cls is _VertexBase:
            return abc_utils.check_methods(C, "__compare", "__eq__", "__getitem__", "__ge__",
                "__gt__", "__hash__", "__iter__", "__le__", "__lt__", "__repr__", "__setitem__",
                "__str__", "adj_vertices", "attr", "degree", "remove_loops", "incident_edges",
                "is_incident_edge", "label", "loop_edge")
        return NotImplemented

    @property
    @abstractmethod
    def adj_vertices(self) -> Set[_VertexBase]:
        """The set of all adjacent vertices."""

    @property
    def attr(self) -> dict:
        """Attribute dictionary to store ad hoc data associated with a vertex."""
        if not self._attr:
            self._attr = dict()
        return self._attr

    @property
    @abstractmethod
    def degree(self) -> int:
        """The degree (or valance) of this vertex.

        The degree is the number of incident edges. Self-loops are counted twice.
        """

    @abstractmethod
    def get_adj_for_search(
        self, parent: Optional[VertexClass] = None, reverse_graph: bool = False
    ) -> Set[_VertexBase]:
        """Method designed for search algorithms to retrieve the correct list of adjacent vertices
        depending on the type of graph.

        For undirected graphs, the reachable vertices are all adjacent vertices. However, if this
        vertex had a parent (i.e. predecessor) vertex in the search, then the parent should be
        excluded from the adjacency list.

        Args:
            parent: Optional; The parent vertex (i.e. predecessor) in the search tree.
                Defaults to None.
            reverse_graph: Optional; For directed graphs, setting to True will yield a
                traversal as if the graph were reversed (i.e. the reverse/transpose/converse
                graph). Defaults to False.
        """

    @property
    def incident_edges(self) -> Set[E]:
        """The set of all incident edges (incoming, outgoing, and self-loops)."""
        return self._incident_edges.incident_edges

    def is_incident_edge(self, vertex_or_edge: GraphPrimitive) -> bool:
        """Returns True if the edge specified by ``vertex_or_edge`` is incident on this vertex.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        return self._incident_edges.get_edge(vertex_or_edge) is not None

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
    def loop_edge(self) -> Optional[E]:
        """The loop edge (or edges) if this vertex has one or more self loops.

        Since self-loops are parallel, multiple loops are stored in one :class:`MultiEdge
        <vertizee.classes.edge.MultiEdge>` or one :class:`MultiDiEdge
        <vertizee.classes.edge.MultiDiEdge>` object.

        Returns:
            Optional[E]: The loop edge or None if there are no self loops.
        """
        return self._incident_edges.loop_edge

    def remove_loops(self) -> int:
        """Removes all edges that are loops on this vertex.

        Returns:
            int: The number of loops deleted.
        """
        loops = self._incident_edges.loop_edge
        deletion_count = loops._parallel_edge_count + 1
        self._parent_graph._edges.remove(loops)
        self._parent_graph._edges_with_freq_weight.pop(loops)
        self._incident_edges.remove_edge(loops)
        return deletion_count


class Vertex(_VertexBase[E_UNDIRECTED], Generic[E_UNDIRECTED]):
    """A graph primitive representing a point (also called a node) that may be connected to other
    vertices via undirected edges.

    No two vertices within a graph may share the same label. Labels may be strings or integers, but
    internally, they are always stored as strings. The following statements are equivalent::

        graph.add_vertex(1)
        graph.add_vertex("1")

    Any context that accepts a ``VertexType`` (defined as
    ``Union[Vertex, VertexLabel, VertexTupleAttr]``) permits specifying vertices using labels (i.e.
    strings or integers), ``Vertex`` objects, or a tuple of the form
    ``Tuple[VertexLabel, AttributesDict]``. For example, the following
    statements are equivalent and return the :class:`Vertex` object with label "1"::

        obj_one: Vertex = graph.add_vertex(1)
        graph[obj_one]
        graph[1]
        graph["1"]

    To help ensure the integrity of graphs, the ``Vertex`` class is abstract and cannot be
    instantiated directly. To create vertices, use :meth:`Graph.add_vertex
    <vertizee.classes.graph.Graph.add_vertex>` and :meth:`Graph.add_vertices_from
    <vertizee.classes.graph.Graph.add_vertices_from>`.

    Attributes:
        attr: Attribute dictionary to store ad hoc data associated with the vertex.

    See Also:
        * :class:`DiVertex`
    """

    __slots__ = ()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Vertex:
            return abc_utils.check_methods(C, "__compare", "__eq__", "__getitem__", "__ge__",
                "__gt__", "__hash__", "__iter__", "__le__", "__lt__", "__repr__", "__setitem__",
                "__str__", "adj_vertices", "attr", "degree", "remove_loops", "get_adj_for_search",
                "incident_edges", "is_incident_edge", "label", "loop_edge")
        return NotImplemented

    @property
    def adj_vertices(self) -> Set[Vertex]:
        """The set of all adjacent vertices."""
        return self._incident_edges.adj_vertices.copy()

    @property
    @abstractmethod
    def degree(self) -> int:
        """The degree (or valance) of this vertex.

        The degree is the number of incident edges. Self-loops are counted twice.
        """

    @abstractmethod
    def get_adj_for_search(
        self, parent: Optional[Vertex] = None, reverse_graph: bool = False
    ) -> Set[Vertex]:
        """Method designed for search algorithms to retrieve the correct list of adjacent vertices
        depending on the type of graph.

        For undirected graphs, the reachable vertices are all adjacent vertices. However, if this
        vertex had a parent (i.e. predecessor) vertex in the search, then the parent should be
        excluded from the adjacency list.

        Args:
            parent: Optional; The parent vertex (i.e. predecessor) in the search tree.
                Defaults to None.
            reverse_graph: Optional; For directed graphs, setting to True will yield a
                traversal as if the graph were reversed (i.e. the reverse/transpose/converse
                graph). Defaults to False.
        """


class DiVertex(_VertexBase[E_DIRECTED], Generic[E_DIRECTED]):
    """A graph primitive representing a point (also called a node) in a directed graph that may be
    connected to other vertices via directed edges.

    Attributes:
        attr: Attribute dictionary to store ad hoc data associated with the vertex.

    See Also:
        * :class:`Vertex`
    """

    __slots__ = ()

    @classmethod
    def __subclasshook__(cls, C):
        if cls is DiVertex:
            return abc_utils.check_methods(C, "__compare", "__eq__", "__getitem__", "__ge__",
                "__gt__", "__hash__", "__iter__", "__le__", "__lt__", "__repr__", "__setitem__",
                "__str__", "adj_vertices", "adj_vertices_incoming", "adj_vertices_outgoing", "attr",
                "degree", "remove_loops", "get_adj_for_search", "incident_edges",
                "incident_edges_incoming", "incident_edges_outgoing", "indegree",
                "is_incident_edge", "label", "loop_edge", "outdegree")
        return NotImplemented

    @property
    def adj_vertices(self) -> Set[DiVertex]:
        """The set of all adjacent vertices."""
        return self._incident_edges.adj_vertices.copy()

    @property
    def adj_vertices_incoming(self) -> Set[DiVertex]:
        """The set of all adjacent vertices from incoming edges. This is an empty set for
        undirected graphs."""
        return self._incident_edges.adj_vertices_incoming.copy()

    @property
    def adj_vertices_outgoing(self) -> Set[DiVertex]:
        """The set of all adjacent vertices from outgoing edges. This is an empty set for
        undirected graphs."""
        return self._incident_edges.adj_vertices_outgoing.copy()

    @property
    @abstractmethod
    def degree(self) -> int:
        """The degree (or valance) of this vertex.

        The degree is the number of incident edges. Self-loops are counted twice.
        """

    @abstractmethod
    def get_adj_for_search(
        self, parent: Optional[DiVertex] = None, reverse_graph: bool = False
    ) -> Set[DiVertex]:
        """Method designed for search algorithms to retrieve the correct list of adjacent vertices
        based on the graph type.

        For directed graphs, the reachable vertices in a search are the adjacent vertices of the
        outgoing edges. However, if the algorithm is performing a search on the reverse (or
        transpose) of the directed graph, then the reachable adjacent vertices are on the incoming
        edges.

        For undirected graphs, the reachable vertices are all adjacent vertices. However, if this
        vertex had a parent (i.e. predecessor) vertex in the search, then the parent should be
        excluded from the adjacency list.

        Args:
            parent: Optional; The parent vertex (i.e. predecessor) in the search tree.
                Defaults to None.
            reverse_graph: Optional; For directed graphs, setting to True will yield a
                traversal as if the graph were reversed (i.e. the reverse/transpose/converse
                graph). Defaults to False.
        """

    @property
    def incident_edges_incoming(self) -> Set[E_DIRECTED]:
        """The set of incoming incident edges (i.e. edges where this vertex is the head).

        This is an empty set for undirected graphs. Use ``incident_edges`` instead.

        Returns:
            Set[E_DIRECTED]: The incoming edges.
        """
        return self._incident_edges.incoming.copy()

    @property
    def incident_edges_outgoing(self) -> Set[E_DIRECTED]:
        """The set of outgoing incident edges (i.e. edges where this vertex is the tail).

        This is an empty set for undirected graphs. Use ``incident_edges`` instead.

        Returns:
            Set[E_DIRECTED]: The outgoing edges.
        """
        return self._incident_edges.outgoing.copy()

    @property
    def indegree(self) -> int:
        """The indegree of this vertex.

        The indegree is the number of incoming incident edges. For undirected graphs, the indegree
        is the same as the degree.
        """
        if not self._parent_graph.is_directed_graph():
            return self.degree

        total = 0
        for edge in self._incident_edges.incoming:
            total += edge.multiplicity
        return total

    @property
    def outdegree(self) -> int:
        """The outdegree of this vertex.

        The outdegree is the number of outgoing incident edges. For undirected graphs, the outdegree
        is the same as the degree.
        """
        if not self._parent_graph.is_directed_graph():
            return self.degree

        total = 0
        for edge in self._incident_edges.outgoing:
            total += edge.multiplicity
        return total


class _Vertex(Vertex, Generic[E_UNDIRECTED]):
    """Concrete implementation of the abstract :class:`Vertex` class."""

    __slots__ = ()

    @property
    def degree(self) -> int:
        """The degree (or valance) of this vertex.

        The degree is the number of incident edges. Self-loops are counted twice.
        """
        total = 0
        for e in self.incident_edges:
            if e.is_loop():
                total += 2 * (1 + e._parallel_edge_count)
            else:
                total += (1 + e._parallel_edge_count)
        return total

    def get_adj_for_search(
        self, parent: Optional[Vertex] = None, reverse_graph: bool = False
    ) -> Set[Vertex]:
        """Method designed for search algorithms to retrieve the correct list of adjacent vertices
        based on the graph type.

        For directed graphs, the reachable vertices in a search are the adjacent vertices of the
        outgoing edges. However, if the algorithm is performing a search on the reverse (or
        transpose) of the directed graph, then the reachable adjacent vertices are on the incoming
        edges.

        For undirected graphs, the reachable vertices are all adjacent vertices. However, if this
        vertex had a parent (i.e. predecessor) vertex in the search, then the parent should be
        excluded from the adjacency list.

        Args:
            parent: Optional; The parent vertex (i.e. predecessor) in the search tree.
                Defaults to None.
            reverse_graph: Optional; For directed graphs, setting to True will yield a
                traversal as if the graph were reversed (i.e. the reverse/transpose/converse
                graph). Defaults to False.
        """
        adj_vertices = self.adj_vertices
        if parent is not None:
            adj_vertices = adj_vertices - {parent}
        return adj_vertices

    def _add_edge(self, edge: E_UNDIRECTED) -> None:
        """Adds an edge.

        If an incident edge already exists with the same vertices, it is overwritten.

        Raises:
            ValueError: If the new edge does not include this vertex.
        """
        if edge.vertex1.label != self.label and edge.vertex2.label != self.label:
            raise ValueError(
                f"edge {edge} did not have a vertex matching this vertex {self}"
            )
        self._incident_edges.add_edge(edge)

    def _get_edge(self, vertex_or_edge: GraphPrimitive) -> Optional[E_UNDIRECTED]:
        """Gets the incident edge specified by ``vertex_or_edge``, or None if no such edge exists.

        Args:
            vertex_or_edge: For undirected graphs, the adjacent vertex (relative to this vertex)
                will suffice, but may also be an edge. For directed graphs, an edge must be
                specified that includes this vertex and the adjacent vertex, since the order of the
                vertices distinguishes between incoming and outgoing edges.

        Returns:
            Optional[E_UNDIRECTED]: The specified edge or None if no such edge exists in the graph.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        return self._incident_edges.get_edge(vertex_or_edge)

    def _remove_edge(self, edge: E_UNDIRECTED) -> int:
        """Removes an incident edge.

        Args:
            edge: The edge to remove.

        Returns:
            int: Number of edges removed (more than one for parallel edges).
        """
        if edge.vertex1.label != self.label and edge.vertex2.label != self.label:
            raise ValueError(
                f"edge {edge} did not have a vertex matching this vertex {self}"
            )
        self._incident_edges.remove_edge(edge)
        return 1 + edge.parallel_edge_count


class _DiVertex(DiVertex, Generic[E_DIRECTED]):
    """Concrete implementation of the abstract :class:`DiVertex` class."""

    __slots__ = ()

    @property
    def degree(self) -> int:
        """The degree (or valance) of this vertex.

        The degree is the number of incident edges. Self-loops are counted twice.
        """
        total = 0
        for e in self.incident_edges:
            if e.is_loop():
                total += 2 * (1 + e._parallel_edge_count)
            else:
                total += (1 + e._parallel_edge_count)
        return total

    def get_adj_for_search(
        self, parent: Optional[DiVertex] = None, reverse_graph: bool = False
    ) -> Set[DiVertex]:
        """Method designed for search algorithms to retrieve the correct list of adjacent vertices
        based on the graph type.

        For directed graphs, the reachable vertices in a search are the adjacent vertices of the
        outgoing edges. However, if the algorithm is performing a search on the reverse (or
        transpose) of the directed graph, then the reachable adjacent vertices are on the incoming
        edges.

        For undirected graphs, the reachable vertices are all adjacent vertices. However, if this
        vertex had a parent (i.e. predecessor) vertex in the search, then the parent should be
        excluded from the adjacency list.

        Args:
            parent: Optional; The parent vertex (i.e. predecessor) in the search tree.
                Defaults to None.
            reverse_graph: Optional; For directed graphs, setting to True will yield a
                traversal as if the graph were reversed (i.e. the reverse/transpose/converse
                graph). Defaults to False.
        """
        if reverse_graph:
            return self.adj_vertices_incoming
        return self.adj_vertices_outgoing

    def _add_edge(self, edge: E_DIRECTED) -> None:
        """Adds an edge.

        If an incident edge already exists with the same vertices, it is overwritten.

        Args:
            edge: The edge to be added.

        Raises:
            ValueError: If the new edge does not include this vertex.
        """
        if edge.vertex1.label != self.label and edge.vertex2.label != self.label:
            raise ValueError(
                f"Edge {edge} did not have a vertex matching this vertex {{{self.label}}}"
            )
        self._incident_edges.add_edge(edge)

    def _get_edge(self, vertex_or_edge: GraphPrimitive) -> Optional[E_DIRECTED]:
        """Gets the incident edge specified by ``vertex_or_edge``, or None if no such edge exists.

        Args:
            vertex_or_edge: For undirected graphs, the adjacent vertex (relative to this vertex)
                will suffice, but may also be an edge. For directed graphs, an edge must be
                specified that includes this vertex and the adjacent vertex, since the order of the
                vertices distinguishes between incoming and outgoing edges.

        Returns:
            Optional[E_DIRECTED]: The specified edge or None if no such edge exists in the graph.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        return self._incident_edges.get_edge(vertex_or_edge)

    def _remove_edge(self, edge: E_DIRECTED) -> int:
        """Removes an incident edge.

        Args:
            edge: The edge to be removed.

        Returns:
            int: Number of edges removed (more than one for parallel edges).
        """
        if edge.vertex1.label != self.label and edge.vertex2.label != self.label:
            raise ValueError(
                f"edge {edge} did not have a vertex matching this vertex {self}"
            )
        self._incident_edges.remove_edge(edge)
        return 1 + edge.parallel_edge_count


class _IncidentEdges(Generic[E]):
    """Collection of edges that are incident on a shared vertex.

    Attempting to add an edge that does not have the shared vertex raises an error. Self loops are
    tracked and may be accessed with the ``loop_edge`` property. Loops cause a vertex to be adjacent
    to itself and the loop is an incident edge. [LOOP2020]_ In the case of directed edges, incident
    edges are also classified as ``incoming`` and ``outgoing``. Collections of adjacent vertices are
    also maintained for algorithmic efficiency.

    Args:
        shared_vertex_label: The vertex label of the vertex shared by the incident edges.
        parent_graph: The graph to which the incident edges belong.

    References:
     .. [LOOP2020] Wikipedia contributors. "Loop (graph theory)." Wikipedia, The Free
                   Encyclopedia. Available from: https://en.wikipedia.org/wiki/Loop_(graph_theory).
                   Accessed 29 September 2020.
    """

    __slots__ = ("_incident_edge_labels", "has_loop", "parent_graph", "shared_vertex_label")

    def __init__(self, shared_vertex_label: str, parent_graph: GraphType) -> None:
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
        return f"_IncidentEdges: {{{str_edges}}}"

    @property
    def incident_edge_labels(self) -> Set[str]:
        """Property to handle lazy initialization of incident edge label set."""
        if not self._incident_edge_labels:
            self._incident_edge_labels = set()
        return self._incident_edge_labels

    def add_edge(self, edge: E) -> None:
        """Adds an edge incident to the vertex specified by ``shared_vertex_label``.

        Args:
            edge: The edge to add.
        """
        shared = self.shared_vertex_label
        if shared not in (edge.vertex1.label, edge.vertex2.label):
            raise ValueError(f"cannot add edge {edge} since it does not share vertex {{{shared}}}.")

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
        if not self.parent_graph.is_directed_graph():
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
    def incident_edges(self) -> Set[E]:
        """The set of all incident edges: parallel, self loops, incoming, and outgoing."""
        return set(self.parent_graph._edges[e] for e in self.incident_edge_labels)

    @property
    def incoming(self) -> Set[E]:
        """Edges whose head vertex is ``shared_vertex_label``. For undirected graphs, this is an
        empty set."""
        if not self.parent_graph.is_directed_graph():
            return set()

        edges = set()
        for label in self.incident_edge_labels:
            comma_pos = label.find(',')
            v1_label = label[1:comma_pos]
            if v1_label != self.shared_vertex_label:
                edges.add(label)

        if self.has_loop:
            shared = self.shared_vertex_label
            edges.add(self.parent_graph[shared, shared].label)
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
            edges.add(self.parent_graph[shared, shared].label)
        return set(self.parent_graph._edges[e] for e in edges)

    def get_edge(self, vertex_or_edge: GraphPrimitive) -> Optional[E]:
        """Gets the incident edge specified by ``vertex_or_edge``, or None if no such edge exists.

        Args:
            vertex_or_edge: For undirected graphs, the adjacent vertex (relative to this vertex)
                will suffice, but an edge may be specified as well. For directed graphs, an edge
                must be provided that includes this vertex and the adjacent vertex, since the vertex
                order distinguishes between incoming and outgoing edges.

        Returns:
            Optional[E]: The specified edge or None if no such edge exists in the graph.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        primitives = primitives_parsing.parse_graph_primitive(vertex_or_edge)

        if primitives.vertices:
            if self.parent_graph.is_directed_graph():
                raise ValueError("directed graphs must specify both endpoint vertices to get edge")
            if (self.shared_vertex_label, primitives.vertices[0].get_label()) in self.parent_graph:
                return self.parent_graph[self.shared_vertex_label, primitives.vertices[0].label]
            return None

        edge_label = primitives.edges[0].get_label(self.parent_graph.is_directed_graph())

        if edge_label in self.incident_edge_labels:
            return self.parent_graph._edges[edge_label]
        return None

    def remove_edge(self, edge: E) -> None:
        """Removes an incident edge."""
        if edge.label in self.incident_edge_labels:
            self.incident_edge_labels.remove(edge.label)
            if edge.is_loop():
                self.has_loop = False
