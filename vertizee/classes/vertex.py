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
    * :class:`VertexType` - A type alias defined as ``Union[VertexLabel, VertexLabelAttr, Vertex]``.
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
from typing import Any, Dict, Iterator, Optional, Set, Tuple, TYPE_CHECKING, Union

from vertizee.classes.primitives_parsing import parse_graph_primitive
from vertizee.utils import abc_utils

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.edge import DiEdge, Edge, MultiDiEdge, MultiEdge
    from vertizee.classes.graph_base import GraphBase
    from vertizee.classes.primitives_parsing import GraphPrimitive

# Type aliases
AttributesDict = dict
VertexLabel = Union[int, str]
VertexLabelAttr = Tuple[VertexLabel, AttributesDict]
VertexType = Union["Vertex", VertexLabel, VertexLabelAttr]


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

    The ``VertexType`` type alias is defined as ``Union[VertexLabel, VertexLabelAttr, "Vertex"]``
    and ``VertexLabelAttr`` is defined as ``Tuple[VertexLabel, AttributesDict]``. A vertex label may
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


class Vertex(ABC):
    """A graph primitive representing a point (also called a node) that may be connected to other
    vertices via edges.

    No two vertices within a graph may share the same label. Labels may be strings or integers, but
    internally, they are always stored as strings. The following statements are equivalent::

        graph.add_vertex(1)
        graph.add_vertex("1")

    Any context that accepts a ``VertexType`` (defined as
    ``Union[Vertex, VertexLabel, VertexLabelAttr]``) permits specifying vertices using labels (i.e.
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
    def __init__(self, label: VertexLabel, parent_graph: GraphBase, **attr) -> None:
        self._label = str(label)

        self._attr: dict = {}
        for k, v in attr.items():
            self._attr[k] = v

        self._incident_edges: _IncidentEdges = _IncidentEdges(self.label, parent_graph)
        self._parent_graph = parent_graph

    def __compare(self, other: VertexType, operator: str) -> bool:
        if not is_vertex_type(other):
            return False
        other_label = get_vertex_label(other)
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
            >>> g[1]["color"]  # <== calls __getitem__()
            'blue'

        Args:
            key: The `attr` dictionary key.

        Returns:
            Any: The value indexed by `key`.
        """
        return self._attr[key]

    def __ge__(self, other: VertexType) -> bool:
        return self.__compare(other, ">=")

    def __gt__(self, other: VertexType) -> bool:
        return self.__compare(other, ">")

    def __hash__(self) -> int:
        return hash(self.label)

    def __iter__(self) -> Iterator[Edge]:
        return self._incident_edges.__iter__()

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
            >>> g[1]["color"] = "blue"  # <== calls __setitem__()
            >>> g[1]["color"]
            'blue'

        Args:
            key: The `attr` dictionary key.
            value: The value to assign to `key` in the `attr` dictionary.
        """
        self._attr[key] = value

    def __str__(self) -> str:
        return f"{self.label}"

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Vertex:
            return abc_utils.check_methods(C, "__compare", "__eq__", "__getitem__", "__ge__",
                "__gt__", "__hash__", "__iter__", "__le__", "__lt__", "__repr__", "__setitem__",
                "__str__", "adj_vertices", "attr", "degree", "delete_loops", "get_adj_for_search",
                "incident_edges", "is_incident_edge", "label", "loop_edge")
        return NotImplemented

    @property
    def adj_vertices(self) -> Set["Vertex"]:
        """The set of all adjacent vertices."""
        return self._incident_edges._adj_vertices.copy()

    @property
    def attr(self) -> dict:
        """Attribute dictionary to store ad hoc data associated with a vertex."""
        return self._attr

    @property
    @abstractmethod
    def degree(self) -> int:
        """The degree (or valance) of this vertex.

        The degree is the number of incident edges. Self-loops are counted twice.
        """

    @abstractmethod
    def delete_loops(self) -> int:
        """Deletes all edges that are loops on this vertex.

        Returns:
            int: The number of loops deleted.
        """

    @abstractmethod
    def get_adj_for_search(
        self, parent: Optional["Vertex"] = None, reverse_graph: bool = False
    ) -> Set["Vertex"]:
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
    def incident_edges(self) -> Set[Edge]:
        """The set of all incident edges (incoming, outgoing, and self-loops)."""
        return self._incident_edges.incident_edges

    def is_incident_edge(self, vertex_or_edge: GraphPrimitive) -> bool:
        """Returns True if the edge specified by ``vertex_or_edge`` is incident on this vertex.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        return self._incident_edges.get_edge(vertex_or_edge) is not None

    @property
    def label(self) -> str:
        """The vertex label string."""
        return self._label

    @property
    def loop_edge(self) -> Optional[Union[Edge, MultiEdge]]:
        """The loop edge (or multi-edge) if there are one or more loops, otherwise None.

        Note:
            Since a single :class:`Edge <vertizee.classes.edge.Edge>` object represents all
            parallel edges between two vertices [or in the case of a directed graph, two
            :class:`DiEdge <vertizee.classes.edge.DiEdge>` objects, one for edge :math:`(a, b)`
            and a second for :math:`(b, a)`], loops will always contain exactly zero or
            one edge object; and the edge object could be a multi-edge.

        Returns:
            Optional[Edge]: The loop edge (or multi-edge), if there are one or more loops.
        """
        return self._incident_edges.loop_edge


class DiVertex(Vertex):
    """A graph primitive representing a point (also called a node) in a directed graph that may be
    connected to other vertices via directed edges.

    Attributes:
        attr: Attribute dictionary to store ad hoc data associated with the vertex.

    See Also:
        * :class:`Vertex`
    """
    @classmethod
    def __subclasshook__(cls, C):
        if cls is Vertex:
            return abc_utils.check_methods(C, "__compare", "__eq__", "__getitem__", "__ge__",
                "__gt__", "__hash__", "__iter__", "__le__", "__lt__", "__repr__", "__setitem__",
                "__str__", "adj_vertices", "adj_vertices_incoming", "adj_vertices_outgoing", "attr",
                "degree", "delete_loops", "get_adj_for_search", "incident_edges",
                "incident_edges_incoming", "incident_edges_outgoing", "indegree",
                "is_incident_edge", "label", "loop_edge", "outdegree")
        return NotImplemented

    @property
    def adj_vertices_incoming(self) -> Set["Vertex"]:
        """The set of all adjacent vertices from incoming edges. This is an empty set for
        undirected graphs."""
        return self._incident_edges._adj_vertices_incoming.copy()

    @property
    def adj_vertices_outgoing(self) -> Set["Vertex"]:
        """The set of all adjacent vertices from outgoing edges. This is an empty set for
        undirected graphs."""
        return self._incident_edges._adj_vertices_outgoing.copy()

    @property
    @abstractmethod
    def degree(self) -> int:
        """The degree (or valance) of this vertex.

        The degree is the number of incident edges. Self-loops are counted twice.
        """

    @abstractmethod
    def delete_loops(self) -> int:
        """Deletes all edges that are loops on this vertex.

        Returns:
            int: The number of loops deleted.
        """

    @abstractmethod
    def get_adj_for_search(
        self, parent: Optional["Vertex"] = None, reverse_graph: bool = False
    ) -> Set["Vertex"]:
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
    def incident_edges_incoming(self) -> Set[Edge]:
        """The set of incoming incident edges (i.e. edges where this vertex is the head).

        This is an empty set for undirected graphs. Use ``incident_edges`` instead.

        Returns:
            Set[Edge]: The incoming edges.
        """
        return self._incident_edges.incoming

    @property
    def incident_edges_outgoing(self) -> Set[Edge]:
        """The set of outgoing incident edges (i.e. edges where this vertex is the tail).

        This is an empty set for undirected graphs. Use ``incident_edges`` instead.

        Returns:
            Set[Edge]: The outgoing edges.
        """
        return self._incident_edges.outgoing

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


class _IncidentEdges:
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

    def __init__(self, shared_vertex_label: str, parent_graph: GraphBase) -> None:
        self._parent_graph = parent_graph

        self._adj_vertices: Set["Vertex"] = set()
        """The set of all vertices adjacent to the shared vertex."""

        self._adj_vertices_incoming: Set["Vertex"] = set()
        """Directed graphs only: the set of all vertices adjacent to the shared vertex from incoming
        edges."""

        self._adj_vertices_outgoing: Set["Vertex"] = set()
        """Directed graphs only: the set of all vertices adjacent to the shared vertex from outgoing
        edges."""

        self._incident_edges: Dict[str, Edge] = {}
        """The dictionary of all incident edges: parallel, self loops, incoming, and outgoing.

        The dictionary keys are created by :func:`create_edge_label
        <vertizee.classes.edge.create_edge_label>`, and are a mapping from edge vertices to a
        consistent string representation.
        """

        self._incoming: Set[Edge] = set()
        """Directed graphs only: edges whose head vertex is ``_shared_vertex_label``."""

        self._loop_edge: Optional[Edge] = None
        """Loops on ``_shared_vertex``.

        Since all loops are parallel to each other, only one Edge object is needed.
        """

        self._outgoing: Set[Edge] = set()
        """Directed graphs only: edges whose tail vertex is ``_shared_vertex``."""

        self._shared_vertex_label: str = shared_vertex_label
        """The label of the vertex common between all of the incident edges."""

    def __eq__(self, other) -> bool:
        if not isinstance(other, _IncidentEdges):
            return False
        if self._shared_vertex_label != other._shared_vertex_label or len(
            self._incident_edges
        ) != len(other._incident_edges):
            return False
        if self._incident_edges != other._incident_edges:
            return False
        return True

    def __iter__(self) -> Iterator[Edge]:
        return iter(self._incident_edges.values())

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        str_edges = ", ".join(self._incident_edges.keys())
        return f"_IncidentEdges: {{{str_edges}}}"

    def add_edge(self, edge: Edge) -> None:
        """Adds an edge incident to the vertex specified by ``shared_vertex_label``.

        If an existing edge has the same vertices, it is overwritten.

        Args:
            edge (Edge): The edge to add.
        """
        # Local import to avoid circular reference.
        # pylint: disable=import-outside-toplevel
        from vertizee.classes import edge

        if (
            edge.vertex1.label != self._shared_vertex_label
            and edge.vertex2.label != self._shared_vertex_label
        ):
            raise ValueError(
                f"Cannot add edge {edge} since it does not share vertex "
                f"{{{self._shared_vertex_label}}}."
            )

        edge_label = edge.create_edge_label(
            edge.vertex1.label, edge.vertex2.label, self._parent_graph.is_directed_graph()
        )
        if edge.vertex1 == edge.vertex2:
            self._loop_edge = edge
            self._incident_edges[edge_label] = edge
            self._adj_vertices.add(edge.vertex1)
            if self._parent_graph.is_directed_graph():
                self._outgoing.add(edge)
                self._adj_vertices_outgoing.add(edge.vertex1)
                self._incoming.add(edge)
                self._adj_vertices_incoming.add(edge.vertex1)
            return

        adj_vertex = None
        is_outgoing_edge = True
        if edge.vertex1.label != self._shared_vertex_label:
            is_outgoing_edge = False
            adj_vertex = edge.vertex1
        else:
            adj_vertex = edge.vertex2

        self._incident_edges[edge_label] = edge
        self._adj_vertices.add(adj_vertex)

        if self._parent_graph.is_directed_graph():
            if is_outgoing_edge:
                self._outgoing.add(edge)
                self._adj_vertices_outgoing.add(adj_vertex)
            else:
                self._incoming.add(edge)
                self._adj_vertices_incoming.add(adj_vertex)

    @property
    def adj_vertices(self) -> Set["Vertex"]:
        """The set of all vertices adjacent to the shared vertex."""
        return self._adj_vertices.copy()

    @property
    def adj_vertices_incoming(self) -> Set["Vertex"]:
        """The set of all vertices adjacent to the shared vertex from incoming edges. This property
        is only defined for directed graphs."""
        return self._adj_vertices_incoming.copy()

    @property
    def adj_vertices_outgoing(self) -> Set["Vertex"]:
        """The set of all vertices adjacent to the shared vertex from outgoing edges. This property
        is only defined for directed graphs."""
        return self._adj_vertices_outgoing.copy()

    @property
    def incident_edges(self) -> Set[Edge]:
        """The set of all incident edges."""
        return set(self._incident_edges.values())

    def get_edge(self, vertex_or_edge: GraphPrimitive) -> Optional[Edge]:
        """Gets the incident edge specified by ``vertex_or_edge``, or None if no such edge exists.

        Args:
            vertex_or_edge: For undirected graphs, the adjacent vertex (relative to this vertex)
                will suffice, but may also be an edge. For directed graphs, an edge must be
                specified that includes this vertex and the adjacent vertex, since the order of the
                vertices distinguishes between incoming and outgoing edges.

        Returns:
            Edge: The specified edge or None if no such edge exists in the graph.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        # Local import to avoid circular reference.
        # pylint: disable=import-outside-toplevel
        from vertizee.classes import edge

        primitives = parse_graph_primitive(vertex_or_edge)

        if len(primitives.edges) == 0:
            if self._parent_graph.is_directed_graph():
                raise ValueError("directed graphs must specify both endpoint vertices to get edge")
            edge_label = edge.create_label(
                self._shared_vertex_label,
                primitives.vertices[0].label,
                is_directed=self._parent_graph.is_directed_graph()
            )
        else:
            edge_label = edge.create_label(
                primitives.edges[0].vertex1.label,
                primitives.edges[0].vertex2.label,
                is_directed=self._parent_graph.is_directed_graph()
            )

        if edge_label in self._incident_edges:
            return self._incident_edges[edge_label]
        return None

    @property
    def incoming(self) -> Set[Edge]:
        """The set of all incoming edges. This property is only defined for directed graphs."""
        return self._incoming.copy()

    @property
    def incoming_edge_vertices(self) -> Set["Vertex"]:
        """The set of all incoming edge vertices. This property is only defined for directed
        graphs."""
        vertices = set()
        for edge in self._incoming:
            vertices.add(edge.vertex1)
        return vertices

    @property
    def loop_edge(self) -> Optional[Edge]:
        """The loop edge (or multi-edge) if there are one or more loops, otherwise None."""
        return self._loop_edge

    @property
    def outgoing(self) -> Set[Edge]:
        """The set of all outgoing edges. This property is only defined for directed graphs."""
        return self._outgoing.copy()

    def remove_edge_from(self, edge: Edge) -> None:
        """Removes an edge."""
        # Local import to avoid circular reference.
        # pylint: disable=import-outside-toplevel
        from vertizee.classes import edge as edge_module

        is_directed = self._parent_graph.is_directed_graph()
        if edge.label in self._incident_edges:
            self._incident_edges.pop(edge.label)

        if self._loop_edge == edge:
            self._loop_edge = None
        if edge in self._outgoing:
            self._outgoing.remove(edge)
        if edge in self._incoming:
            self._incoming.remove(edge)

        if edge.vertex1.label == self._shared_vertex_label:
            other_vertex = edge.vertex2
            outgoing_edge = True
        else:
            other_vertex = edge.vertex1
            outgoing_edge = False

        if is_directed:
            reverse_edge_label = edge_module.create_label(
                edge.vertex2.label, edge.vertex1.label, is_directed=is_directed
            )
            if reverse_edge_label not in self._incident_edges:
                if other_vertex in self._adj_vertices:
                    self._adj_vertices.remove(other_vertex)

            if outgoing_edge and other_vertex in self._adj_vertices_outgoing:
                self._adj_vertices_outgoing.remove(other_vertex)
            elif other_vertex in self._adj_vertices_incoming:
                self._adj_vertices_incoming.remove(other_vertex)
        else:  # undirected graph
            if other_vertex in self._adj_vertices:
                self._adj_vertices.remove(other_vertex)


class _Vertex(Vertex):
    """Concrete implementation of the abstract :class:`Vertex` class."""

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

    def delete_loops(self) -> int:
        """Deletes all edges that are loops on this vertex.

        Returns:
            int: The number of loops deleted.
        """
        loops = self._incident_edges.loop_edge
        deletion_count = loops._parallel_edge_count + 1
        self._parent_graph._edges.remove(loops)
        self._parent_graph._edges_with_freq_weight.pop(loops)
        self._incident_edges.remove_edge_from(loops)
        return deletion_count

    def get_adj_for_search(
        self, parent: Optional["Vertex"] = None, reverse_graph: bool = False
    ) -> Set["Vertex"]:
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

    def _add_edge(self, edge: Edge) -> None:
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

    def _get_edge(self, vertex_or_edge: GraphPrimitive) -> Optional[Edge]:
        """Gets the incident edge specified by ``vertex_or_edge``, or None if no such edge exists.

        Args:
            vertex_or_edge: For undirected graphs, the adjacent vertex (relative to this vertex)
                will suffice, but may also be an edge. For directed graphs, an edge must be
                specified that includes this vertex and the adjacent vertex, since the order of the
                vertices distinguishes between incoming and outgoing edges.

        Returns:
            Edge: The specified edge or None if no such edge exists in the graph.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        return self._incident_edges.get_edge(vertex_or_edge)

    def _remove_edge(self, edge: Edge) -> int:
        """Removes an incident edge.

        Returns:
            int: Number of edges removed (more than one for parallel edges).
        """
        if edge.vertex1.label != self.label and edge.vertex2.label != self.label:
            raise ValueError(
                f"edge {edge} did not have a vertex matching this vertex {self}"
            )
        self._incident_edges.remove_edge_from(edge)
        return 1 + edge.parallel_edge_count


class _DiVertex(DiVertex):
    """Concrete implementation of the abstract :class:`DiVertex` class."""

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

    def delete_loops(self) -> int:
        """Deletes all edges that are loops on this vertex.

        Returns:
            int: The number of loops deleted.
        """
        loops = self._incident_edges.loop_edge
        deletion_count = loops._parallel_edge_count + 1
        self._parent_graph._edges.remove(loops)
        self._parent_graph._edges_with_freq_weight.pop(loops)
        self._incident_edges.remove_edge_from(loops)
        return deletion_count

    def get_adj_for_search(
        self, parent: Optional["Vertex"] = None, reverse_graph: bool = False
    ) -> Set["Vertex"]:
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

    def _add_edge(self, edge: Edge) -> None:
        """Adds an edge.

        If an incident edge already exists with the same vertices, it is overwritten.

        Raises:
            ValueError: If the new edge does not include this vertex.
        """
        if edge.vertex1.label != self.label and edge.vertex2.label != self.label:
            raise ValueError(
                f"Edge {edge} did not have a vertex matching this vertex {{{self.label}}}"
            )
        self._incident_edges.add_edge(edge)

    def _get_edge(self, vertex_or_edge: GraphPrimitive) -> Optional[Edge]:
        """Gets the incident edge specified by ``vertex_or_edge``, or None if no such edge exists.

        Args:
            vertex_or_edge: For undirected graphs, the adjacent vertex (relative to this vertex)
                will suffice, but may also be an edge. For directed graphs, an edge must be
                specified that includes this vertex and the adjacent vertex, since the order of the
                vertices distinguishes between incoming and outgoing edges.

        Returns:
            Edge: The specified edge or None if no such edge exists in the graph.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        return self._incident_edges.get_edge(vertex_or_edge)

    def _remove_edge(self, edge: Edge) -> int:
        """Removes an incident edge.

        Returns:
            int: Number of edges removed (more than one for parallel edges).
        """
        if edge.vertex1.label != self.label and edge.vertex2.label != self.label:
            raise ValueError(
                f"edge {edge} did not have a vertex matching this vertex {self}"
            )
        self._incident_edges.remove_edge_from(edge)
        return 1 + edge.parallel_edge_count
