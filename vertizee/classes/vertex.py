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

"""A point in a graph. Also called a node.

Classes and type aliases:
    * :class:`Vertex` - The class used to create vertex objects.
    * :class:`VertexType` - The type alias ``Union[int, str, Vertex]``. Any context that accepts
      ``VertexType`` permits referring to vertices by label (``str`` or ``int``) or by
      ``Vertex`` object.
    * :class:`IncidentEdges` - Collection class to manage edges incident on a shared vertex.

Function summary:
    * :func:`get_vertex_label` - Returns the vertex label string for the specified vertex.
    * :func:`is_vertex_type` - Returns True for instances of ``VertexType`` and False otherwise.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Set, TYPE_CHECKING, Union

from vertizee.classes import parsed_primitives

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.edge import EdgeType
    from vertizee.classes.graph_base import GraphBase
    from vertizee.classes.parsed_primitives import GraphPrimitive


# Type alias
VertexType = Union[int, str, "Vertex"]


def get_vertex_label(other: "VertexType") -> str:
    """Returns the vertex label string for the specified vertex."""
    # if not is_vertex_type(other):
    #     raise TypeError(f"{type(other).__name__} object found; must be int, str, or Vertex")
    if isinstance(other, Vertex):
        return other.label
    return str(other)


def is_vertex_type(other: "VertexType") -> bool:
    """Returns True if ``other`` is an instance of one of the types defined by ``VertexType``:
    ``int``, ``str``, or ``Vertex``."""
    if not isinstance(other, int) and not isinstance(other, str) and not isinstance(other, Vertex):
        return False
    return True


class Vertex:
    """A graph primitive representing a point (also called a node) that may be connected to other
    vertices via edges.

    No two vertices within a graph may share the same label. Labels may be strings or integers, but
    internally, they are always converted to strings. The following statements are equivalent::

        graph.add_vertex(1)
        graph.add_vertex("1")

    Vertices may be referenced by specifying labels as either strings or integers  (if the label
    represents an integer), or by using a `Vertex` object directly. For example, the following
    statements are equivalent and return the :class:`Vertex` object with label "1"::

        one = graph.add_vertex(1)
        graph[1]
        graph["1"]
        graph[one]

    To ensure the integrity of the graph, vertices should never be instantiated directly.
    Attempting to construct a vertex using its ``__init__`` method will raise an error. Instead,
    use the methods :meth:`GraphBase.add_vertex <vertizee.classes.graph_base.GraphBase.add_vertex>`
    and :meth:`GraphBase.add_vertices_from
    <vertizee.classes.graph_base.GraphBase.add_vertices_from>`.

    Each vertex stores references to its incident edges and the parent graph to which it belongs.

    Args:
        label: The label for this vertex. Must be unique to the graph.
        parent_graph: The parent graph to which this vertex belongs.

    Attributes:
        attr: Attribute dictionary to store ad hoc data associated with the vertex.
    """

    # Limit initialization to protected method `_create`.
    __create_key = object()

    @classmethod
    def _create(cls, label: Union[int, str], parent_graph: GraphBase) -> "Vertex":
        """Initializes a new Vertex object."""
        return Vertex(cls.__create_key, label, parent_graph)

    def __init__(self, create_key, label: Union[int, str], parent_graph: GraphBase):
        if create_key != Vertex.__create_key:
            raise ValueError(
                f"{self._runtime_type()} objects should be created using method "
                "GraphBase.add_vertex(); do not use __init__"
            )
        self._label = str(label)

        self.attr: dict = {}

        self._edges: IncidentEdges = IncidentEdges(self.label, parent_graph)
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

    def __eq__(self, other: VertexType):
        return self.__compare(other, "==")

    def __getitem__(self, key: Any) -> Any:
        """Supports index accessor notation to retrieve values from the `attr` dictionary.

        Example:
            >>> import vertizee as vz
            >>> g = vz.Graph()
            >>> g.add_vertex(1)
            1
            >>> g[1]["color"] = "blue"
            >>> g[1]["color"]
            'blue'

        Args:
            key: The `attr` dictionary key.

        Returns:
            Any: The value indexed by `key`.
        """
        return self.attr[key]

    def __ge__(self, other: VertexType):
        return self.__compare(other, ">=")

    def __gt__(self, other: VertexType):
        return self.__compare(other, ">")

    def __hash__(self):
        return hash(self.label)

    def __iter__(self):
        return self._edges.__iter__()

    def __le__(self, other: VertexType):
        return self.__compare(other, "<=")

    def __lt__(self, other: VertexType):
        return self.__compare(other, "<")

    def __repr__(self):
        return f"{self.label}"

    def __setitem__(self, key: Any, value: Any):
        """Supports index accessor notation to set values in the `attr` dictionary.

        Example:
            >>> import vertizee as vz
            >>> g = vz.Graph()
            >>> g.add_vertex(1)
            1
            >>> g[1]["color"] = "blue"
            >>> g[1]["color"]
            'blue'

        Args:
            key: The `attr` dictionary key.
            value: The value to assign to `key` in the `attr` dictionary.
        """
        self.attr[key] = value

    def __str__(self):
        return f"{self.label}"
        # return f"{self._runtime_type()} {{{self.label}}} with {self._edges}"

    @property
    def adjacent_vertices(self) -> Set["Vertex"]:
        """The set of all adjacent vertices."""
        return self._edges._adj_vertices.copy()

    @property
    def adjacent_vertices_incoming(self) -> Set["Vertex"]:
        """The set of all adjacent vertices from incoming edges. This is an empty set for
        undirected graphs."""
        return self._edges._adj_vertices_incoming.copy()

    @property
    def adjacent_vertices_outgoing(self) -> Set["Vertex"]:
        """The set of all adjacent vertices from outgoing edges. This is an empty set for
        undirected graphs."""
        return self._edges._adj_vertices_outgoing.copy()

    @property
    def degree(self) -> int:
        """The degree (or valance) of this vertex.

        The degree is the number of incident edges and self-loops are counted twice.
        """
        total = 0
        for edge in self.edges:
            if edge.is_loop():
                total += 2 * (1 + edge.parallel_edge_count)
            else:
                total += 1 + edge.parallel_edge_count
        return total

    def delete_loops(self) -> int:
        """Deletes all edges that are loops on this vertex.

        Returns:
            int: The number of loops deleted.
        """
        deletion_count = 0
        loops = []
        for edge in self._edges.loops:
            loops.append(edge)
            deletion_count += 1 + edge.parallel_edge_count
            self._parent_graph._edges.remove(edge)
            self._parent_graph._edges_with_freq_weight.pop(edge)
        for loop in loops:
            self._edges.remove_edge_from(loop)
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
        if self._parent_graph.is_directed_graph():
            if reverse_graph:
                return self.adjacent_vertices_incoming
            return self.adjacent_vertices_outgoing

        # undirected graph
        adj_vertices = self.adjacent_vertices
        if parent is not None:
            adj_vertices = adj_vertices - {parent}
        return adj_vertices

    def get_edge(self, *args: GraphPrimitive) -> Optional[EdgeType]:
        """Retrieves edge incident to this vertex by specifying a second vertex in ``args``.

        Args:
            *args: Any combination of graph primitives yielding an incident vertex.

        Returns:
            EdgeType: The edge specified by this vertex and ``args``, or None if no such edge
            exists.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.parsed_primitives>`
        """
        return self._edges.get_edge(*args)

    @property
    def edges(self) -> Set[EdgeType]:
        """The set of all incident edges (incoming, outgoing, and self-loops)."""
        return self._edges.edges

    @property
    def edges_incoming(self) -> Set[EdgeType]:
        """The set of incoming edges (i.e. edges where this vertex is the head).

        This is an empty set for undirected graphs. Use ``edges`` instead.

        Returns:
            Set[Edge]: The incoming edges.
        """
        return self._edges.incoming

    @property
    def edges_outgoing(self) -> Set[EdgeType]:
        """The set of outgoing edges (i.e. edges where this vertex is the tail).

        This is an empty set for undirected graphs. Use ``edges`` instead.

        Returns:
            Set[Edge]: The outgoing edges.
        """
        return self._edges.outgoing

    def is_incident_edge(self, *args: GraphPrimitive) -> bool:
        """Returns True if the edge specified by ``args`` is incident on this vertex.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.parsed_primitives>`
        """
        return self._edges.get_edge(*args) is not None

    @property
    def label(self) -> str:
        """The vertex label string."""
        return self._label

    @property
    def loops(self) -> Set[EdgeType]:
        """The set of self loop edges.

        Note:
            Since a single :class:`Edge <vertizee.classes.edge.Edge>` object represents all
            parallel edges between two vertices [or in the case of a directed graph, two
            :class:`DiEdge <vertizee.classes.edge.DiEdge>` objects, one for edge :math:`(a, b)`
            and a second for :math:`(b, a)`], the loops set will always contain exactly zero or
            one edge. A ``set`` is used for consistency with property :attr:`edges`.

        Returns:
            Set[Edge]: The set of loop edges.
        """
        return self._edges.loops

    @property
    def non_loop_edges(self) -> Set[EdgeType]:
        """ The set of all incident edges excluding self-loops."""
        non_loops = self._edges.edges - self._edges.loops
        return non_loops

    def _add_edge(self, edge: EdgeType):
        """Adds an edge.

        If an incident edge already exists with the same vertices, it is overwritten.

        Raises:
            ValueError: If the new edge does not include this vertex.
        """
        if edge.vertex1.label != self.label and edge.vertex2.label != self.label:
            raise ValueError(
                f"Edge ({{{edge.vertex1.label}, {edge.vertex2.label}}}) did not "
                f"have a vertex matching this vertex {{{self.label}}}"
            )
        self._edges.add_edge(edge)

    def _remove_edge(self, edge: EdgeType) -> int:
        """Removes an incident edge.

        Returns:
            int: Number of edges removed (more than one for parallel edges).
        """
        if edge.vertex1.label != self.label and edge.vertex2.label != self.label:
            raise ValueError(
                f"Edge ({{{edge.vertex1.label}, {edge.vertex2.label}}}) did not "
                f"have a vertex matching this vertex {{{self.label}}}"
            )
        self._edges.remove_edge_from(edge)
        return 1 + edge.parallel_edge_count

    def _runtime_type(self):
        """Returns the name of the runtime subclass."""
        return self.__class__.__name__


class IncidentEdges:
    """Collection of edges that are incident on a shared vertex.

    Attempting to add an edge that does not have the shared vertex raises an error. Self loops are
    tracked and may be accessed with the ``loops`` property. In the case of directed edges,
    the ``incoming`` and ``outgoing`` properties may be used. Collections of adjacent vertices are
    also maintained for algorithmic efficiency.

    Args:
        shared_vertex_label: The vertex label of the vertex shared by the incident edges.
        parent_graph: The graph to which the incident edges belong.
    """

    def __init__(self, shared_vertex_label: str, parent_graph: GraphBase):
        self._parent_graph = parent_graph

        self._adj_vertices: Set["Vertex"] = set()
        """The set of all vertices adjacent to the shared vertex."""

        self._adj_vertices_incoming: Set["Vertex"] = set()
        """Directed graphs only: the set of all vertices adjacent to the shared vertex from incoming
        edges."""

        self._adj_vertices_outgoing: Set["Vertex"] = set()
        """Directed graphs only: the set of all vertices adjacent to the shared vertex from outgoing
        edges."""

        self._edges: Dict[str, EdgeType] = {}
        """The dictionary of all incident edges: parallel, self loops, incoming, and outgoing.

        The dictionary keys are created by :func:`__create_edge_label`, and are a mapping from edge
        vertex keys to a consistent string representation.
        """

        self._incoming: Set[EdgeType] = set()
        """Directed graphs only: edges whose head vertex is ``_shared_vertex``."""

        self._loops: EdgeType = None
        """Loops on ``_shared_vertex``.

        Since all loops are parallel to each other, only one Edge object is needed.
        """

        self._outgoing: Set[EdgeType] = set()
        """Directed graphs only: edges whose tail vertex is ``_shared_vertex``."""

        self._shared_vertex_label: str = shared_vertex_label
        """The label of the vertex common between all of the incident edges."""

    def __eq__(self, other):
        if not isinstance(other, IncidentEdges):
            return False
        if self._shared_vertex_label != other._shared_vertex_label or len(self._edges) != len(
            other._edges
        ):
            return False
        if self._edges != other._edges:
            return False
        return True

    def __iter__(self):
        return iter(self._edges.values())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        str_edges = ", ".join(self._edges.keys())
        return f"IncidentEdges: {{{str_edges}}}"

    def add_edge(self, edge: EdgeType):
        """Adds an edge incident to the vertex specified by ``shared_vertex_label``.

        If an existing edge has the same vertices, it is overwritten.

        Args:
            edge (EdgeType): The edge to add.
        """
        if (
            edge.vertex1.label != self._shared_vertex_label
            and edge.vertex2.label != self._shared_vertex_label
        ):
            raise ValueError(
                f"Cannot add edge ({edge.vertex1.label}, {edge.vertex2.label}) since it does not"
                f" share vertex {{{self._shared_vertex_label}}}."
            )

        edge_label = _create_edge_label(
            edge.vertex1.label, edge.vertex2.label, self._parent_graph.is_directed_graph()
        )
        if edge.vertex1 == edge.vertex2:
            self._loops = edge
            self._edges[edge_label] = edge
            return

        adj_vertex = None
        is_outgoing_edge = True
        if edge.vertex1.label != self._shared_vertex_label:
            is_outgoing_edge = False
            adj_vertex = edge.vertex1
        else:
            adj_vertex = edge.vertex2

        self._edges[edge_label] = edge
        self._adj_vertices.add(adj_vertex)

        if self._parent_graph.is_directed_graph():
            if is_outgoing_edge:
                self._outgoing.add(edge)
                self._adj_vertices_outgoing.add(adj_vertex)
            else:
                self._incoming.add(edge)
                self._adj_vertices_incoming.add(adj_vertex)

    @property
    def adjacent_vertices(self) -> Set["Vertex"]:
        """The set of all vertices adjacent to the shared vertex."""
        return self._adj_vertices.copy()

    @property
    def adjacent_vertices_incoming(self) -> Set["Vertex"]:
        """The set of all vertices adjacent to the shared vertex from incoming edges. This property
        is only defined for directed graphs."""
        return self._adj_vertices_incoming.copy()

    @property
    def adjacent_vertices_outgoing(self) -> Set["Vertex"]:
        """The set of all vertices adjacent to the shared vertex from outgoing edges. This property
        is only defined for directed graphs."""
        return self._adj_vertices_outgoing.copy()

    @property
    def edges(self) -> Set[EdgeType]:
        """The set of all adjacent edges."""
        return set(self._edges.values())

    def get_edge(self, *args: GraphPrimitive) -> Optional[EdgeType]:
        """Gets the incident edge specified by ``args``, or None if no such edge exists.

        Args:
            *args: Graph primitives specifying an edge.

        Returns:
            Edge: The incident edge specified by ``args``, or None if no edge found.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.parsed_primitives>`
        """
        primitives = parsed_primitives.parse_graph_primitives(*args)
        edge_tuple = parsed_primitives.get_edge_tuple_from_parsed_primitives(primitives)
        if edge_tuple is None:
            return None

        edge_label = _create_edge_label(
            edge_tuple[0], edge_tuple[1], self._parent_graph.is_directed_graph()
        )
        if edge_label in self._edges:
            return self._edges[edge_label]
        return None

    @property
    def incoming(self) -> Set[EdgeType]:
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
    def loops(self) -> Set[EdgeType]:
        """The set of all self-loop edges."""
        if self._loops is None:
            return set()
        return {self._loops}

    @property
    def outgoing(self) -> Set[EdgeType]:
        """The set of all outgoing edges. This property is only defined for directed graphs."""
        return self._outgoing.copy()

    def remove_edge_from(self, edge: EdgeType):
        """Removes an edge."""
        edge_label = _create_edge_label(
            edge.vertex1.label,
            edge.vertex2.label,
            is_directed=self._parent_graph.is_directed_graph(),
        )
        if edge_label in self._edges:
            self._edges.pop(edge_label)

        if self._loops == edge:
            self._loops = None
        if edge in self._outgoing:
            self._outgoing.remove(edge)
        if edge in self._incoming:
            self._incoming.remove(edge)


def _create_edge_label(v1_label: str, v2_label: str, is_directed: bool) -> str:
    """Creates an edge key based on the keys of two vertices.

    For undirected graphs, the vertex keys are sorted, such that if v1_label <= v2_label, then the
    new edge key will contain v1_label followed by v2_label. This provides a consistent mapping of
    undirected edges, such that both (v1_label, v2_label) as well as (v2_label, v1_label) produce
    the same edge key.

    Args:
        v1_label (str): The first vertex key of the edge.
        v2_label (str): The second vertex key of the edge.
        is_directed (bool): True indicates a directed graph, False an undirected graph.

    Returns:
        str: The edge key.
    """
    if is_directed:
        return f"({v1_label}, {v2_label})"

    # undirected edge
    if v1_label > v2_label:
        return f"({v2_label}, {v1_label})"
    return f"({v1_label}, {v2_label})"
