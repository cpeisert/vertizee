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

"""Vertex data type.

* Vertex - Base class for graph vertex implementations.
* VertexKeyType - Type alias for Union[int, str, Vertex] used for flexible lookup by referring
    to vertex keys as either integers, strings, or Vertex objects, which all map to the same
    interval string representation of the vertex key.
* IncidentEdges - Collection class to manage edges incident on a shared vertex.
"""

from __future__ import annotations
from typing import Dict, Optional, Set, TYPE_CHECKING, Union

from vertizee.classes import graph_primitives

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.edge import EdgeType
    from vertizee.classes.graph_base import GraphBase


# Type alias
VertexKeyType = Union[int, str, 'Vertex']


def get_vertex_key(other: VertexKeyType) -> str:
    # if not is_vertex_key_type(other):
    #     raise TypeError(f"{type(other).__name__} object found; must be int, str, or Vertex")
    if isinstance(other, Vertex):
        return other.key
    return str(other)


def is_vertex_key_type(other: VertexKeyType) -> bool:
    if not isinstance(other, int) and not isinstance(other, str) and not isinstance(other, Vertex):
        return False
    return True


class Vertex:
    """Vertex is a graph primitive representing a point (also called a node) that may be connected
    to other vertices via edges.

    No two vertices within a graph may share the same label. Labels may be strings or integers.
    The following are equivalent and would result in the creation of one vertex labeled "1".

        graph.add_vertex(1)
        graph.add_vertex('1')

    To ensure the integrity of the graph, vertices should never be instantiated directly.
    Attempting to construct a vertex using its `__init__` method will raise an error.

    Each vertex stores references to its incident edges and the parent graph to which it belongs.

     Args:
        key_label (Union[int, str]): The key label for this vertex. Must be unique to the graph.
        parent_graph (GraphBase): The parent graph to which this vertex belongs.
    """
    # Limit initialization to protected method `_create`.
    __create_key = object()

    @classmethod
    def _create(cls, key_label: Union[int, str],
                parent_graph: GraphBase) -> 'Vertex':
        """Initializes a new Vertex object."""
        return Vertex(cls.__create_key, key_label, parent_graph)

    def __init__(self, create_key, key_label: Union[int, str],
                 parent_graph: GraphBase) -> 'Vertex':
        if create_key != Vertex.__create_key:
            raise ValueError('must initialize using `_create`; do not use `__init__`')
        self._key = str(key_label)

        self.attr: dict = {}
        """Custom attribute dictionary to store any additional data associated with vertices."""

        self._edges: IncidentEdges = IncidentEdges(self.key, parent_graph)
        self._parent_graph = parent_graph

    def __compare(self, other: VertexKeyType, operator: str) -> bool:
        if not is_vertex_key_type(other):
            return False
        other_key = get_vertex_key(other)
        compare = False
        if operator == '==':
            if self.key == other_key:
                compare = True
        elif operator == '<':
            if self.key < other_key:
                compare = True
        elif operator == '<=':
            if self.key <= other_key:
                compare = True
        elif operator == '>':
            if self.key > other_key:
                compare = True
        elif operator == '>=':
            if self.key >= other_key:
                compare = True
        return compare

    def __eq__(self, other: VertexKeyType):
        return self.__compare(other, '==')

    def __getitem__(self, vertex_key: VertexKeyType) -> EdgeType:
        """Support index accessor notation to retrieve edges.

        Example::

            vertex1 = graph[1]
            edge12 = vertex1[2]  # same as graph[1][2]

        Args:
            vertex_key (Union[int, str, Vertex]): The key label of the edge's second vertex.

        Returns:
            Edge: The edge specified by vertex pair (self.key, vertex_key).
        """
        return self._parent_graph.get_edge(self.key, vertex_key)

    def __ge__(self, other: VertexKeyType):
        return self.__compare(other, '>=')

    def __gt__(self, other: VertexKeyType):
        return self.__compare(other, '>')

    def __hash__(self):
        return hash(self.key)

    def __iter__(self):
        return self._edges.__iter__()

    def __le__(self, other: VertexKeyType):
        return self.__compare(other, '<=')

    def __lt__(self, other: VertexKeyType):
        return self.__compare(other, '<')

    def __repr__(self):
        return f'{self.key}'

    def __str__(self):
        return f'{self.key}'
        # return f'{self._runtime_type()} {{{self.key}}} with {self._edges}'

    @property
    def adjacent_vertices(self) -> Set['Vertex']:
        return self._edges._adj_vertices.copy()

    @property
    def adjacent_vertices_incoming(self) -> Set['Vertex']:
        return self._edges._adj_vertices_incoming.copy()

    @property
    def adjacent_vertices_outgoing(self) -> Set['Vertex']:
        return self._edges._adj_vertices_outgoing.copy()

    @property
    def degree(self) -> int:
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

    def get_adj_for_search(self, parent: Optional['Vertex'] = None,
                           reverse_graph: Optional[bool] = False) -> Set['Vertex']:
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
            parent (Vertex, optional): The parent vertex (i.e. predecessor) in the search tree.
                Defaults to None.
            reverse_graph (bool, optional): For directed graphs, setting to True will yield a
                traversal as if the graph were reversed (i.e. the reverse/transpose/converse
                graph). Defaults to False.
        """
        if self._parent_graph.is_directed_graph():
            if reverse_graph:
                return self.adjacent_vertices_incoming
            else:
                return self.adjacent_vertices_outgoing
        else:  # undirected graph
            adj_vertices = self.adjacent_vertices
            if parent is not None:
                adj_vertices = adj_vertices - {parent}
            return adj_vertices

    def get_edge(self, *args: graph_primitives.GraphPrimitive) -> Optional[EdgeType]:
        """Retrieves edge incident to this vertex by specifying a second vertex in args.

        Args:
            *args (GraphPrimitive): Any combination of graph primitives yielding a vertex of an
                incident edge.

        Returns:
            Edge: The edge specified by args.
        """
        return self._edges.get_edge(*args)

    @property
    def edges(self) -> Set[EdgeType]:
        """The set of all incident edges: incoming, outgoing, and self-loops.

        Returns:
            Set[Edge]: The set of edges incident to this vertex.
        """
        return self._edges.edges

    @property
    def edges_incoming(self) -> Set[EdgeType]:
        """Return incoming edges (i.e. edges where this vertex is the head).

        If the graph is undirected, then `edges_incoming` will be an empty set. Use `edges`
        instead.

        Returns:
            Set[Edge]: The incoming edges.
        """
        return self._edges.incoming

    @property
    def edges_outgoing(self) -> Set[EdgeType]:
        """Return outgoing edges (i.e. edges where this vertex is the tail).

        If the graph is undirected, then `edges_outgoing` will be an empty set. Use `edges`
        instead.

        Returns:
            Set[Edge]: The outgoing edges.
        """
        return self._edges.outgoing

    def is_incident_edge(self, *args: graph_primitives.GraphPrimitive) -> bool:
        return self._edges.get_edge(*args) is not None

    @property
    def key(self) -> str:
        return self._key

    @property
    def loops(self) -> Set[EdgeType]:
        """The set of self loop edges.

        Note:
            Since one Edge object represents all parallel edges between two vertices (or in
            the case of a directed graph, two Edge objects, one for edge (a, b) and a second
            for (b, a)), the loops set will always contain exactly zero or one item. A Set is used
            for consistency with property `~graph_base.Vertex.edges`.

        Returns:
            Set[Edge]: The set of loop edges.
        """
        return self._edges.loops

    @property
    def non_loop_edges(self) -> Set[EdgeType]:
        """The set of self loop edges.

        Returns:
            Set[Edge]: The set of all incident edges excluding self-loops.
        """
        non_loops = self._edges.edges - self._edges.loops
        return non_loops

    def _add_edge(self, edge: EdgeType):
        """Adds an edge.

        If an incident edge already exists with the same vertices, it is overwritten.

        Raises:
            ValueError: If the new edge does not include this vertex.
        """
        if edge.vertex1.key != self.key and edge.vertex2.key != self.key:
            raise ValueError(f'Edge ({{{edge.vertex1.key}, {edge.vertex2.key}}}) did not '
                             f'have a vertex matching this vertex {{{self.key}}}')
        self._edges.add_edge(edge)

    def _remove_edge(self, edge: EdgeType) -> int:
        """Removes an incident edge.

        Returns:
            int: Number of edges removed (more than one for parallel edges).
        """
        if edge.vertex1.key != self.key and edge.vertex2.key != self.key:
            raise ValueError(f'Edge ({{{edge.vertex1.key}, {edge.vertex2.key}}}) did not '
                             f'have a vertex matching this vertex {{{self.key}}}')
        self._edges.remove_edge_from(edge)
        return 1 + edge.parallel_edge_count

    def _runtime_type(self):
        """Returns the name of the runtime subclass."""
        return self.__class__.__name__


class IncidentEdges:
    """Collection of edges that are incident on a shared vertex.

    Attempting to add an edge that does not have the shared vertex raises an error. Incident edges
    are classified as loops, and in the case of directed edges: incoming and outgoing. Collections
    of adjacent nodes are also maintained for algorithmic efficiency.

    Args:
        shared_vertex_key (str): The vertex key of the vertex shared by the incident edges.
        parent_graph (GraphBase): The graph to which the incident edges belong.
    """
    def __init__(self, shared_vertex_key: str, parent_graph: GraphBase):
        self._parent_graph = parent_graph

        self._adj_vertices: Set['Vertex'] = set()
        """The set of all nodes adjacent to the shared vertex."""

        self._adj_vertices_incoming: Set['Vertex'] = set()
        """Directed graphs only: the set of all nodes adjacent to the shared vertex from incoming
        edges."""

        self._adj_vertices_outgoing: Set['Vertex'] = set()
        """Directed graphs only: the set of all nodes adjacent to the shared vertex from outgoing
        edges."""

        self._edges: Dict[str, EdgeType] = {}
        """The dictionary of all incident edges: parallel, self loops, incoming, and outgoing.

        The dictionary keys are created by `graph_base.IncidentEdges.__create_edge_key`, and are
        a mapping from edge vertex keys to a consistent string representation.
        """

        self._incoming: Set[EdgeType] = set()
        """Directed graphs only: edges whose head vertex is `_shared_vertex`."""

        self._loops: EdgeType = None
        """Loops on `_shared_vertex`.

        Since all loops are parallel to each other, only one Edge object is needed. The loop
        edge is also referenced in `~graph_base.IncidentEdges._edges`."""

        self._outgoing: Set[EdgeType] = set()
        """Directed graphs only: edges whose tail vertex is `_shared_vertex`."""

        self._shared_vertex_key: str = shared_vertex_key
        """The key of the vertex common between all of the incident edges."""

    def __eq__(self, other):
        if not isinstance(other, IncidentEdges):
            return False
        if self._shared_vertex_key != other._shared_vertex_key \
                or len(self._edges) != len(other._edges):
            return False
        if self._edges != other._edges:
            return False
        return True

    def __iter__(self):
        return iter(self._edges.values())

    def __str__(self):
        str_edges = ', '.join(self._edges.keys())
        return f'IncidentEdges: {{{str_edges}}}'

    def add_edge(self, edge: EdgeType):
        """Adds an edge incident to the vertex specified by `_shared_vertex_key`.

        If an existing edge has the same vertices, it is overwritten.

        Args:
            edge (EdgeType): The edge to add.
        """
        if edge.vertex1.key != self._shared_vertex_key and \
                edge.vertex2.key != self._shared_vertex_key:
            raise ValueError(
                f'Cannot add edge ({edge.vertex1.key}, {edge.vertex2.key}) since it does not'
                f' share vertex {{{self._shared_vertex_key}}}.')

        edge_key = _create_edge_key(
            edge.vertex1.key, edge.vertex2.key, self._parent_graph._is_directed_graph)
        if edge.vertex1 == edge.vertex2:
            self._loops = edge
            self._edges[edge_key] = edge
            return

        adj_vertex = None
        is_outgoing_edge = True
        if edge.vertex1.key != self._shared_vertex_key:
            is_outgoing_edge = False
            adj_vertex = edge.vertex1
        else:
            adj_vertex = edge.vertex2

        self._edges[edge_key] = edge
        self._adj_vertices.add(adj_vertex)

        if self._parent_graph._is_directed_graph:
            if is_outgoing_edge:
                self._outgoing.add(edge)
                self._adj_vertices_outgoing.add(adj_vertex)
            else:
                self._incoming.add(edge)
                self._adj_vertices_incoming.add(adj_vertex)

    @property
    def adjacent_vertices(self) -> Set['Vertex']:
        return self._adj_vertices.copy()

    @property
    def adjacent_vertices_incoming(self) -> Set['Vertex']:
        return self._adj_vertices_incoming.copy()

    @property
    def adjacent_vertices_outgoing(self) -> Set['Vertex']:
        return self._adj_vertices_outgoing.copy()

    @property
    def edges(self) -> Set[EdgeType]:
        return set(self._edges.values())

    def get_edge(self, *args: graph_primitives.GraphPrimitive) -> Optional[EdgeType]:
        """Gets the incident edge specified by `args`, or None if no such edge exists.

        Args:
            *args (GraphPrimitive): Graph primitives specifying an edge.

        Returns:
            Edge: The incident edge specified by `args`, or None if no edge found.
        """
        parsed_primitives = graph_primitives.parse_graph_primitives(*args)
        edge_tuple = graph_primitives.get_edge_tuple_from_parsed_primitives(parsed_primitives)
        if edge_tuple is None:
            return None

        edge_key = _create_edge_key(
            edge_tuple[0], edge_tuple[1], self._parent_graph._is_directed_graph)
        if edge_key in self._edges:
            return self._edges[edge_key]
        else:
            return None

    @property
    def incoming(self) -> Set[EdgeType]:
        return self._incoming.copy()

    @property
    def incoming_edge_vertices(self) -> Set['Vertex']:
        vertices = set()
        for edge in self._incoming:
            vertices.add(edge.vertex1)
        return vertices

    @property
    def loops(self) -> Set[EdgeType]:
        if self._loops is None:
            return set()
        else:
            return {self._loops}

    @property
    def outgoing(self) -> Set[EdgeType]:
        return self._outgoing.copy()

    def remove_edge_from(self, edge: EdgeType):
        """Remove an edge."""
        edge_key = _create_edge_key(edge.vertex1.key, edge.vertex2.key,
                                    is_directed=self._parent_graph._is_directed_graph)
        if edge_key in self._edges:
            self._edges.pop(edge_key)

        if self._loops == edge:
            self._loops = None
        if edge in self._outgoing:
            self._outgoing.remove(edge)
        if edge in self._incoming:
            self._incoming.remove(edge)


def _create_edge_key(v1_key: str, v2_key: str, is_directed: bool) -> str:
    """Creates an edge key based on the keys of two vertices.

    For undirected graphs, the vertex keys are sorted, such that if v1_key <= v2_key, then the
    new edge key will contain v1_key followed by v2_key. This provides a consistent mapping of
    undirected edges, such that both (v1_key, v2_key) as well as (v2_key, v1_key) produce the
    same edge key.

    Args:
        v1_key (str): The first vertex key of the edge.
        v2_key (str): The second vertex key of the edge.
        is_directed (bool): True indicates a directed graph, False an undirected graph.

    Returns:
        str: The edge key.
    """
    if is_directed:
        return f'({v1_key}, {v2_key})'
    else:  # Undirected edge
        if v1_key <= v2_key:
            return f'({v1_key}, {v2_key})'
        else:
            return f'({v2_key}, {v1_key})'
