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

"""Graph data type supporting directed and undirected multigraphs.

This is parent classes of specific graph implementations:
    Undirected Graphs
    * Graph
    * MultiGraph
    * SimpleGraph

    Directed Graphs
    * DiGraph
    * MultiDiGraph
"""
# pylint: disable=too-many-public-methods

from __future__ import annotations
import random
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from vertizee.classes import graph_primitives
from vertizee.classes.edge import DEFAULT_WEIGHT

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.edge import EdgeType
    from vertizee.classes.vertex import Vertex, VertexKeyType


class GraphBase:
    """GraphBase is a generic graph class supporting both directed and undirected multigraphs.

    Initialization of `GraphBase` is limited to protected method `_create`, since this
    class should only be used for subclassing.

    Args:
        is_directed_graph (bool): Indicates if the graph is directed (True) or undirected
            (False).
        is_multigraph (bool): If True, then parallel edges are allowed. If False, then attempting
            to add a parallel edge raises an error.
        is_simple_graph (bool, optional): If True, then the graph enforces the property that it
            be simple graph (i.e. no parallel edges and no self loops). Attempting to add a
            parallel edge or a self loop raises an error. Defaults to False.
    """
    _create_key = object()

    @classmethod
    def _create(cls, is_directed_graph: bool, is_multigraph: bool,
                is_simple_graph: bool = False) -> 'GraphBase':
        """Initializes a new graph. Subclasses should enable initialization using the standard
        `__init__` method."""
        return GraphBase(
            cls._create_key, is_directed_graph=is_directed_graph,
            is_multigraph=is_multigraph, is_simple_graph=is_simple_graph)

    def __init__(self, create_key, is_directed_graph: bool, is_multigraph: bool,
                 is_simple_graph: bool = False):
        if create_key != GraphBase._create_key:
            raise ValueError('Graph objects must be initialized using `_create`.')

        self._edges: Set[EdgeType] = set()
        self._edges_with_freq_weight: Dict[EdgeType, int] = {}
        """ `_edges_with_freq_weight` associates each edge with the number of times it occurs in
        the graph as a parallel edge. For non-parallel edges, this count is 1. This is used for
        correctly weighting the random sampling of edges.
        """

        self._graph_state_is_simple_graph = True
        """Tracks the current state of the graph as edges are added, and changes to False (not
        a simple graph) if a parallel edge or self loop is added."""

        self._is_directed_graph = is_directed_graph
        self._is_multigraph = is_multigraph
        self._is_simple_graph = is_simple_graph
        self._vertices: Dict[str, Vertex] = {}

    def __contains__(self, vertex: VertexKeyType) -> bool:
        parsed_primitives = graph_primitives.parse_graph_primitives(vertex)
        if len(parsed_primitives.vertex_keys) == 0:
            raise ValueError('Must specify a vertex when checking for Graph membership with `in`'
                             'operator.')
        return parsed_primitives.vertex_keys[0] in self._vertices

    def __getitem__(self, vertex_key: VertexKeyType) -> Vertex:
        """Support index accessor notation to retrieve vertices.

        Example:
            vertex1 = graph[1]
            edge12 = vertex1[2]  # same as graph[1][2]

        Args:
            vertex_key (VertexKeyType): The key label of the vertex.

        Returns:
            Vertex: The vertex specified by the vertex key label.
        """
        return self._get_vertex(vertex_key)

    def __iter__(self):
        return iter(self._vertices.values())

    def __len__(self):
        """Returns the number of vertices in the graph when the `len` function is applied."""
        return len(self._vertices)

    def add_edge(self, v1: VertexKeyType, v2: VertexKeyType,
                 weight: Optional[float] = DEFAULT_WEIGHT, parallel_edge_count: Optional[int] = 0,
                 parallel_edge_weights: Optional[List[float]] = None) -> EdgeType:
        """Adds a new edge to the graph.

        If there is already an edge with matching vertices, then the internal Edge object is
        modified by incrementing the parallel edge count. If only one vertex specified, then the
        edge is assumed to be a self loop.

        Args:
            vertex1 (VertexKeyType): The first vertex.
            vertex2 (VertexKeyType): The second vertex.
            weight (float, optional): The edge weight. Defaults to 0.0.
            parallel_edge_count (int, optional): The number of parallel edges, not including the
                initial edge between the vertices. Defaults to 0.
            parallel_edge_weights (List[float], optional): The weights of parallel edges. Defaults
                to None.

        Returns:
            EdgeType: The newly added edge (or pre-existing edge if parallel edged was
            added). If the graph is directed, an instance of `DiEdge` will be returned, otherwise
            `Edge`.
        """
        # Import locally to avoid circular dependency errors.
        # pylint: disable=import-outside-toplevel
        from vertizee.classes.edge import DiEdge
        from vertizee.classes.edge import Edge

        parsed_primitives = graph_primitives.parse_graph_primitives(v1)
        vertex1 = self._get_or_add_vertex(parsed_primitives.vertex_keys[0])
        parsed_primitives = graph_primitives.parse_graph_primitives(v2)
        vertex2 = self._get_or_add_vertex(parsed_primitives.vertex_keys[0])

        error_msg = None
        existing_v1v2 = self.get_edge(vertex1, vertex2)

        if parallel_edge_count > 0 or existing_v1v2 is not None:
            error_msg = 'parallel edge'
        elif vertex1.key == vertex2.key:
            error_msg = 'edge with a loop'

        if error_msg is not None:
            if self._is_simple_graph:
                raise ValueError(f'Attempted to add {error_msg}. This graph was initialized as '
                                 'a simple graph (i.e. no loops and no parallel edges).')
            if error_msg == 'parallel edge' and not self._is_multigraph:
                raise ValueError(
                    f'Attempted to add {error_msg}. This graph is not a multigraph and therefore '
                    'does not support parallel edges.')
            self._graph_state_is_simple_graph = False

        if self._is_directed_graph:
            new_edge = DiEdge._create(
                vertex1, vertex2, weight=weight, parallel_edge_count=parallel_edge_count,
                parallel_edge_weights=parallel_edge_weights)
        else:
            new_edge = Edge._create(
                vertex1, vertex2, weight=weight, parallel_edge_count=parallel_edge_count,
                parallel_edge_weights=parallel_edge_weights)
        if existing_v1v2 is not None:
            _merge_parallel_edges(existing_v1v2, new_edge=new_edge)
            self._edges_with_freq_weight[existing_v1v2] = 1 + existing_v1v2.parallel_edge_count
            return existing_v1v2
        else:
            self._edges.add(new_edge)
            self._edges_with_freq_weight[new_edge] = 1 + new_edge.parallel_edge_count
            # Handle vertex book keeping.
            new_edge.vertex1._add_edge(new_edge)
            new_edge.vertex2._add_edge(new_edge)
            return new_edge

    def add_edges_from(self, *args: graph_primitives.GraphPrimitive) -> int:
        """Adds all edges that match `args`.

        Args:
            *args (GraphPrimitive): Graph primitives specifying edges to add.

        Returns:
            int: The number of edges added.

        Example:
            graph.add_edges_from([(0, 1), (0, 2), (2, 1), (2, 2)])
        """
        parsed_primitives = graph_primitives.parse_graph_primitives(*args)
        edge_tuples = graph_primitives.get_all_edge_tuples_from_parsed_primitives(parsed_primitives)

        for edge_tuple in edge_tuples:
            if len(edge_tuple) == 2:
                self.add_edge(edge_tuple[0], edge_tuple[1])
            elif len(edge_tuple) == 3:
                self.add_edge(edge_tuple[0], edge_tuple[1], weight=edge_tuple[2])
            else:
                raise ValueError(f'Expected `edge_tuple` to have either 2 or 3 elements. Actual '
                                 f' length {len(edge_tuple)}.')

        return len(edge_tuples)

    def add_vertex(self, vertex_key: VertexKeyType) -> Vertex:
        """Adds a vertex to the graph and returns the new Vertex object.

        If an existing vertex matches the vertex key, the existing vertex is returned.

        Args:
            vertex_key (VertexKeyType): The new vertex key (or Vertex object from
                which to get the key).

        Raises:
            ValueError: [description]

        Returns:
            Vertex: The new vertex (or an existing vertex matching the vertex key).
        """
        # Import Vertex locally to avoid circular dependency errors.
        # pylint: disable=import-outside-toplevel
        from vertizee.classes.vertex import Vertex

        parsed_primitives = graph_primitives.parse_graph_primitives(vertex_key)
        if len(parsed_primitives.vertex_keys) == 0:
            raise ValueError('Must specify a valid vertex key.')
        new_key = parsed_primitives.vertex_keys[0]

        if new_key not in self._vertices:
            new_vertex = Vertex._create(new_key, parent_graph=self)
            self._vertices[new_key] = new_vertex
            return new_vertex
        else:
            return self._vertices[new_key]

    def add_vertices_from(self, *args: graph_primitives.GraphPrimitive) -> int:
        """Adds all vertices that match `args`.

        Args:
            *args (GraphPrimitive): Graph primitives specifying vertices to add.

        Returns:
            int: The number of vertices added.

        Example:
            graph.add_vertices_from([0, 1, 2, 3], [('a', 'b'), ('a', 'd')])
            # Vertices added: 0, 1, 2, 3, a, b, d
        """
        parsed_primitives = graph_primitives.parse_graph_primitives(*args)
        vertices = graph_primitives.get_all_vertices_from_parsed_primitives(parsed_primitives)

        for vertex_key in vertices:
            self.add_vertex(vertex_key)

        return len(vertices)

    def clear(self):
        """Removes all edges and vertices from the graph."""
        self._vertices.clear()
        self._edges.clear()
        self._edges_with_freq_weight.clear()

    def convert_to_simple_graph(self):
        """Convert this graph to a simple graph (i.e., no loops and no parallel edges)."""
        temp_edges = self._edges.copy()
        while temp_edges:
            edge = temp_edges.pop()
            edge._parallel_edge_count = 0
            edge._parallel_edge_weights = []
            if edge.is_loop():
                self.remove_all_edges_from(edge)
        self._graph_state_is_simple_graph = True

    def current_state_is_simple_graph(self) -> bool:
        """Returns True if the current state of the graph is a simple graph (i.e. no loops or
        parallel edges)."""
        if not self._graph_state_is_simple_graph:
            temp_edges = self._edges.copy()
            while temp_edges:
                edge = temp_edges.pop()
                if edge.is_loop() or edge.parallel_edge_count > 0:
                    return False
            self._graph_state_is_simple_graph = True
        return True

    @property
    def edge_count(self) -> int:
        count = len(self._edges)
        if self._graph_state_is_simple_graph:
            return count

        for edge in self._edges:
            count += edge.parallel_edge_count
        return count

    @property
    def edges(self) -> Set[EdgeType]:
        """The set of graph edges."""
        return set(self._edges)

    def get_edge(self, *args: graph_primitives.GraphPrimitive) -> Optional[EdgeType]:
        """Gets the edge specified by the graph primitives, or None if no such edge exists.

        Args:
            *args (GraphPrimitive): graph primitives (i.e. vertices or edges)

        Returns:
            EdgeType: The edge or None if not found.
        """
        parsed_primitives = graph_primitives.parse_graph_primitives(*args)
        edge_tuple = graph_primitives.get_edge_tuple_from_parsed_primitives(parsed_primitives)
        if edge_tuple is None or edge_tuple[1] is None:
            return None
        if edge_tuple[0] not in self._vertices or edge_tuple[1] not in self._vertices:
            return None

        vertex = self._vertices[edge_tuple[0]]
        return vertex.get_edge(edge_tuple[0], edge_tuple[1])

    def get_all_graph_edges_from_parsed_primitives(
            self, parsed_primitives: 'ParsedPrimitives') -> Set[EdgeType]:
        """Gets edges from the graph matching vertex keys in `parsed_primitives`.

        Args:
            parsed_primitives (ParsedPrimitives): The vertex keys parsed from graph primitives.

        Returns:
            List[EdgeType]: A list of edges found matching vertex keys from
                `parsed_primitives`.
        """
        graph_edges: Set[EdgeType] = set()
        while len(parsed_primitives.edge_tuples) > 0:
            t = parsed_primitives.edge_tuples.pop()
            graph_edges.add(self.get_edge(t))
        while len(parsed_primitives.edge_tuples_weighted) > 0:
            t = parsed_primitives.edge_tuples_weighted.pop()
            graph_edges.add(self.get_edge(t))

        vertex_prev = None
        for vertex_current in parsed_primitives.vertex_keys:
            if vertex_prev is not None:
                graph_edges.add(self.get_edge(vertex_prev, vertex_current))
                vertex_prev = None
            else:
                vertex_prev = vertex_current

        graph_edges = [x for x in graph_edges if x is not None]
        return graph_edges

    def get_random_edge(self) -> Optional[EdgeType]:
        """Returns a randomly selected edge from the graph, or None if there are no edges.

        Since parallel edges are represented by Edge._parallel_edge_count, the random sampling
        is weighted by the parallel edge counts.

        Returns:
            Edge: The random edge, or None if no edges.
        """
        if len(self._edges) > 0:
            sample = random.choices(
                population=list(self._edges_with_freq_weight.keys()),
                weights=list(self._edges_with_freq_weight.values()),
                k=1
            )
            return sample[0]
        else:
            return None

    @property
    def graph_weight(self) -> float:
        """Returns the weight of all edges."""
        weight = 0.0
        for edge in self._edges:
            weight += edge.weight_with_parallel_edges
        return weight

    def has_edge(self, *args: graph_primitives.GraphPrimitive) -> bool:
        """Returns True if there is an edge in the graph matching the specified vertices or edge.

        Returns:
            bool: True if there is a matching edge, otherwise False.
        """
        return self.get_edge(*args) is not None

    def has_vertex(self, vertex_key: VertexKeyType) -> bool:
        """Check for presence of vertex in graph."""
        return self._get_vertex(vertex_key) is not None

    def is_directed_graph(self) -> bool:
        """Returns True if this is a directed graph (i.e. each edge points from a tail vertex
        to a head vertex)."""
        return self._is_directed_graph

    def is_multigraph(self) -> bool:
        """Returns True if this is a multigraph (i.e. a graph that allows parallel edges)."""
        return self._is_multigraph

    def merge_vertices(self, v1: VertexKeyType, v2: VertexKeyType):
        """Merge `vertex2` into `vertex1`.

        After the merge operation:
           - Incident edges of v2 are modified such that v2 is replaced by v1
           - Incident loops on v2 become loops on v1.
           - v2 is deleted from the graph
           - degree(v1) [post-merge] == degree(v1) + degree(v2) [pre-merge]

        Since Edge uses its vertices to compute its hash (see `~graph_base.Edge.__hash__`),
        these properties must be treated as immutable for the lifetime of the object. Therefore,
        each edge incident on vertex2 must be deleted.

        In some cases, an incident edge of vertex2 will be modified such that by replacing v2
        with v1, there exists an edge in the graph matching the new endpoints. In that case, the
        existing edge is updated by incrementing its `_parallel_edge_count` and appending to
        `_parallel_edge_weights` as needed.

        If the graph does not contain an edge matching the new endpoints after replacing v2 with
        v1, then a new Edge object is created and added to the graph.
        """
        parsed_primitives = graph_primitives.parse_graph_primitives(v1, v2)
        if len(parsed_primitives.vertex_keys) < 2 or parsed_primitives.vertex_keys[1] is None:
            raise ValueError('Must specify two vertices to complete vertex merge operation.')
        if parsed_primitives.vertex_keys[0] not in self._vertices \
                or parsed_primitives.vertex_keys[1] not in self._vertices:
            raise ValueError('Both vertices must be existing members of the graph to complete '
                             'a merge operation.')

        vertex1 = self._vertices[parsed_primitives.vertex_keys[0]]
        vertex2 = self._vertices[parsed_primitives.vertex_keys[1]]

        edges_to_delete: List[EdgeType] = []
        # Incident edges of vertex2, where vertex2 is to be replaced by vertex1.
        for edge in vertex2.edges:
            edges_to_delete.append(edge)
            if edge.vertex1 == vertex2:
                existing_edge = self.get_edge(vertex1, edge.vertex2)
            else:
                existing_edge = self.get_edge(edge.vertex1, vertex1)

            if existing_edge is not None:
                _merge_parallel_edges(existing_edge, edge)
            else:
                if edge.vertex1 == vertex2:
                    self.add_edge(
                        vertex1, edge.vertex2, weight=edge.weight,
                        parallel_edge_count=edge.parallel_edge_count,
                        parallel_edge_weights=edge.parallel_edge_weights)
                else:
                    self.add_edge(
                        edge.vertex1, vertex1, weight=edge.weight,
                        parallel_edge_count=edge.parallel_edge_count,
                        parallel_edge_weights=edge.parallel_edge_weights)

        # Incident loops on vertex2 become loops on vertex1.
        if len(vertex2.loops) > 0:
            v2_loop_edge = vertex2.loops.pop()
            edges_to_delete.append(v2_loop_edge)
            if len(vertex1.loops) == 0:
                self.add_edge(
                    vertex1, vertex1, weight=v2_loop_edge.weight,
                    parallel_edge_count=v2_loop_edge.parallel_edge_count,
                    parallel_edge_weights=v2_loop_edge.parallel_edge_weights)
            else:
                v1_loop_edge, *_ = vertex1.loops
                _merge_parallel_edges(v1_loop_edge, v2_loop_edge)
        # Delete indicated edges after finishing iteration.
        for deleted_edge in edges_to_delete:
            self.remove_all_edges_from(deleted_edge)

        # Delete v2 from the graph.
        self.remove_vertex(v2)

    def remove_all_edges_from(self, *args: graph_primitives.GraphPrimitive) -> int:
        """Deletes all edges (including parallel edges) that match `args`.

        Args:
            *args (GraphPrimitive): Graph primitives specifying edges to remove.

        Returns:
            int: The number of edges deleted.
        """
        parsed_primitives = graph_primitives.parse_graph_primitives(*args)
        edges_to_remove = self.get_all_graph_edges_from_parsed_primitives(parsed_primitives)

        deletion_count = len(edges_to_remove)
        for edge in edges_to_remove:
            deletion_count += edge.parallel_edge_count
            self._edges.remove(edge)
            self._edges_with_freq_weight.pop(edge)
            edge.vertex1._remove_edge(edge)
            edge.vertex2._remove_edge(edge)

        return deletion_count

    def remove_edge_from(self, *args: graph_primitives.GraphPrimitive) -> int:
        """Deletes only one of the edges for each edge specified by `args`.

        For example, if `args` specifies vertices u and w, then if there are two parallel edges
        between (u, w), only one of the parallel edges will be removed. For edges without parallel
        edges, then this method is the same as `~graph_base.GraphBase.remove_all_edges_from`.

        Args:
            *args (GraphPrimitive): Graph primitives specifying edges from which exactly one
                edge will be removed.

        Returns:
            int: The number of edges deleted.
        """
        parsed_primitives = graph_primitives.parse_graph_primitives(*args)
        edges_to_remove = self.get_all_graph_edges_from_parsed_primitives(parsed_primitives)

        deletion_count = len(edges_to_remove)
        for edge in edges_to_remove:
            if edge.parallel_edge_count == 0:
                self.remove_all_edges_from(edge)
            else:
                self._edges_with_freq_weight[edge] -= 1
                edge._parallel_edge_count -= 1
                while len(edge._parallel_edge_weights) > edge._parallel_edge_count:
                    edge._parallel_edge_weights.pop()
        return deletion_count

    def remove_degree_zero_vertices(self) -> int:
        """Removes all vertices with degree zero and returns the count of deleted vertices."""
        vertex_keys_to_remove = []
        for key, vertex in self._vertices.items():
            if vertex.degree == 0:
                vertex_keys_to_remove.append(key)
        for k in vertex_keys_to_remove:
            self._vertices.pop(k)
        return len(vertex_keys_to_remove)

    def remove_vertex(self, vertex_key: VertexKeyType):
        """Removes the indicated vertex.

        For a vertex to be removed, it must not have any incident edges (except self loops). Any
        incident edges on the vertex must be removed first.

        Args:
            vertex_key (Union[int, str,): A vertex key (or Vertex from which to retrieve the
                key).
        """
        parsed_primitives = graph_primitives.parse_graph_primitives(vertex_key)
        if len(parsed_primitives.vertex_keys) == 0:
            raise ValueError('Must specify a valid vertex key or Vertex.')

        lookup_key = parsed_primitives.vertex_keys[0]
        if lookup_key in self._vertices:
            graph_vertex = self._vertices[lookup_key]
            if len(graph_vertex.edges) > 0:
                raise ValueError(
                    f'Vertex {{{lookup_key}}} has incident edges. All incident edges (excluding '
                    'loops) must be deleted prior to deleting a vertex.')
            self._vertices.pop(lookup_key)

    @property
    def vertex_count(self):
        return len(self._vertices)

    @property
    def vertices(self) -> Set[Vertex]:
        """The set of graph vertices."""
        return set(self._vertices.values())

    def _deepcopy_into(self, graph_copy: 'GraphBase'):
        """Initializes a `GraphBase` instance `graph_copy` with a deep copy of this graph's
        properties.

        Args:
            graph_copy (GraphBase): The graph instance into which properties will be copied.
        """
        graph_copy._is_directed_graph = self._is_directed_graph
        graph_copy._is_multigraph = self._is_multigraph
        graph_copy._is_simple_graph = self._is_simple_graph
        for vertex_key in self._vertices:
            graph_copy.add_vertex(vertex_key)
        for edge in self._edges:
            graph_copy.add_edge(
                edge.vertex1, edge.vertex2, weight=edge.weight,
                parallel_edge_count=edge.parallel_edge_count,
                parallel_edge_weights=edge.parallel_edge_weights)

    def _get_or_add_vertex(self, vertex_key: VertexKeyType) -> Vertex:
        """Helper method to get a vertex, or if not found, add a new vertex.

        Args:
            vertex_key (Union[int, str,): A vertex key (or Vertex from which to retrieve the
                key).

        Returns:
            Vertex: The vertex that was either found or added.
        """
        parsed_primitives = graph_primitives.parse_graph_primitives(vertex_key)
        if len(parsed_primitives.vertex_keys) == 0:
            raise ValueError('Must specify a valid vertex key or Vertex.')
        key = parsed_primitives.vertex_keys[0]
        if key in self._vertices:
            return self._vertices[key]
        else:
            return self.add_vertex(key)

    def _get_vertex(self, vertex_key: VertexKeyType) -> Optional[Vertex]:
        """Returns the vertex specified by the given `vertex_key`, or None if not found."""
        parsed_primitives = graph_primitives.parse_graph_primitives(vertex_key)
        if len(parsed_primitives.vertex_keys) == 0:
            raise ValueError('Must specify a valid vertex key or Vertex.')
        lookup_key = parsed_primitives.vertex_keys[0]
        if lookup_key in self._vertices:
            return self._vertices[lookup_key]
        else:
            return None

    def _reverse_graph_into(self, reverse_graph: 'GraphBase'):
        """Initializes `reverse_graph` with the reverse of this graph (i.e. all directed edges
        pointing in the opposite direction).

        The reverse of a directed graph is also called the transpose or the converse. See
        https://en.wikipedia.org/wiki/Transpose_graph.

        Args:
            reverse_graph (GraphBase): The graph instance to initialize with the reverse of this
                graph.
        """
        if not self._is_directed_graph or not reverse_graph._is_directed_graph:
            raise ValueError('Reverse graphs may only be created for directed graphs.')
        reverse_graph.clear()

        for vertex_key in self._vertices:
            reverse_graph.add_vertex(vertex_key)
        for edge in self.edges:
            reverse_graph.add_edge(
                edge.vertex2, edge.vertex1, weight=edge.weight,
                parallel_edge_count=edge.parallel_edge_count,
                parallel_edge_weights=edge.parallel_edge_weights)


def _merge_parallel_edges(existing: EdgeType, new_edge: EdgeType):
    """Helper method to merge an edge into an existing parallel edge."""
    existing._parallel_edge_weights += [new_edge.weight] + new_edge.parallel_edge_weights
    existing._parallel_edge_count += 1 + new_edge.parallel_edge_count
    while len(existing.parallel_edge_weights) > existing.parallel_edge_count:
        existing._parallel_edge_weights.pop()
