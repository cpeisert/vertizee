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

"""Base class for graphs, multigraphs, and their directed graph variants.

See Also:
    * :class:`DiEdge <vertizee.classes.edge.DiEdge>`
    * :class:`DiGraph <vertizee.classes.digraph.DiGraph>`
    * :class:`Edge <vertizee.classes.edge.Edge>`
    * :class:`Graph <vertizee.classes.graph.Graph>`
    * :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
    * :class:`MultiDiGraph <vertizee.classes.digraph.MultiDiGraph>`
    * :class:`MultiGraph <vertizee.classes.graph.MultiGraph>`
    * :class:`SimpleGraph <vertizee.classes.graph.SimpleGraph>`
    * :class:`Vertex <vertizee.classes.vertex.Vertex>`
"""
# pylint: disable=too-many-public-methods

from __future__ import annotations
from collections import abc
import copy
import random
from typing import Dict, List, Iterable, Iterator, Optional, overload, Set, Tuple, TYPE_CHECKING

from vertizee.classes import primitives_parsing
from vertizee.classes import edge
from vertizee import exception

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.primitives_parsing import _EdgeData, _VertexData
    from vertizee.classes.primitives_parsing import GraphPrimitive
    from vertizee.classes.edge import Edge, EdgeType
    from vertizee.classes.vertex import Vertex, VertexLabel, VertexType


class GraphBase:
    """A base class for graphs, multigraphs, and their directed graph variants..

    Initialization of ``GraphBase`` is limited to protected method ``_create``, since this
    class should only be used for subclassing.

    Args:
        is_directed_graph: Indicates if the graph is directed or undirected.
        is_multigraph: If True, then parallel edges are allowed. If False, then attempting
            to add a parallel edge raises an error.
        is_simple_graph: Optional; If True, then the graph enforces the property that it
            be a simple graph (i.e. no parallel edges and no self loops). Attempting to add a
            parallel edge or a self loop raises an error. Defaults to False.
    """

    _create_key = object()

    @classmethod
    def _create(
        cls, is_directed_graph: bool, is_multigraph: bool, is_simple_graph: bool = False
    ) -> "GraphBase":
        """Initializes a new graph. Subclasses should provide initialization using the standard
        ``__init__`` method."""
        return GraphBase(
            cls._create_key,
            is_directed_graph=is_directed_graph,
            is_multigraph=is_multigraph,
            is_simple_graph=is_simple_graph,
        )

    def __init__(
        self,
        create_key,
        is_directed_graph: bool,
        is_multigraph: bool,
        is_simple_graph: bool = False,
    ):
        if create_key != GraphBase._create_key:
            raise ValueError("GraphBase objects must be initialized using `_create`.")

        self._graph_state_is_simple_graph = True
        """Tracks the current state of the graph as edges are added. If a parallel edge or
        self loop is added, then this flag is set to False."""

        self._is_directed_graph = is_directed_graph
        self._is_multigraph = is_multigraph
        self._is_simple_graph = is_simple_graph
        self._is_weighted_graph = False
        """If an edge is added with a weight that is not equal to `DEFAULT_WEIGHT`, then this flag
        is set to True."""

        self._edges: Dict[str, Edge] = set()
        """A dictionary mapping edge labels (keys) to Edge objects. See
        :func:`create_label <vertizee.classes.edge.create_label>`."""

        self._edges_with_freq_weight: Dict[Edge, int] = {}
        """A dictionary mapping each edge to its frequency in the graph. This is used for
        correctly weighting the random sampling of edges. For non-parallel edges, this count is 1.
        """
        self._vertices: Dict[str, Vertex] = {}

    def __contains__(self, vertex: VertexType) -> bool:
        vertex_data: _VertexData = primitives_parsing.parse_vertex_type(vertex)
        return vertex_data.label in self._vertices

    def __deepcopy__(self, memo):
        new = self.__class__(
            self._create_key,
            self._is_directed_graph,
            self._is_multigraph,
            self._is_simple_graph
        )
        for vertex in self._vertices.values():
            new.add_vertex(vertex)
        for edge in self._edges:
            new._add_edge_from_edge(edge)
        return new

    @overload
    def __getitem__(self, vertex: "VertexType") -> Vertex:
        ...

    @overload
    def __getitem__(self, edge_tuple: Tuple["VertexType", "VertexType"]) -> Edge:
        ...

    def __getitem__(self, keys):
        """Supports index accessor notation to retrieve vertices (one vertex index) and edges (two
        vertex indices).

        Example:
            >>> import vertizee as vz
            >>> g = vz.Graph()
            >>> g.add_edge(1, 2)
            (1, 2)
            >>> g[1]
            1
            >>> g[1, 2]
            (1, 2)

        Args:
            keys: A vertex label, vertex tuple, vertex object, or edge tuple. Specifying one vertex
                will retrieve the ``Vertex`` object from the graph. Specifying two vertices will
                retrieve the associated edge object (``Edge`` or ``DiEdge``) from the graph.

        Returns:
            Union[Vertex, Edge]: The vertex specified by the vertex label or the edge
                specified by two vertices. If no matching vertex or edge found, returns None.

        Raises:
            IndexError: If ``keys`` is not exactly 1 or 2 ``VertexType`` keys.
            KeyError: If the graph does not contain a vertex or an edge matching ``keys``.
        """
        return_value: Optional[Edge, Vertex] = None
        if isinstance(keys, tuple):
            if len(keys) > 2:
                raise IndexError(
                    f"graph index lookup supports one or two vertices; found {len(keys)} keys"
                )
            if len(keys) == 1:
                return_value = self._get_vertex(keys[0])
            elif isinstance(keys[1], dict):
                return_value = self._get_vertex(keys[0])
            else:
                return_value = self._get_edge(keys[0], keys[1])
        else:
            return_value = self._get_vertex(keys)

        if return_value is None:
            raise KeyError(keys)
        return return_value

    def __iter__(self) -> Iterator[Vertex]:
        return iter(self._vertices.values())

    def __len__(self) -> int:
        """Returns the number of vertices in the graph when the built-in ``len`` function is
        applied."""
        return len(self._vertices)

    def add_edge(
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        weight: float = edge.DEFAULT_WEIGHT,
        **attr
    ) -> Edge:
        """Adds a new edge to the graph.

        If there is already an edge with matching vertices and the graph is a multigraph, then the
        existing :class:`Edge <vertizee.classes.edge.Edge>` object is modified by incrementing its
        parallel edge count.

        Args:
            vertex1: The first vertex.
            vertex2: The second vertex.
            weight: Optional; The edge weight. Defaults to 1.
            **attr: Optional; Keyword arguments to be added to the ``attr`` dictionary.

        Returns:
            Edge: The newly added edge (or pre-existing edge if a parallel edge was
            added). If the graph is directed, an instance of :class:`DiEdge
            <vertizee.classes.edge.DiEdge>` will be returned, otherwise :class:`Edge
            <vertizee.classes.edge.Edge>`.
        """
        existing_v1v2: Optional[Edge] = None
        edge_label = edge.create_edge_label(vertex1, vertex2, self._is_directed_graph)
        if edge_label in self._edges:
            existing_v1v2 = self._edges[edge_label]

        vertex1_data = primitives_parsing.parse_vertex_type(vertex1)
        vertex2_data = primitives_parsing.parse_vertex_type(vertex2)

        if existing_v1v2 and not self._is_multigraph:
            raise ValueError("parallel edges are only allowed in multigraphs; attempted to add "
                f"parallel edge {edge_label}")
        if vertex1_data.label == vertex2_data.label and self._is_simple_graph:
            raise ValueError("loops are not allowed in simple graphs; attempted to add loop edge "
                f"{edge_label}")

        if existing_v1v2 or vertex1_data.label == vertex2_data.label:
            self._graph_state_is_simple_graph = False

        if weight != edge.DEFAULT_WEIGHT:
            self._is_weighted_graph = True

        v1 = self._get_or_add_vertex(vertex1_data)
        v2 = self._get_or_add_vertex(vertex2_data)

        if self.is_directed_graph():
            new_edge = edge.DiEdge._create(v1, v2, weight=weight, **attr)
        else:
            new_edge = Edge._create(v1, v2, weight=weight, **attr)

        if existing_v1v2 is not None:
            _merge_parallel_edges(existing_v1v2, new_edge)
            self._edges_with_freq_weight[existing_v1v2] = existing_v1v2.multiplicity
            return existing_v1v2

        self._edges[edge_label] = new_edge
        self._edges_with_freq_weight[new_edge] = new_edge.multiplicity
        # Handle vertex book keeping.
        new_edge.vertex1._add_edge(new_edge)
        new_edge.vertex2._add_edge(new_edge)
        return new_edge

    def add_edges_from(self, edge_container: Iterable["EdgeType"], **attr) -> None:
        """Adds edges from a container where the edges are most often specified as tuples.

        Args:
            edge_container: Sequence of edges to add.
            **attr: Optional; Keyword arguments to be added to the ``attr`` dictionaries of each
                edge.

        See Also:
            :mod:`EdgeType <vertizee.classes.edge>`

        Example:
            >>> graph.add_edges_from([(0, 1), (0, 2), (2, 1), (2, 2)])
        """
        if not isinstance(edge_container, abc.Iterable):
            raise TypeError("edge_container must be iterable")

        for e in edge_container:
            edge_data: _EdgeData = primitives_parsing.parse_edge_type(e)
            new_edge = self.add_edge(edge_data.vertex1, edge_data.vertex2, weight=edge_data.weight,
                **edge_data.attr)
            for k, v in attr.items():
                new_edge[k] = v

    def add_vertex(self, label: "VertexLabel", **attr) -> Vertex:
        """Adds a vertex to the graph and returns the new Vertex object.

        If an existing vertex matches the vertex label, the existing vertex is returned.

        Args:
            label: The label to use for the new vertex. In order for a new vertex to be added, the
                label must not match any existing vertices in the graph.
            **attr: Optional; Keyword arguments to be added to the ``attr`` dictionary.

        Returns:
            Vertex: The new vertex (or an existing vertex matching the vertex label).
        """
        if not isinstance(label, (int, str)):
            raise ValueError(f"a vertex label must be a string or integer; found {type(label)}")
        vertex_label = str(label)

        if vertex_label not in self._vertices:
            new_vertex = Vertex._create(vertex_label, parent_graph=self, **attr)
            self._vertices[vertex_label] = new_vertex
            return new_vertex

        existing_vertex = self._vertices[vertex_label]
        for v, k in attr.items():
            existing_vertex[k] = v
        return existing_vertex

    def add_vertices_from(self, vertex_container: Iterable["VertexType"], **attr) -> None:
        """Adds vertices from a container, where the vertices are most often specified as strings
        or integers, but may also be tuples of the form ``Tuple[VertexLabel, AttributesDict]``.

        Args:
            vertex_container: Sequence of vertices to add.
            **attr: Optional; Keyword arguments to be added to the ``attr`` dictionaries of each
                vertex.

        See Also:
            :mod:`VertexType <vertizee.classes.vertex>`

        Example:
            >>> graph.add_vertices_from([0, 1, 2, 3])
        """
        if not isinstance(vertex_container, abc.Iterable):
            raise TypeError("vertex_container must be iterable")

        for vertex in vertex_container:
            vertex_data: _VertexData = primitives_parsing.parse_vertex_type(vertex)
            new_vertex = self.add_vertex(vertex_data.label, vertex_data.attr)
            for k, v in attr.items():
                new_vertex[k] = v

    def clear(self) -> None:
        """Removes all edges and vertices from the graph."""
        self._vertices.clear()
        self._edges.clear()
        self._edges_with_freq_weight.clear()

    def convert_to_simple_graph(self) -> None:
        """Convert this graph to a simple graph (i.e., no loops and no parallel edges)."""
        temp_edges = self._edges.copy()
        while temp_edges:
            edge = temp_edges.pop()
            if edge.is_loop():
                self.remove_all_edges_from(edge)
            else:
                edge._parallel_edge_count = 0
                edge._parallel_edge_weights = []

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

    def deepcopy(self) -> "GraphBase":
        """Returns a deep copy of this graph."""
        return copy.deepcopy(self)

    @property
    def edge_count(self) -> int:
        """The number of edges, including parallel edges if present."""
        count = len(self._edges)
        if self._graph_state_is_simple_graph:
            return count

        for edge in self._edges:
            count += edge.parallel_edge_count
        return count

    @property
    def edges(self) -> Set[Edge]:
        """The set of graph edges."""
        return set(self._edges)

    def get_all_graph_edges_from_parsed_primitives(
        self, primitives: ParsedVerticesAndEdges
    ) -> List[Edge]:
        """Gets graph edges that match vertex pairs and edge tuples in a
        :class:`ParsedVerticesAndEdges <vertizee.classes.primitives_parsing.ParsedVerticesAndEdges>` object.

        Args:
            primitives: The vertex labels and edge tuples parsed from graph primitives.

        Returns:
            List[Edge]: A list of edges based on the vertex labels and edge tuples in
            ``primitives``.

        See Also:
            :class:`ParsedVerticesAndEdges <vertizee.classes.primitives_parsing.ParsedVerticesAndEdges>`
        """
        graph_edges: Set[Edge] = set()
        while len(primitives.edge_tuples) > 0:
            t = primitives.edge_tuples.pop()
            edge = self._get_edge(t)
            if edge is not None:
                graph_edges.add(edge)
        while len(primitives.edge_tuples_weighted) > 0:
            t_weighted = primitives.edge_tuples_weighted.pop()
            edge = self._get_edge(t_weighted)
            if edge is not None:
                graph_edges.add(edge)

        vertex_prev = None
        for vertex_current in primitives.vertex_labels:
            if vertex_prev is not None:
                graph_edges.add(self._get_edge(vertex_prev, vertex_current))
                vertex_prev = None
            else:
                vertex_prev = vertex_current

        return [x for x in graph_edges if x is not None]

    def _get_edge(self, vertex1: VertexType, vertex2: VertexType,) -> Optional[Edge]:
        """Gets the edge specified by the vertices, or None if no such edge exists.

        Args:
            vertex1: The first vertex; in directed graphs this is the *tail*.
            vertex2: The second vertex; in directed graphs, this is the *head*.

        Returns:
            Edge: The edge or None if not found.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        edge_label = edge.create_edge_label(vertex1, vertex2, self._is_directed_graph)
        if edge_label in self._edges:
            return self._edges[edge_label]
        return None

    def get_random_edge(self) -> Optional[Edge]:
        """Returns a randomly selected edge from the graph, or None if there are no edges.

        The random sampling is weighted by edge frequency. The default frequency is 1 and
        increases for each additional parallel edge.

        Returns:
            Edge: The random edge, or None if there are no edges.
        """
        if len(self._edges) > 0:
            sample = random.choices(
                population=list(self._edges_with_freq_weight.keys()),
                weights=list(self._edges_with_freq_weight.values()),
                k=1,
            )
            return sample[0]
        return None

    @property
    def weight(self) -> float:
        """Returns the weight of all edges."""
        weight = 0.0
        for edge in self._edges:
            weight += edge.weight_with_parallel_edges
        return weight

    def has_edge(self, *args: "GraphPrimitive") -> bool:
        """Returns True if the graph contains an edge specified by the graph primitives.

        Returns:
            bool: True if there is a matching edge, otherwise False.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        return self._get_edge(*args) is not None

    def has_vertex(self, vertex: "VertexType") -> bool:
        """Returns True if the graph contains the specified vertex."""
        return self._get_vertex(vertex) is not None

    def is_directed_graph(self) -> bool:
        """Returns True if this is a directed graph (i.e. each edge points from a tail vertex
        to a head vertex)."""
        return self._is_directed_graph

    def is_multigraph(self) -> bool:
        """Returns True if this is a multigraph (i.e. a graph that allows parallel edges)."""
        return self._is_multigraph

    def is_weighted(self) -> bool:
        """Returns True if this is a weighted graph, i.e., contains edges with weights != 1."""
        return self._is_weighted_graph

    def contract_edge(self, vertex1: VertexType, vertex2: VertexType) -> None:
        """Merge ``vertex2`` into ``vertex1``.

        After the merge operation:

           - Incident edges of ``vertex2`` are modified such that ``vertex2`` is replaced by
             ``vertex1``
           - Incident loops on ``vertex2`` become loops on ``vertex1``
           - ``vertex2`` is deleted from the graph
           - :math:`degree(vertex1)` [post-merge]
             :math:`\\Longleftrightarrow degree(vertex1) + degree(vertex2)` [pre-merge]

        Since an Edge's vertices are used in its hash function, they must be treated as immutable
        for the lifetime of the object. Therefore, when ``vertex2`` is deleted, its incident edges
        must also be deleted.

        In some cases, an incident edge of ``vertex2`` will be modified such that by replacing
        ``vertex2`` with ``vertex1``, there exists an edge in the graph matching the new endpoints.
        In this case, the existing edge is updated by incrementing its ``parallel_edge_count`` and
        appending to ``parallel_edge_weights`` as needed.

        If the graph does not contain an edge matching the new endpoints after replacing ``vertex2``
        with ``vertex1``, then a new Edge is added to the graph.
        """
        primitives = parsed_primitives.parse_graph_primitives(vertex1, vertex2)
        if len(primitives.vertex_labels) < 2 or primitives.vertex_labels[1] is None:
            raise ValueError("Must specify two vertices to complete vertex merge operation.")
        if (
            primitives.vertex_labels[0] not in self._vertices
            or primitives.vertex_labels[1] not in self._vertices
        ):
            raise ValueError(
                "Both vertices must be existing members of the graph to complete "
                "a merge operation."
            )

        v1 = self._vertices[primitives.vertex_labels[0]]
        v2 = self._vertices[primitives.vertex_labels[1]]

        edges_to_delete: List[Edge] = []
        # Incident edges of vertex2, where vertex2 is to be replaced by vertex1.
        for edge in v2.incident_edges:
            edges_to_delete.append(edge)
            if edge.vertex1 == v2:
                existing_edge = self._get_edge(v1, edge.vertex2)
            else:  # edge.vertex2 == v2
                existing_edge = self._get_edge(edge.vertex1, v1)

            if existing_edge is not None:
                _merge_parallel_edges(existing_edge, edge)
            else:
                if edge.vertex1 == v2:
                    self.add_edge(
                        v1,
                        edge.vertex2,
                        weight=edge.weight,
                        parallel_edge_count=edge.parallel_edge_count,
                        parallel_edge_weights=edge.parallel_edge_weights,
                    )
                else:
                    self.add_edge(
                        edge.vertex1,
                        v1,
                        weight=edge.weight,
                        parallel_edge_count=edge.parallel_edge_count,
                        parallel_edge_weights=edge.parallel_edge_weights,
                    )

        # Incident loops on vertex2 become loops on vertex1.
        if v2.loops is not None:
            v2_loop_edge = v2.loops
            edges_to_delete.append(v2_loop_edge)
            if v1.loops is None:
                self.add_edge(
                    v1,
                    v1,
                    weight=v2_loop_edge.weight,
                    parallel_edge_count=v2_loop_edge.parallel_edge_count,
                    parallel_edge_weights=v2_loop_edge.parallel_edge_weights,
                )
            else:
                v1_loop_edge = v1.loops
                _merge_parallel_edges(v1_loop_edge, v2_loop_edge)
        # Delete indicated edges after finishing iteration.
        for deleted_edge in edges_to_delete:
            self.remove_all_edges_from(deleted_edge)

        # Delete v2 from the graph.
        self.remove_vertex(v2)

    def remove_all_edges_from(self, *args: "GraphPrimitive") -> int:
        """Deletes all edges (including parallel edges) specified by the graph primitives.

        Args:
            *args: Graph primitives specifying edges to remove.

        Returns:
            int: The number of edges deleted.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        primitives = parsed_primitives.parse_graph_primitives(*args)
        edges_to_remove = self.get_all_graph_edges_from_parsed_primitives(primitives)

        deletion_count = len(edges_to_remove)
        for edge in edges_to_remove:
            deletion_count += edge.multiplicity
            self._edges.remove(edge)
            self._edges_with_freq_weight.pop(edge)
            edge.vertex1._remove_edge(edge)
            edge.vertex2._remove_edge(edge)

        return deletion_count

    def remove_edge_from(self, *args: "GraphPrimitive") -> int:
        """Deletes only one of the edges for each edge specified by the graph primitives.

        For example, if ``args`` specifies vertices :math:`u` and :math:`w`, then if there are two
        parallel edges between :math:`(u, w)`, only one of the parallel edges will be removed. For
        edge objects without parallel edges, this method is the same as
        :meth:`remove_all_edges_from`.

        Args:
            *args: Graph primitives specifying edges from which exactly one edge will be removed.

        Returns:
            int: The number of edges deleted.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        primitives = parsed_primitives.parse_graph_primitives(*args)
        edges_to_remove = self.get_all_graph_edges_from_parsed_primitives(primitives)

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

    def remove_isolated_vertices(self) -> int:
        """Removes all isolated vertices and returns the count of deleted vertices.

        Isolated vertices are vertices that either have zero incident edges or only self-loops.
        """
        vertex_labels_to_remove = []
        for key, vertex in self._vertices.items():
            if vertex.degree == 0:
                vertex_labels_to_remove.append(key)
            elif vertex.loops is not None:
                if len(vertex.incident_edges) == 1:
                    vertex_labels_to_remove.append(key)

        for k in vertex_labels_to_remove:
            self._vertices.pop(k)
        return len(vertex_labels_to_remove)

    def remove_vertex(self, vertex: VertexType) -> None:
        """Removes the indicated vertex.

        For a vertex to be removed, it must not have any incident edges (except self loops). Any
        incident edges must be deleted prior to vertex removal.

        Args:
            vertex: The vertex to remove.
        """
        primitives = parsed_primitives.parse_graph_primitives(vertex)
        if not primitives.vertex_labels:
            raise ValueError("Must specify a valid vertex key or Vertex.")

        lookup_key = primitives.vertex_labels[0]
        if lookup_key in self._vertices:
            graph_vertex = self._vertices[lookup_key]
            if len(graph_vertex.incident_edges) > 0:
                raise ValueError(
                    f"Vertex {{{lookup_key}}} has incident edges. All incident edges (excluding "
                    "loops) must be deleted prior to deleting a vertex."
                )
            self._vertices.pop(lookup_key)

    @property
    def vertex_count(self) -> int:
        """The count of vertices in the graph."""
        return len(self._vertices)

    @property
    def vertices(self) -> Set[Vertex]:
        """The set of graph vertices."""
        return set(self._vertices.values())

    def _add_edge_from_edge(self, edge: Edge) -> Edge:
        """Protected method for adding edges by copying data from an existing Edge object. This
        method exists to simplying making copies of graphs. If the edge object is already a member
        of this graph instance, the edge is returned and no additional edge is added."""
        if edge._parent_graph == self:  # `edge` is already a member of this graph instance.
            return edge

        new_edge = self.add_edge(edge.vertex1, edge.vertex2, edge.weight, **edge.attr)

        if edge.parallel_edge_count > 0 and not self._is_multigraph:
            raise ValueError("parallel edges are only allowed in multigraphs; attempted to add "
                f"parallel edge ({edge.vertex1.label, edge.vertex2.label})")

        new_edge._parallel_edge_count += edge.parallel_edge_count
        new_edge._parallel_edge_weights += edge.parallel_edge_weights
        while len(new_edge.parallel_edge_weights) > new_edge.parallel_edge_count:
            new_edge._parallel_edge_weights.pop()
        self._edges_with_freq_weight[new_edge] = new_edge.multiplicity

    def _get_or_add_vertex(self, vertex_data: _VertexData) -> Vertex:
        """Helper method to get a vertex, or if not found, add a new vertex.

        Args:
            vertex_data: The vertex to get or add.

        Returns:
            Vertex: The vertex that was either found or added.
        """
        if vertex_data.label in self._vertices:
            return self._vertices[vertex_data.label]

        return self.add_vertex([vertex_data.label, vertex_data.attr])

    def _get_vertex(self, vertex: VertexType) -> Optional[Vertex]:
        """Returns the specified vertex or None if not found."""
        primitives = parsed_primitives.parse_graph_primitives(vertex)
        if not primitives.vertex_labels:
            raise ValueError("Must specify a valid vertex label or Vertex object.")
        lookup_key = primitives.vertex_labels[0]
        if lookup_key in self._vertices:
            return self._vertices[lookup_key]
        return None

    def _reverse_graph_into(self, reverse_graph: "GraphBase") -> None:
        """Initializes ``reverse_graph`` with the reverse of this graph (i.e. all directed edges
        pointing in the opposite direction).

        The reverse of a directed graph is also called the transpose or the converse. See
        https://en.wikipedia.org/wiki/Transpose_graph.

        Args:
            reverse_graph: The graph instance to initialize with the reverse of this graph.
        """
        if not self._is_directed_graph or not reverse_graph._is_directed_graph:
            raise exception.GraphTypeNotSupported("only directed graphs may be reversed")
        reverse_graph.clear()

        for vertex in self._vertices.values():
            reverse_graph.add_vertex(copy.deepcopy(vertex))
        for edge in self._edges:
            reverse_graph.add_edge(
                edge.vertex2,
                edge.vertex1,
                weight=edge.weight,
                parallel_edge_count=edge.parallel_edge_count,
                parallel_edge_weights=edge.parallel_edge_weights,
                **(copy.deepcopy(edge.attr))
            )


def _merge_parallel_edges(existing: Edge, new_edge: Edge) -> None:
    """Helper method to merge an edge into an existing parallel edge."""
    existing._parallel_edge_weights += [new_edge.weight] + new_edge.parallel_edge_weights
    existing._parallel_edge_count += new_edge.multiplicity
    for k, v in new_edge.attr.items():
        existing[k] = v
    while len(existing.parallel_edge_weights) > existing.parallel_edge_count:
        existing._parallel_edge_weights.pop()
