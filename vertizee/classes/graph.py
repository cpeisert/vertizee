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

"""Data types supporting directed and undirected graphs, including multigraph variants.

* :class:`Graph` - Undirected graph without parallel edges.
* :class:`DiGraph` - Directed graph without parallel edges.
* :class:`MultiGraph` - Undirected graph that allows parallel edges.
* :class:`MultiDiGraph` - Directed graph that allows parallel edges.

See Also:
    * :class:`Edge <vertizee.classes.edge.Edge>`
    * :class:`DiEdge <vertizee.classes.edge.DiEdge>`
    * :class:`MultiEdge <vertizee.classes.edge.MultiEdge>`
    * :class:`MultiDiEdge <vertizee.classes.edge.MultiDiEdge>`
    * :class:`Vertex <vertizee.classes.vertex.Vertex>`
    * :class:`DiVertex <vertizee.classes.vertex.DiVertex>`
    * :class:`MultiVertex <vertizee.classes.vertex.MultiVertex>`
    * :class:`MultiDiVertex <vertizee.classes.vertex.MultiDiVertex>`
    * :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`

Example:
    >>> import vertizee as vz
    >>> g = vz.Graph()
    >>> edge01 = g.add_edge(0, 1)
    >>> edge12 = g.add_edge(1, 2)
    >>> edge20 = g.add_edge(2, 0)
    >>> g[0].degree
    2
"""
# pylint: disable=attribute-defined-outside-init
# Due to pylint bug. See pylint issue #2981: https://github.com/PyCQA/pylint/issues/2981


# Note: In order to prevent Sphinx from unfolding type aliases, future
# annotations must be imported and type aliases that should not be unfolded must be quoted.
from __future__ import annotations
from abc import ABC, abstractmethod
import collections.abc
import copy
import random
from typing import (
    Dict, Generic, Iterable, Iterator, Optional, overload, Type, ValuesView
)

from vertizee import exception
from vertizee.classes import edge as edge_module
from vertizee.classes import primitives_parsing
from vertizee.classes import vertex as vertex_module

from vertizee.classes.edge import (
    ConnectionKey, DiEdge, E, Edge, EdgeClass, EdgeType, MultiConnection, MultiDiEdge, MultiEdge
)
from vertizee.classes.primitives_parsing import (
    EdgeData, VertexData, GraphPrimitive, ParsedEdgeAndVertexData
)
from vertizee.classes.vertex import (
    DiVertex, MultiDiVertex, MultiVertex, V, Vertex, VertexType
)


def _add_edge_obj_to_graph(graph: G[V, E], edge: EdgeClass) -> E:
    """Adds an edge to ``graph`` by copying data from ``edge``. This function is motivated to
    simplify making graph copies.

    If ``edge`` is a multiedge but ``graph`` is not a multigraph, then the new edge will only
    contain the weight and attributes of the first edge connection returned by
    ``edge.connection_items()``.

    Args:
        graph: The graph to which the new edge is to be added.
        edge: The edge to copy.

    Returns:
        E: The new edge that was added to ``graph``, or an existing edge matching the vertex labels
        of ``edge``.
    """
    if edge.label in graph._edges:
        return edge

    new_edge = None
    if isinstance(edge, MultiConnection):
        for key, connection in edge.connection_items():
            attr = connection.attr if connection.has_attributes_dict() else {}
            if new_edge:
                new_edge.add_connection(weight=connection.weight, key=key, **copy.deepcopy(attr))
            else:
                new_edge = graph.add_edge(
                    edge.vertex1, edge.vertex2, weight=connection.weight, key=key,
                    **copy.deepcopy(attr)
                )
                if not graph.is_multigraph():
                    # Multiedges are not supported, so ignore additional parallel connections.
                    break
    else:
        attr = edge.attr if edge.has_attributes_dict() else {}
        new_edge = graph.add_edge(
            edge.vertex1, edge.vertex2, weight=edge.weight, **copy.deepcopy(attr)
        )
    return new_edge


def _add_edge_to_graph(
    graph: G[V, E], edge_class: Type[E], vertex1: "VertexType", vertex2: "VertexType",
    weight: float, key: Optional[ConnectionKey] = None, **attr
) -> E:
    """Adds a new edge to the graph. If an existing edge matches the vertices, the existing edge is
    returned.

    NOTE:
        This function does not handle adding parallel edge connections to multiedges. If there is no
        connection between a pair of vertices in a multigraph, then this function will create a new
        multiedge with exactly one connection. Additional parallel connections must be handled
        separately.

    Args:
        graph: The graph to which the new edge is to be added.
        edge_class: The class to use to instantiate a new edge object.
        vertex1: The first vertex.
        vertex2: The second vertex.
        weight: The edge weight.
        key: Optional; If ``edge_class`` is an instance of ``MultiConnection``, then the key is
            used within the multiedge to reference the new edge connection.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

    Returns:
        E: The newly added edge, or an existing edge matching the vertices.
    """
    vertex1_data: VertexData = primitives_parsing.parse_vertex_type(vertex1)
    vertex2_data: VertexData = primitives_parsing.parse_vertex_type(vertex2)
    v1_label = vertex1_data.label
    v2_label = vertex2_data.label

    edge_label = edge_module.create_edge_label(v1_label, v2_label, graph.is_directed())
    if edge_label in graph._edges:
        return graph._edges[edge_label]

    if v1_label == v2_label and not graph.allows_self_loops():
        raise exception.SelfLoopsNotAllowed(f"attempted to add loop edge ({v1_label}, {v2_label})")

    if weight != edge_module.DEFAULT_WEIGHT:
        graph._is_weighted_graph = True
    if weight < 0:
        graph._has_negative_edge_weights = True

    v1_obj = graph._add_vertex_from_vertex_data(vertex1_data)
    v2_obj = graph._add_vertex_from_vertex_data(vertex2_data)

    if isinstance(edge_class, MultiConnection):
        if not key:
            key = edge_module.DEFAULT_CONNECTION_KEY
        new_edge = edge_class(v1_obj, v2_obj, weight=weight, key=key, **attr)
    else:
        new_edge = edge_class(v1_obj, v2_obj, weight=weight, **attr)

    graph._edges[edge_label] = new_edge
    # Handle vertex bookkeeping.
    new_edge.vertex1._add_edge(new_edge)
    new_edge.vertex2._add_edge(new_edge)
    return new_edge


def _add_vertex_to_graph(
    graph: G[V, E], vertex_class: Type[V], label: "VertexLabel", **attr
) -> V:
    """Adds a vertex to the graph. If an existing vertex matches the vertex label, the existing
    vertex is returned.

    Args:
        graph: The graph to which the new vertex is to be added.
        vertex_class: The class to use to instantiate a new vertex object.
        label: The label (``str`` or ``int``) to use for the new vertex.
        **attr: Optional; Keyword arguments to add to the vertex ``attr`` dictionary.

    Returns:
        V: The new vertex (or an existing vertex matching the vertex label).
    """
    if not isinstance(label, (str, int)):
        raise TypeError(f"label must be a string or integer; found {type(label).__name__}")
    vertex_label = str(label)

    if vertex_label in graph._vertices:
        return graph._vertices[vertex_label]

    new_vertex = vertex_class(vertex_label, parent_graph=graph, **attr)
    graph._vertices[vertex_label] = new_vertex
    return new_vertex


def _init_graph_from_graph(new_graph: G, other: G) -> None:
    """Initialize a graph using the data from another graph.

    Args:
        new_graph: The new graph to be initialized.
        other: The graph whose data is to be copied.
    """
    for vertex in other._vertices.values():
        attr = vertex._attr if vertex.has_attributes_dict() else {}
        new_graph.add_vertex(vertex.label, **copy.deepcopy(attr))
    for edge in other._edges.values():
        _add_edge_obj_to_graph(new_graph, edge)


class G(ABC, Generic[V, E]):
    """Generic abstract base class from which all graph classes inherit.

    Args:
        allow_self_loops: Indicates if self loops are allowed. A self loop is an edge that
            connects a vertex to itself. Defaults to True.
        is_directed: True indicates that the graph has directed edges. Defaults to False.
        is_multigraph: True indicates that the graph is a multigraph (i.e. there can be multiple
             parallel edges between a pair of vertices). Defaults to False.
        is_weighted_graph: True indicates that the graph is weighted (i.e. there are edges with
            weights other than the default 1.0). Defaults to False.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.
    """

    __slots__ = ("_allow_self_loops", "_attr", "_edges", "_has_negative_edge_weights",
        "__is_directed", "__is_multigraph", "_is_weighted_graph", "_vertices")

    def __init__(self, allow_self_loops: bool = True, is_directed: bool = False,
            is_multigraph: bool = False, is_weighted_graph: bool = False, **attr) -> None:
        self._allow_self_loops = allow_self_loops

        self.__is_directed = is_directed
        self.__is_multigraph = is_multigraph
        self._is_weighted_graph = is_weighted_graph
        """If an edge is added with a weight that is not equal to
        ``vertizee.classes.edge.DEFAULT_WEIGHT``, then this flag is set to True."""

        self._has_negative_edge_weights: bool = False
        """If an edge is added with a negative weight, this flag is set to True."""

        self._edges: Dict[str, E] = dict()
        """A dictionary mapping edge labels to edge objects. See :func:`create_edge_label
        <vertizee.classes.edge.create_edge_label>`."""

        self._vertices: Dict[str, V] = dict()
        """A dictionary mapping vertex labels to vertex objects."""

        self._attr = dict()
        for k, v in attr.items():
            self._attr[k] = v

    def __contains__(self, edge_or_vertex: GraphPrimitive) -> bool:
        data: ParsedEdgeAndVertexData = primitives_parsing.parse_graph_primitive(edge_or_vertex)
        if data.edges:
            return self.has_edge(data.edges[0].vertex1.label, data.edges[0].vertex2.label)
        if data.vertices:
            return data.vertices[0].label in self._vertices

        raise ValueError("expected GraphPrimitive (EdgeType or VertexType); found "
            f"{type(edge_or_vertex).__name__}")

    def __deepcopy__(self, memo) -> "G":
        new = self.__class__()
        new._allow_self_loops = self._allow_self_loops
        new.__is_directed = self.__is_directed
        new.__is_multigraph = self.__is_multigraph
        new._is_weighted_graph = self._is_weighted_graph
        new._has_negative_edge_weights = self._has_negative_edge_weights
        new._attr = copy.deepcopy(self.attr)
        for vertex in self._vertices.values():
            if vertex.has_attributes_dict():
                new.add_vertex(vertex.label, **vertex.attr)
            else:
                new.add_vertex(vertex.label)
        for edge in self._edges.values():
            _add_edge_obj_to_graph(new, edge)
        return new

    @overload
    def __getitem__(self, vertex: VertexType) -> V:
        ...

    @overload
    def __getitem__(self, edge_tuple: EdgeType) -> E:
        ...

    def __getitem__(self, keys):
        """Supports index accessor notation to retrieve vertices and edges.

        Args:
            keys: Usually one vertex (to retrieve a vertex) or two vertices (to retrieve an edge).
                However, any valid ``VertexType`` or ``EdgeType`` may be used.

        Returns:
            Union[EdgeClass, VertexClass, None]: The vertex specified by the vertex label or the
                edge specified by two vertices. If no matching vertex or edge found, returns None.

        Raises:
            IndexError: If ``keys`` is not a valid ``GraphPrimitive`` (that is a ``VertexType`` or
                an ``EdgeType``).
            KeyError: If the graph does not contain a vertex or an edge matching ``keys``.

        Example:
            >>> import vertizee as vz
            >>> g = vz.Graph()
            >>> g.add_edge(1, 2)
            (1, 2)
            >>> g[1]  # __getitem__(1)
            1
            >>> g[1, 2]  # __getitem__((1, 2))
            (1, 2)
        """
        data: ParsedEdgeAndVertexData = primitives_parsing.parse_graph_primitive(keys)
        if data.edges:
            edge_label = edge_module.create_edge_label(
                data.edges[0].vertex1.label, data.edges[0].vertex2.label, self.__is_directed)
            return self._edges[edge_label]
        if data.vertices:
            return self._vertices[data.vertices[0].label]

        raise ValueError("expected GraphPrimitive (EdgeType or VertexType); "
            f"found {type(keys).__name__}")

    def __iter__(self) -> Iterator[Vertex]:
        return iter(self._vertices.values())

    def __len__(self) -> int:
        """Returns the number of vertices in the graph when the built-in Python function ``len`` is
        used."""
        return len(self._vertices)

    @abstractmethod
    def add_edge(
        self, vertex1: "VertexType", vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT, **attr
    ) -> E:
        """Adds a new edge to the graph.

        Args:
            vertex1: The first vertex.
            vertex2: The second vertex.
            weight: Optional; The edge weight. Defaults to ``edge.DEFAULT_WEIGHT`` (1.0).
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

        Returns:
            E: The newly added edge (or pre-existing edge).
        """

    def add_edges_from(self, edges: Iterable["EdgeType"], **attr) -> None:
        """Adds edges from a container.

        Args:
            edges: Sequence of edges to add.
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionaries of each
                edge.

        See Also:
            :mod:`EdgeType <vertizee.classes.edge>`

        Example:
            >>> graph.add_edges_from([(0, 1), (0, 2), (2, 1), (2, 2)])
        """
        if not isinstance(edges, collections.abc.Iterable):
            raise TypeError("edges must be iterable")

        for e in edges:
            edge_data: EdgeData = primitives_parsing.parse_edge_type(e)
            new_edge = self._add_edge_from_edge_data(edge_data)
            for k, v in attr.items():
                new_edge[k] = v

    @abstractmethod
    def add_vertex(self, label: "VertexLabel", **attr) -> V:
        """Adds a vertex to the graph and returns the new vertex object.

        If an existing vertex matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex.
            **attr: Optional; Keyword arguments to add to the vertex ``attr`` dictionary.

        Returns:
            V: The new vertex (or an existing vertex).
        """

    def add_vertices_from(self, vertex_container: Iterable["VertexType"], **attr) -> None:
        """Adds vertices from a container, where the vertices are most often specified as strings
        or integers, but may also be tuples of the form ``Tuple[VertexLabel, AttributesDict]``.

        Args:
            vertex_container: Sequence of vertices to add.
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionaries of each
                vertex.

        Example:
            >>> graph.add_vertices_from([0, 1, 2, 3])

        See Also:
            :mod:`VertexType <vertizee.classes.vertex>`
        """
        if not isinstance(vertex_container, collections.abc.Iterable):
            raise TypeError("vertex_container must be iterable")

        for vertex in vertex_container:
            vertex_data: VertexData = primitives_parsing.parse_vertex_type(vertex)
            new_vertex = self.add_vertex(vertex_data.label, **vertex_data.attr)
            for k, v in attr.items():
                new_vertex[k] = v

    def allows_self_loops(self) -> bool:
        """Returns True if this graph allows self loops, otherwise False."""
        return self._allow_self_loops

    @property
    def attr(self) -> dict:
        """Attribute dictionary to store optional data associated with a graph."""
        return self._attr

    @abstractmethod
    def clear(self) -> None:
        """Removes all edges and vertices from the graph."""

    @abstractmethod
    def deepcopy(self) -> "G":
        """Returns a deep copy of this graph."""

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """The number of edges."""

    @abstractmethod
    def edges(self) -> ValuesView[E]:
        """A view of the graph edges."""
        return self._edges.values()

    @abstractmethod
    def get_random_edge(self, *args, **kwargs) -> Optional[E]:
        """Returns a randomly selected edge from the graph, or None if there are no edges.

        Note that ``*args`` and ``**kwargs`` are part of the method signature in order to provide
        flexibility for subclasses such as :class:`MultiGraph` and :class:`MultiDiGraph`.

        Returns:
            EdgeClass: The random edge, or None if there are no edges.
        """

    def has_edge(self, vertex1: "VertexType", vertex2: "VertexType") -> bool:
        """Returns True if the graph contains the edge.

        Instead of using this method, it is also possible to use the ``in`` operator:

            >>> if ("s", "t") in graph:

        or with objects:

            >>> edge_st = graph.add_edge("s", "t")
            >>> if edge_st in graph:

        Args:
            vertex1: The first vertex of the edge.
            vertex2: The second vertex of the edge.

        Returns:
            bool: True if there is a matching edge in the graph, otherwise False.

        See Also:
            :mod:`EdgeType <vertizee.classes.edge>`
        """
        label = edge_module.create_edge_label(vertex1, vertex2, self.__is_directed)
        return label in self._edges

    def has_negative_edge_weights(self) -> bool:
        """Returns True if the graph contains an edge with a negative weight."""
        return self._has_negative_edge_weights

    def has_vertex(self, vertex: "VertexType") -> bool:
        """Returns True if the graph contains the specified vertex."""
        vertex_data: VertexData = primitives_parsing.parse_vertex_type(vertex)
        return vertex_data.label in self._vertices

    def is_directed(self) -> bool:
        """Returns True if this is a directed graph (i.e. each edge points from a tail vertex
        to a head vertex)."""
        return self.__is_directed

    def is_multigraph(self) -> bool:
        """Returns True if this is a multigraph (i.e. a graph that allows parallel edges)."""
        return self.__is_multigraph

    def is_weighted(self) -> bool:
        """Returns True if this is a weighted graph, i.e., contains edges with weights != 1."""
        return self._is_weighted_graph

    def remove_edge(
        self, vertex1: "VertexType", vertex2: "VertexType", remove_isolated_vertices: bool = False
    ) -> E:
        """Removes an edge from the graph.

        Args:
            vertex1: The first vertex of the edge.
            vertex2: The second vertex of the edge.
            remove_isolated_vertices: If True, then vertices adjacent to ``edge`` that become
                isolated after the edge removal are also removed. Defaults to False.

        Returns:
            E: The edge that was removed.

        Raises:
            EdgeNotFound: If the edge is not in the graph.
        """
        label = edge_module.create_edge_label(vertex1, vertex2, self.__is_directed)
        if not label in self._edges:
            raise exception.EdgeNotFound(f"graph does not have edge {label}")

        edge = self._edges[label]
        self._edges.pop(edge.label)
        edge.vertex1._remove_edge(edge)
        edge.vertex2._remove_edge(edge)
        if remove_isolated_vertices:
            if edge.vertex1.is_isolated():
                edge.vertex1.remove()
            if edge.vertex2.is_isolated():
                edge.vertex2.remove()
        return edge

    def remove_edges_from(self, edges: Iterable["EdgeType"]) -> int:
        """Removes all specified edges.

        This method will fail silently for edges not found in the graph.

        Args:
            edges: A container of edges to remove.

        Returns:
            int: The number of edges deleted.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        deletion_count = 0

        for edge_type in edges:
            try:
                edge = self.__getitem__(edge_type)
                self.remove_edge(edge.vertex1, edge.vertex2)
                if self.__is_multigraph:
                    deletion_count += edge.multiplicity
                else:
                    deletion_count += 1
            except exception.EdgeNotFound:
                pass

        return deletion_count

    def remove_isolated_vertices(self) -> int:
        """Removes all isolated vertices in the graph and returns the deletion count.

        Isolated vertices are vertices that either have zero incident edges or only self-loops.
        """
        vertex_labels_to_remove = []
        for label, vertex in self._vertices.items():
            if vertex.is_isolated():
                vertex_labels_to_remove.append(label)

        for label in vertex_labels_to_remove:
            if self._vertices[label].loop_edge:
                self.remove_edge(self._vertices[label], self._vertices[label])
            self._vertices.pop(label)
        return len(vertex_labels_to_remove)

    def remove_vertex(self, vertex: VertexType) -> None:
        """Removes the indicated vertex.

        For a vertex to be removed, it must be isolated. That means that the vertex has no incident
        edges (except self loops). Any incident edges must be deleted prior to vertex removal.

        Args:
            vertex: The vertex to remove.

        Raises:
            VertizeeException: If the vertex has non-loop incident edges.
            VertexNotFound: If the vertex is not in the graph.
        """
        vertex_data: VertexData = primitives_parsing.parse_vertex_type(vertex)
        label = vertex_data.label
        if label not in self._vertices:
            raise exception.VertexNotFound(f"vertex '{label}' not found")

        vertex_obj = self._vertices[label]
        if not vertex_obj.is_isolated():
            raise exception.VertizeeException(f"cannot remove vertex '{vertex_obj}' due to "
                "adjacent non-loop edges; adjacent edges must be deleted first")

        if vertex_obj.loop_edge:
            vertex_obj.loop_edge.remove()
        self._vertices.pop(label, None)

    @property
    def vertex_count(self) -> int:
        """The count of vertices in the graph."""
        return len(self._vertices)

    @abstractmethod
    def vertices(self) -> ValuesView[V]:
        """A view of the graph vertices."""
        return self._vertices.values()

    @property
    def weight(self) -> float:
        """Returns the weight of all edges."""
        return sum([edge.weight for edge in self._edges.values()])

    def _add_edge_from_edge_data(self, edge_data: EdgeData) -> E:
        """Helper method to a add a new edge from an EdgeData object.

        Args:
            edge_data: The edge to add.

        Returns:
            EdgeClass: The new edge.
        """
        self._add_vertex_from_vertex_data(edge_data.vertex1)
        self._add_vertex_from_vertex_data(edge_data.vertex2)
        return self.add_edge(edge_data.vertex1.label, edge_data.vertex2.label, edge_data.weight,
            **edge_data.attr)

    def _add_vertex_from_vertex_data(self, vertex_data: VertexData) -> V:
        """Helper method to a add a new vertex from a VertexData object.

        Args:
            vertex_data: The vertex to add.

        Returns:
            VertexClass: The new vertex, or an existing vertex matching the specified vertex label.
        """
        if vertex_data.label in self._vertices:
            return self._vertices[vertex_data.label]

        if vertex_data.vertex_object:
            return self.add_vertex(vertex_data.vertex_object)
        return self.add_vertex(vertex_data.label, **vertex_data.attr)

    def _get_edge(self, vertex1: VertexType, vertex2: VertexType) -> Optional[E]:
        """Returns the edge specified by the vertices, or None if no such edge exists.

        Args:
            vertex1: The first vertex (the tail in directed graphs).
            vertex2: The second vertex (the head in directed graphs).

        Returns:
            Edge: The specified edge or None if not found.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self.__is_directed)
        if edge_label in self._edges:
            return self._edges[edge_label]
        return None

    def _get_vertex(self, vertex: VertexType) -> Optional[V]:
        """Returns the specified vertex or None if not found."""
        vertex_data: VertexData = primitives_parsing.parse_vertex_type(vertex)
        if vertex_data.label in self._vertices:
            return self._vertices[vertex_data.label]
        return None


class Graph(G[Vertex, Edge]):
    """An undirected graph without parallel edges.

    Args:
        edges_or_graph: An iterable container of edges or a graph object. If a multigraph is passed,
            only the first connection in each multiedge is added to the new graph.
        allow_self_loops: Indicates if self loops are allowed. A self loop is an edge that
            connects a vertex to itself. Defaults to True.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary of the graph.

    See also:
        * :class:`DiGraph`
        * :class:`MultiGraph`
        * :class:`MultiDiGraph`
    """
    @overload
    def __init__(
        self, edges: Optional[Iterable[EdgeType]] = None,
        allow_self_loops: bool = True,
        **attr
    ) -> None:
        ...

    @overload
    def __init__(
        self, graph: Optional[G] = None,
        allow_self_loops: bool = True,
        **attr
    ) -> None:
        ...

    def __init__(self, edges_or_graph=None, allow_self_loops=True, **attr) -> None:
        super().__init__(allow_self_loops=allow_self_loops, is_directed=False,
            is_multigraph=False, is_weighted_graph=False, **attr)

        if edges_or_graph and isinstance(edges_or_graph, G):
            _init_graph_from_graph(self, edges_or_graph)

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(edges_or_graph)

        elif edges_or_graph:
            raise TypeError(f"edges_or_graph must be None or an instance of G or Iterable;"
                f" found {type(edges_or_graph).__name__}")

    def add_edge(
        self, vertex1: "VertexType", vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        **attr
    ) -> Edge:
        """Adds a new edge to the graph.

        If an existing edge matches the vertices, the existing edge is returned.

        Args:
            vertex1: The first vertex.
            vertex2: The second vertex.
            weight: Optional; The edge weight. Defaults to ``edge.DEFAULT_WEIGHT`` (1.0).
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

        Returns:
            Edge: The newly added edge, or an existing edge with matching vertices.
        """
        return _add_edge_to_graph(self, edge_module._Edge, vertex1=vertex1, vertex2=vertex2,
            weight=weight, **attr)

    def add_vertex(self, label: "VertexLabel", **attr) -> Vertex:
        """Adds a vertex to the graph and returns the new Vertex object. If an existing vertex
        matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex.
            **attr: Optional; Keyword arguments to add to the vertex ``attr`` dictionary.

        Returns:
            Vertex: The new vertex (or an existing vertex matching the vertex label).
        """
        return _add_vertex_to_graph(self, vertex_module._Vertex, label=label, **attr)

    def clear(self) -> None:
        """Removes all edges and vertices from the graph."""
        self._edges.clear()
        self._vertices.clear()

    def deepcopy(self) -> Graph:
        """Returns a deep copy of this graph."""
        return copy.deepcopy(self)

    @property
    def edge_count(self) -> int:
        """The number of edges."""
        return len(self._edges)

    def edges(self) -> ValuesView[Edge]:
        """A view of graph edges."""
        return self._edges.values()

    def get_random_edge(self) -> Optional[Edge]:
        """Returns a randomly selected edge from the graph, or None if there are no edges."""
        if self._edges:
            return random.choice(list(self._edges.values()))
        return None

    def vertices(self) -> ValuesView[Vertex]:
        """A view of the graph vertices."""
        return self._vertices.values()


class DiGraph(G[DiVertex, DiEdge]):
    """A directed graph without parallel edges.

    Args:
        edges_or_graph: An iterable container of directed edges or a graph object. If a multigraph
            is passed, only the first connection in each multiedge is added to the new digraph.
        allow_self_loops: Indicates if self loops are allowed. A self loop is an edge that
            connects a vertex to itself. Defaults to True.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary of the graph.

    See also:
        * :class:`Graph`
        * :class:`MultiGraph`
        * :class:`MultiDiGraph`
    """
    @overload
    def __init__(
        self, edges: Optional[Iterable["EdgeType"]] = None,
        allow_self_loops: bool = True,
        **attr
    ) -> None:
        ...

    @overload
    def __init__(
        self, graph: Optional[G] = None,
        allow_self_loops: bool = True,
        **attr
    ) -> None:
        ...

    def __init__(self, edges_or_graph=None, allow_self_loops=True, **attr) -> None:
        super().__init__(allow_self_loops=allow_self_loops, is_directed=True,
            is_multigraph=False, is_weighted_graph=False, **attr)

        if edges_or_graph and isinstance(edges_or_graph, G):
            _init_graph_from_graph(self, edges_or_graph)

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(edges_or_graph)

        elif edges_or_graph:
            raise TypeError(f"edges_or_graph must be None or an instance of G or Iterable;"
                f" found {type(edges_or_graph).__name__}")

    def add_edge(
        self, tail: "VertexType", head: "VertexType", weight: float = edge_module.DEFAULT_WEIGHT,
        **attr
    ) -> DiEdge:
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
        return _add_edge_to_graph(self, edge_module._DiEdge, vertex1=tail, vertex2=head,
            weight=weight, **attr)

    def add_vertex(self, label: "VertexLabel", **attr) -> DiVertex:
        """Adds a vertex to the graph and returns the new Vertex object. If an existing vertex
        matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex.
            **attr: Optional; Keyword arguments to add to the vertex ``attr`` dictionary.

        Returns:
            DiVertex: The new vertex (or an existing vertex matching the vertex label).
        """
        return _add_vertex_to_graph(self, vertex_module._DiVertex, label=label, **attr)

    def clear(self) -> None:
        """Removes all edges and vertices from the graph."""
        self._edges.clear()
        self._vertices.clear()

    def deepcopy(self) -> DiGraph:
        """Returns a deep copy of this graph."""
        return copy.deepcopy(self)

    @property
    def edge_count(self) -> int:
        """The number of edges."""
        return len(self._edges)

    def edges(self) -> ValuesView[DiEdge]:
        """A view of graph edges."""
        return self._edges.values()

    def get_random_edge(self) -> Optional[DiEdge]:
        """Returns a randomly selected edge from the graph, or None if there are no edges."""
        if self._edges:
            return random.choice(list(self._edges.values()))
        return None

    def vertices(self) -> ValuesView[DiVertex]:
        """A view of the graph vertices."""
        return self._vertices.values()


class MultiGraph(G[MultiVertex, MultiEdge]):
    """An undirected graph that supports multiple parallel connections between each pair of
    vertices.

    Args:
        edges_or_graph: An iterable container of edges or a graph object.
        allow_self_loops: Indicates if self loops are allowed. A self loop is an edge that
            connects a vertex to itself. Defaults to True.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary of the graph.

    See also:
        * :class:`Graph`
        * :class:`DiGraph`
        * :class:`MultiDiGraph`
    """
    @overload
    def __init__(
        self, edges: Optional[Iterable["EdgeType"]] = None,
        allow_self_loops: bool = True,
        **attr
    ) -> None:
        ...

    @overload
    def __init__(
        self, graph: Optional[G] = None,
        allow_self_loops: bool = True,
        **attr
    ) -> None:
        ...

    def __init__(self, edges_or_graph=None, allow_self_loops=True, **attr) -> None:
        super().__init__(allow_self_loops=allow_self_loops, is_directed=False,
            is_multigraph=True, is_weighted_graph=False, **attr)

        if edges_or_graph and isinstance(edges_or_graph, G):
            _init_graph_from_graph(self, edges_or_graph)

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(edges_or_graph)

        elif edges_or_graph:
            raise TypeError(f"edges_or_graph must be None or an instance of G or Iterable;"
                f" found {type(edges_or_graph).__name__}")

    def add_edge(
        self, vertex1: "VertexType", vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT, key: Optional[ConnectionKey] = None, **attr
    ) -> MultiEdge:
        """Adds a new multiedge to the graph.

        If an existing edge matches the vertices, the existing edge is returned.

        Args:
            vertex1: The first vertex.
            vertex2: The second vertex.
            weight: Optional; The edge weight. Defaults to ``edge.DEFAULT_WEIGHT`` (1.0).
            key: Optional; The key to associate with the new connection that distinguishes it from
                other parallel connections in the multiedge.
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

        Returns:
            MultiEdge: The newly added edge, or an existing edge with matching vertices.
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        if edge_label in self._edges:
            multiedge: MultiEdge = self._edges[edge_label]
            multiedge.add_connection(weight=weight, key=key, **attr)
            return multiedge

        return _add_edge_to_graph(self, edge_module._MultiEdge, vertex1=vertex1, vertex2=vertex2,
                                  weight=weight, key=key, **attr)

    def add_vertex(self, label: "VertexLabel", **attr) -> MultiVertex:
        """Adds a vertex to the graph and returns the new Vertex object. If an existing vertex
        matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex.
            **attr: Optional; Keyword arguments to add to the vertex ``attr`` dictionary.

        Returns:
            MultiVertex: The new vertex (or an existing vertex matching the vertex label).
        """
        return _add_vertex_to_graph(self, vertex_module._MultiVertex, label=label, **attr)

    def clear(self) -> None:
        """Removes all edges and vertices from the graph."""
        self._edges.clear()
        self._vertices.clear()

    def deepcopy(self) -> MultiGraph:
        """Returns a deep copy of this graph."""
        return copy.deepcopy(self)

    @property
    def edge_count(self) -> int:
        """The number of edges, including parallel edge connections."""
        return sum(e.multiplicity for e in self._edges.values())

    def edges(self) -> ValuesView[MultiEdge]:
        """A view of the graph edges."""
        return self._edges.values()

    def get_random_edge(self, ignore_multiplicity: bool = False) -> Optional[MultiEdge]:
        """Returns a randomly chosen multiedge from the graph, or None if there are no edges.

        Args:
            ignore_multiplicity: If True, the multiplicity of the multiedges is ignored when
                choosing a random sample. If False, the multiplicity is used as a sample weighting.
                For example, a multiedge with ten parallel connections would be ten times more
                likely to be chosen than a multiedge with only one edge connection. Defaults to
                False.

        Returns:
            MultiEdge: A randomly chosen multiedge from the graph.

        """
        if self._edges:
            if ignore_multiplicity:
                return random.choice(list(self._edges.values()))

            sample = random.choices(
                population=list(self._edges.values()),
                weights=[e.multiplicity for e in self._edges.values()],
                k=1,
            )
            return sample[0]

        return None

    def vertices(self) -> ValuesView[MultiVertex]:
        """A view of the graph vertices."""
        return self._vertices.values()


class MultiDiGraph(G[MultiDiVertex, MultiDiEdge]):
    """A directed graph that supports multiple parallel connections between each pair of
    vertices.

    Args:
        edges_or_graph: An iterable container of edges or a graph object.
        allow_self_loops: Indicates if self loops are allowed. A self loop is an edge that
            connects a vertex to itself. Defaults to True.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary of the graph.

    See also:
        * :class:`Graph`
        * :class:`DiGraph`
        * :class:`MultiGraph`
    """
    @overload
    def __init__(
        self, edges: Optional[Iterable["EdgeType"]] = None,
        allow_self_loops: bool = True,
        **attr
    ) -> None:
        ...

    @overload
    def __init__(
        self, graph: Optional[G] = None,
        allow_self_loops: bool = True,
        **attr
    ) -> None:
        ...

    def __init__(self, edges_or_graph=None, allow_self_loops=True, **attr) -> None:
        super().__init__(allow_self_loops=allow_self_loops, is_directed=True,
            is_multigraph=True, is_weighted_graph=False, **attr)

        if edges_or_graph and isinstance(edges_or_graph, G):
            _init_graph_from_graph(self, edges_or_graph)

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(edges_or_graph)

        elif edges_or_graph:
            raise TypeError(f"edges_or_graph must be None or an instance of G or Iterable;"
                f" found {type(edges_or_graph).__name__}")

    def add_edge(
        self, tail: "VertexType", head: "VertexType", weight: float = edge_module.DEFAULT_WEIGHT,
        key: Optional[ConnectionKey] = None, **attr
    ) -> MultiDiEdge:
        """Adds a new directed multiedge to the graph.

        If an existing edge matches the vertices, the existing edge is returned.

        Args:
            tail: The starting vertex. This is a synonym for ``vertex1``.
            head: The destination vertex to which the ``tail`` points. This is a synonym for
                ``vertex2``.
            weight: Optional; The edge weight. Defaults to ``edge.DEFAULT_WEIGHT`` (1.0).
            key: Optional; The key to associate with the new connection that distinguishes it from
                other parallel connections in the multiedge.
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

        Returns:
            MultiDiEdge: The newly added edge, or an existing edge with matching vertices.
        """
        edge_label = edge_module.create_edge_label(tail, head, self.is_directed())
        if edge_label in self._edges:
            multiedge: MultiDiEdge = self._edges[edge_label]
            multiedge.add_connection(weight=weight, key=key, **attr)
            return multiedge

        return _add_edge_to_graph(self, edge_module._MultiDiEdge, vertex1=tail, vertex2=head,
                                  weight=weight, key=key, **attr)

    def add_vertex(self, label: "VertexLabel", **attr) -> MultiDiVertex:
        """Adds a vertex to the graph and returns the new Vertex object. If an existing vertex
        matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex.
            **attr: Optional; Keyword arguments to add to the vertex ``attr`` dictionary.

        Returns:
            MultiDiVertex: The new vertex (or an existing vertex matching the vertex label).
        """
        return _add_vertex_to_graph(self, vertex_module._MultiDiVertex, label=label, **attr)

    def clear(self) -> None:
        """Removes all multiedges and vertices from the graph."""
        self._edges.clear()
        self._vertices.clear()

    def deepcopy(self) -> MultiDiGraph:
        """Returns a deep copy of this graph."""
        return copy.deepcopy(self)

    @property
    def edge_count(self) -> int:
        """The number of edges, including parallel edge connections."""
        return sum(e.multiplicity for e in self._edges.values())

    def edges(self) -> ValuesView[MultiDiEdge]:
        """A view of the graph multiedges."""
        return self._edges.values()

    def get_random_edge(self, ignore_multiplicity: bool = False) -> Optional[MultiDiEdge]:
        """Returns a randomly chosen multiedge from the graph, or None if there are no edges.

        Args:
            ignore_multiplicity: If True, the multiplicity of the multiedges is ignored when
                choosing a random sample. If False, the multiplicity is used as a sample weighting.
                For example, a multiedge with ten parallel connections would be ten times more
                likely to be chosen than a multiedge with only one edge connection. Defaults to
                False.

        Returns:
            MultiDiEdge: A randomly chosen multiedge from the graph.

        """
        if self._edges:
            if ignore_multiplicity:
                return random.choice(list(self._edges.values()))

            sample = random.choices(
                population=list(self._edges.values()),
                weights=[e.multiplicity for e in self._edges.values()],
                k=1,
            )
            return sample[0]

        return None

    def vertices(self) -> ValuesView[MultiDiVertex]:
        """A view of the graph vertices."""
        return self._vertices.values()
