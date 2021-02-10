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

# pylint: disable=line-too-long
"""
================
Graph module
================

This module defines data types supporting directed and undirected :term:`graphs <graph>`
as well as directed and undirected :term:`multigraphs <multigraph>`.

**Recommended Tutorial**: :doc:`Getting Started <../../tutorials/getting_started>` - |image-colab-getting-started|

.. |image-colab-getting-started| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/cpeisert/vertizee/blob/master/docs/source/tutorials/getting_started.ipynb


All Vertizee :term:`graph` classes derive from abstract base class
:class:`GraphBase[V] <vertizee.classes.graph.GraphBase>`.

The most flexible classes are prefixed ``Multi``, since they support :term:`parallel edges
<parallel edge>`. The :class:`DiGraph` and :class:`MultiDiGraph` classes are for graphs with
:term:`directed edges <directed edge>`.

Which graph class should I use?
===============================

+-----------------------------------------+------------+------------------------+
| Class                                   | Type       | Parallel edges allowed |
+=========================================+============+========================+
| :class:`Graph                           | undirected | No                     |
| <vertizee.classes.graph.Graph>`         |            |                        |
+-----------------------------------------+------------+------------------------+
| :class:`MultiGraph                      | undirected | Yes                    |
| <vertizee.classes.graph.MultiGraph>`    |            |                        |
+-----------------------------------------+------------+------------------------+
| :class:`DiGraph                         | directed   | No                     |
| <vertizee.classes.graph.DiGraph>`       |            |                        |
+-----------------------------------------+------------+------------------------+
| :class:`MultiDiGraph                    | directed   | Yes                    |
| <vertizee.classes.graph.MultiDiGraph>`  |            |                        |
+-----------------------------------------+------------+------------------------+

Note:
    All graph classes support :term:`self-loops <loop>`. When initializing a graph, there is an
    option to disable self-loops by setting ``allow_self_loops`` to False.


Class summary
=============

* :class:`V_co` - a generic covariant type parameter defined as
  ``TypeVar("V_co", bound="VertexBase", covariant=True)``. See
  :class:`VertexBase <vertizee.classes.vertex.VertexBase>`.
* :class:`GraphBase[V_co] <GraphBase>` - Generic abstract base class from which all graph classes
  derive.
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
    >>> g.add_edge(0, 1)
    >>> g.add_edge(1, 2)
    >>> g.add_edge(2, 0)
    >>> g[0].degree
    2


Detailed documentation
======================
"""
# pylint: disable=attribute-defined-outside-init

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
    Any,
    cast,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Optional,
    overload,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
    ValuesView,
)

from vertizee import exception
from vertizee.classes import edge as edge_module
from vertizee.classes import primitives_parsing
from vertizee.classes import vertex as vertex_module
from vertizee.classes.collection_views import ListView

from vertizee.classes.edge import (
    _Edge,
    _DiEdge,
    _MultiEdge,
    _MultiDiEdge,
    Attributes,
    ConnectionKey,
    DiEdge,
    Edge,
    EdgeBase,
    EdgeConnectionView,
    MultiDiEdge,
    MultiEdge,
    MultiEdgeBase,
    MutableEdgeBase,
)
from vertizee.classes.primitives_parsing import (
    EdgeData,
    GraphPrimitive,
    ParsedEdgeAndVertexData,
    VertexData,
)
from vertizee.classes.vertex import (
    DiVertex,
    MultiDiVertex,
    MultiVertex,
    V_co,
    Vertex,
    VertexBase,
    VertexLabel,
    VertexType,
)

if TYPE_CHECKING:
    from vertizee.classes.edge import EdgeType

#: **G** - A generic type parameter that represents a graph.
#: ``G = TypeVar("G", bound=GraphBase)``
G = TypeVar("G", "Graph", "DiGraph", "MultiGraph", "MultiDiGraph")


def _add_edge_obj_to_graph(graph: GraphBase[V_co], edge: EdgeBase[V_co]) -> EdgeBase[V_co]:
    """Adds an edge to ``graph`` by copying data from ``edge``. This function is motivated to
    simplify making graph copies.

    If ``edge`` is a multiedge but ``graph`` is not a multigraph, then the new edge will only
    contain the weight and attributes of the first edge connection returned by
    ``edge.connection_items()``.

    Args:
        graph: The graph to which the new edge is to be added.
        edge: The edge to copy.

    Returns:
        EdgeBase[V]: The new edge that was added to ``graph``, or an existing edge matching the
        vertex labels of ``edge``.
    """
    if edge.label in graph._edges:
        return edge

    new_edge = None
    if isinstance(edge, MultiEdgeBase):
        for key, connection in edge.connection_items():
            attr = connection.attr if connection.has_attributes_dict() else {}
            if new_edge:
                assert isinstance(new_edge, MultiEdgeBase)
                new_parallel_connection: EdgeConnectionView[V_co] = new_edge.add_connection(
                    weight=connection.weight, key=key, **copy.deepcopy(attr)
                )
            else:
                new_edge = graph.add_edge(
                    # casts below due to MyPy bug https://github.com/python/mypy/issues/8252
                    edge.vertex1,
                    edge.vertex2,
                    weight=connection.weight,
                    **copy.deepcopy(attr),
                )
                assert isinstance(new_edge, MultiEdgeBase)
                new_parallel_connection = cast(
                    ListView[EdgeConnectionView[V_co]], new_edge.connections
                )[0]

            current_key = new_parallel_connection.key
            if key != current_key:
                new_edge.change_connection_key(current_key=current_key, new_key=key)

    elif isinstance(edge, (Edge, DiEdge)):
        # casts below due to MyPy bug https://github.com/python/mypy/issues/8252
        attr = cast(Edge, edge).attr if cast(Edge, edge).has_attributes_dict() else {}
        new_edge = graph.add_edge(
            cast(Edge, edge).vertex1,
            cast(Edge, edge).vertex2,
            weight=cast(Edge, edge).weight,
            **copy.deepcopy(attr),
        )
    else:
        raise TypeError(
            "expected one of Edge, DiEdge, MultiEdge, or MultiDiEdge; "
            f"{type(edge).__name__} found"
        )
    assert new_edge is not None
    return new_edge


def _add_edge_to_graph(
    graph: GraphBase[V_co],
    edge_class: Type[EdgeBase[V_co]],
    vertex1: "VertexType",
    vertex2: "VertexType",
    weight: float,
    key: Optional[ConnectionKey] = None,
    **attr: Any,
) -> EdgeBase[V_co]:
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
        key: Optional; If ``EdgeBase[V]`` is an instance of ``MultiEdgeBase``, then the key is
            used within the multiedge to reference the new edge connection.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

    Returns:
        EdgeBase[V]: The newly added edge, or an existing edge matching the vertices.
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

    v1_obj: V_co = graph._add_vertex_from_vertex_data(vertex1_data)
    v2_obj: V_co = graph._add_vertex_from_vertex_data(vertex2_data)

    if graph.is_multigraph():
        if not key:
            key = edge_module.DEFAULT_CONNECTION_KEY
        assert key is not None
        if issubclass(edge_class, _MultiEdge):
            assert isinstance(v1_obj, MultiVertex)
            assert isinstance(v2_obj, MultiVertex)
            # `type: ignore` below due to MyPy bug https://github.com/python/mypy/issues/8252
            new_edge = edge_class(v1_obj, v2_obj, weight=weight, key=key, **attr)  # type: ignore
        elif issubclass(edge_class, _MultiDiEdge):
            assert isinstance(v1_obj, MultiDiVertex)
            assert isinstance(v2_obj, MultiDiVertex)
            # `type: ignore` below due to MyPy bug https://github.com/python/mypy/issues/8252
            new_edge = edge_class(v1_obj, v2_obj, weight=weight, key=key, **attr)  # type: ignore
        else:
            raise TypeError(f"expected MultiEdge or MultiDiEdge; {edge_class.__name__} found")
    else:
        if issubclass(edge_class, _Edge):
            assert isinstance(v1_obj, Vertex)
            assert isinstance(v2_obj, Vertex)
            # `type: ignore` below due to MyPy bug https://github.com/python/mypy/issues/8252
            new_edge = edge_class(v1_obj, v2_obj, weight=weight, **attr)  # type: ignore
        elif issubclass(edge_class, _DiEdge):
            assert isinstance(v1_obj, DiVertex)
            assert isinstance(v2_obj, DiVertex)
            # `type: ignore` below due to MyPy bug https://github.com/python/mypy/issues/8252
            new_edge = edge_class(v1_obj, v2_obj, weight=weight, **attr)  # type: ignore
        else:
            raise TypeError(f"expected Edge or DiEdge; {edge_class.__name__} found")

    graph._edges[edge_label] = cast(EdgeBase[V_co], new_edge)
    # Handle vertex bookkeeping.
    new_edge.vertex1._add_edge(cast(EdgeBase[V_co], new_edge))
    new_edge.vertex2._add_edge(cast(EdgeBase[V_co], new_edge))
    return cast(EdgeBase[V_co], new_edge)


def _add_vertex_to_graph(
    graph: GraphBase[V_co], vertex_class: Type[V_co], label: VertexLabel, **attr: Any
) -> V_co:
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
        raise TypeError(f"expected str or int instance; {type(label).__name__} found")
    vertex_label = str(label)

    if vertex_label in graph._vertices:
        return graph._vertices[vertex_label]

    new_vertex = vertex_class(vertex_label, parent_graph=graph, **attr)
    graph._vertices[vertex_label] = new_vertex
    return new_vertex


def _init_graph_from_graph(new_graph: GraphBase[VertexBase], other: GraphBase[VertexBase]) -> None:
    """Initialize a graph using the data from another graph.

    Args:
        new_graph: The new graph to be initialized.
        other: The graph whose data is to be copied.
    """
    for vertex in other._vertices.values():
        attr = vertex._attr if vertex.has_attributes_dict() else {}
        assert isinstance(attr, collections.abc.Mapping)  # MyPy doesn't realize a dict is a mapping
        new_graph.add_vertex(vertex.label, **copy.deepcopy(attr))
    for edge in other._edges.values():
        _add_edge_obj_to_graph(new_graph, edge)


class GraphBase(ABC, Generic[V_co]):
    """Generic abstract base class from which all :term:`graph` classes derive.

    This class has generic type parameter ``V_co``, which enable the type-hint usage
    ``GraphBase[V_co]``.

    * ``V = TypeVar("V", bound="VertexBase")`` See :class:`VertexBase
      <vertizee.classes.vertex.VertexBase>`.

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

    __slots__ = (
        "_allow_self_loops",
        "_attr",
        "_edges",
        "_has_negative_edge_weights",
        "__is_directed",
        "__is_multigraph",
        "_is_weighted_graph",
        "_vertices",
    )

    def __init__(
        self,
        allow_self_loops: bool = True,
        is_directed: bool = False,
        is_multigraph: bool = False,
        is_weighted_graph: bool = False,
        **attr: Any,
    ) -> None:
        self._allow_self_loops = allow_self_loops

        self.__is_directed = is_directed
        self.__is_multigraph = is_multigraph
        self._is_weighted_graph = is_weighted_graph
        """If an edge is added with a weight that is not equal to
        ``vertizee.classes.edge.DEFAULT_WEIGHT``, then this flag is set to True."""

        self._has_negative_edge_weights: bool = False
        """If an edge is added with a negative weight, this flag is set to True."""

        self._edges: Dict[str, EdgeBase[V_co]] = dict()
        """A dictionary mapping edge labels to edge objects. See :func:`create_edge_label
        <vertizee.classes.edge.create_edge_label>`."""

        self._vertices: Dict[str, V_co] = dict()
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

        raise TypeError(
            "expected GraphPrimitive (i.e. EdgeType or VertexType) instance; "
            f"{type(edge_or_vertex).__name__} found"
        )

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "GraphBase[V_co]":
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

    def __getitem__(self, vertex: "VertexType") -> V_co:
        """Supports index accessor notation to retrieve vertices.

        Args:
            vertex: The vertex to retrieve, usually using the vertex label.

        Returns:
            V_co: The specified vertex.

        Raises:
            KeyError: If the graph does not contain a vertex or an edge matching ``keys``.
            VertizeeException: If ``vertex`` is not a valid ``VertexType``.

        Example:
            >>> import vertizee as vz
            >>> g = vz.Graph()
            >>> g.add_edge(1, 2)
            (1, 2)
            >>> g[1]  # __getitem__(1)
            1
        """
        vertex_data: VertexData = primitives_parsing.parse_vertex_type(vertex)
        return self._vertices[vertex_data.label]

    def __iter__(self) -> Iterator[V_co]:
        yield from self._vertices.values()

    def __len__(self) -> int:
        """Returns the number of vertices in the graph when the built-in Python function ``len`` is
        used."""
        return len(self._vertices)

    @abstractmethod
    def add_edge(
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        **attr: Any,
    ) -> EdgeBase[V_co]:
        """Adds a new edge to the graph.

        Args:
            vertex1: The first vertex.
            vertex2: The second vertex.
            weight: Optional; The edge weight. Defaults to ``edge.DEFAULT_WEIGHT`` (1.0).
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

        Returns:
            E: The newly added edge (or pre-existing edge).
        """

    def add_edges_from(self, edges: Iterable["EdgeType"], **attr: Any) -> None:
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
            if isinstance(new_edge, MultiEdgeBase):
                edge = new_edge.connections()[0]
            else:
                edge = new_edge  # type: ignore
            for k, v in attr.items():
                cast(Attributes, edge).attr[k] = v

    @abstractmethod
    def add_vertex(self, label: "VertexLabel", **attr: Any) -> V_co:
        """Adds a vertex to the graph and returns the new vertex object.

        If an existing vertex matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex.
            **attr: Optional; Keyword arguments to add to the vertex ``attr`` dictionary.

        Returns:
            V: The new vertex (or an existing vertex).
        """

    def add_vertices_from(self, vertex_container: Iterable["VertexType"], **attr: Any) -> None:
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
    def attr(self) -> Dict[str, Any]:
        """Attribute dictionary to store optional data associated with a graph."""
        return self._attr

    @abstractmethod
    def clear(self) -> None:
        """Removes all edges and vertices from the graph."""

    @abstractmethod
    def deepcopy(self) -> "GraphBase[V_co]":
        """Returns a deep copy of this graph."""

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """The number of edges (type ``int``)."""

    @abstractmethod
    def edges(self) -> ValuesView[EdgeBase[V_co]]:
        """A view of the graph edges."""
        return self._edges.values()

    @abstractmethod
    def get_edge(self, vertex1: "VertexType", vertex2: "VertexType") -> EdgeBase[V_co]:
        """Returns the :term:`edge` specified by the vertices, or None if no such edge exists.

        Args:
            vertex1: The first vertex (the :term:`tail` in :term:`directed graphs
                <directed graph>`).
            vertex2: The second vertex (the :term:`head` in directed graphs).

        Returns:
            EdgeBase[V]: The specified edge.

        Raises:
            KeyError: If the graph does not contain an edge with the specified vertex endpoints.
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        return self._edges[edge_label]

    @abstractmethod
    def get_random_edge(self) -> Optional[EdgeBase[V_co]]:
        """Returns a randomly selected edge from the graph, or None if there are no edges.

        Returns:
            Optional[EdgeBase[V]]: The random edge, or None if there are no edges.
        """

    def get_vertex(self, vertex: "VertexType") -> V_co:
        """Returns the specified :term:`vertex`.

        Args:
            vertex: The vertex to retrieve, usually using the vertex label.

        Returns:
            V: The specified vertex.

        Raises:
            KeyError: If the graph does not contain a vertex or an edge matching ``keys``.
            VertizeeException: If ``vertex`` is not a valid ``VertexType``.
        """
        vertex_data: VertexData = primitives_parsing.parse_vertex_type(vertex)
        return self._vertices[vertex_data.label]

    def has_edge(self, vertex1: "VertexType", vertex2: "VertexType") -> bool:
        """Returns True if the graph contains the edge.

        Instead of using this method, it is also possible to use the ``in`` operator:

            >>> if ("s", "t") in graph:

        or with objects:

            >>> edge_st = graph.add_edge("s", "t")
            >>> if edge_st in graph:

        Args:
            vertex1: The first endpoint of the edge.
            vertex2: The second endpoint of the edge.

        Returns:
            bool: True if there is a matching edge in the graph, otherwise False.
        """
        label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
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
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        remove_semi_isolated_vertices: bool = False,
    ) -> EdgeBase[V_co]:
        """Removes an edge from the graph.

        Args:
            vertex1: The first endpoint of the edge.
            vertex2: The second endpoint of the edge.
            remove_semi_isolated_vertices: If True, then vertices adjacent to ``edge`` that become
                :term:`semi-isolated` after the edge removal are also removed. Defaults to False.

        Returns:
            E: The edge that was removed.

        Raises:
            EdgeNotFound: If the edge is not in the graph.
        """
        label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        if not label in self._edges:
            raise exception.EdgeNotFound(f"graph does not have edge {label}")

        edge = self._edges[label]
        self._edges.pop(edge.label)
        edge.vertex1._remove_edge(cast(EdgeBase[VertexBase], edge))
        edge.vertex2._remove_edge(cast(EdgeBase[VertexBase], edge))
        if remove_semi_isolated_vertices:
            if edge.vertex1.is_isolated(ignore_self_loops=True):
                edge.vertex1.remove()
            if edge.vertex2.is_isolated(ignore_self_loops=True):
                edge.vertex2.remove()
        return edge

    def remove_edges_from(self, edges: Iterable["EdgeType"]) -> int:
        """Removes all specified edges.

        This method will fail silently for edges not found in the graph.

        Args:
            edges: A container of edges to remove.

        Returns:
            int: The number of edges deleted.

        Raises:
            EdgeNotFound: If one of the edges is not in the graph.

        See Also:
            :mod:`EdgeType <vertizee.classes.edge>`
        """
        deletion_count = 0

        for e in edges:
            edge_data: EdgeData = primitives_parsing.parse_edge_type(e)
            edge = self.remove_edge(edge_data.vertex1.label, edge_data.vertex2.label)
            if isinstance(edge, MultiEdgeBase):
                deletion_count += edge.multiplicity
            else:
                deletion_count += 1

        return deletion_count

    def remove_isolated_vertices(self, ignore_self_loops: bool = False) -> int:
        """Removes all :term:`isolated` vertices from the graph and returns the deletion count.

        Args:
            ignore_self_loops: If True, then self-loops are ignored, meaning that a vertex whose
                only incident edges are self-loops will be considered isolated, and therefore
                removed. Defaults to False.
        """
        vertex_labels_to_remove = []
        for label, vertex in self._vertices.items():
            if vertex.is_isolated(ignore_self_loops):
                vertex_labels_to_remove.append(label)

        for label in vertex_labels_to_remove:
            if self._vertices[label].loop_edge:
                self.remove_edge(self._vertices[label], self._vertices[label])
            self._vertices.pop(label)
        return len(vertex_labels_to_remove)

    def remove_vertex(self, vertex: VertexType) -> None:
        """Removes the indicated vertex.

        For a vertex to be removed, it must be :term:`semi-isolated`. That means that the vertex
        has no incident edges except for self loops. Any non-loop incident edges must be deleted
        prior to vertex removal.

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
        if not vertex_obj.is_isolated(ignore_self_loops=True):
            raise exception.VertizeeException(
                f"cannot remove vertex '{vertex_obj}' due to "
                "adjacent edges; adjacent edges must be deleted first"
            )

        if vertex_obj.loop_edge:
            cast(MutableEdgeBase[V_co], vertex_obj.loop_edge).remove()
        self._vertices.pop(label, None)

    @property
    def vertex_count(self) -> int:
        """The count (type ``int``) of vertices in the graph."""
        return len(self._vertices)

    @abstractmethod
    def vertices(self) -> ValuesView[V_co]:
        """A view of the graph vertices."""
        return self._vertices.values()

    @property
    def weight(self) -> float:
        """Returns the weight (type ``float``) of all edges."""
        return sum([edge.weight for edge in self._edges.values()])

    def _add_edge_from_edge_data(self, edge_data: EdgeData) -> EdgeBase[V_co]:
        """Helper method to a add a new edge from an EdgeData object.

        Args:
            edge_data: The edge to add.

        Returns:
            EdgeBase[V]: The new edge.
        """
        self._add_vertex_from_vertex_data(edge_data.vertex1)
        self._add_vertex_from_vertex_data(edge_data.vertex2)
        return self.add_edge(
            edge_data.vertex1.label, edge_data.vertex2.label, edge_data.weight, **edge_data.attr
        )

    def _add_vertex_from_vertex_data(self, vertex_data: VertexData) -> V_co:
        """Helper method to a add a new vertex from a VertexData object.

        Args:
            vertex_data: The vertex to add.

        Returns:
            V: The new vertex, or an existing vertex matching the specified vertex label.
        """
        if vertex_data.label in self._vertices:
            return self._vertices[vertex_data.label]

        if vertex_data.vertex_object:
            return self.add_vertex(vertex_data.label, **vertex_data.attr)
        return self.add_vertex(vertex_data.label, **vertex_data.attr)


class Graph(GraphBase[Vertex]):
    """An :term:`undirected graph` without :term:`parallel edges <parallel edge>`.

    The ``Graph`` class contains :term:`vertices <vertex>` of type :class:`Vertex
    <vertizee.classes.vertex.Vertex>` and :term:`edges <edge>` of type :class:`Edge
    <vertizee.classes.edge.Edge>`.

    Args:
        edges_or_graph: An iterable container of edges or a graph object. If a multigraph is passed,
            only the first connection in each multiedge is added to the new graph.
        allow_self_loops: Indicates if self loops are allowed. A self loop is an edge that
            connects a vertex to itself. Defaults to True.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary of the graph.

    See also:
        * :mod:`EdgeType <vertizee.classes.edge>`
        * :class:`DiGraph`
        * :class:`MultiGraph`
        * :class:`MultiDiGraph`
    """

    @overload
    def __init__(
        self,
        edges: Optional[Iterable["EdgeType"]] = None,
        allow_self_loops: bool = True,
        **attr: Any,
    ) -> None:
        ...

    @overload
    def __init__(
        self, graph: Optional["GraphBase[V_co]"] = None, allow_self_loops: bool = True, **attr: Any
    ) -> None:
        ...

    def __init__(  # type: ignore
        self,
        edges_or_graph: Optional[Union[Iterable["EdgeType"], "GraphBase[V_co]"]] = None,
        allow_self_loops: bool = True,
        **attr: Any,
    ) -> None:
        super().__init__(
            allow_self_loops=allow_self_loops,
            is_directed=False,
            is_multigraph=False,
            is_weighted_graph=False,
            **attr,
        )

        if edges_or_graph and isinstance(edges_or_graph, GraphBase):
            _init_graph_from_graph(
                cast(GraphBase[VertexBase], self), cast(GraphBase[VertexBase], edges_or_graph)
            )

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(cast(Iterable["EdgeType"], edges_or_graph))

        elif edges_or_graph:
            raise TypeError(
                f"expected GraphBase or Iterable instance; {type(edges_or_graph).__name__} found"
            )

    def add_edge(
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        **attr: Any,
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
        return cast(
            Edge,
            _add_edge_to_graph(
                self, edge_module._Edge, vertex1=vertex1, vertex2=vertex2, weight=weight, **attr
            ),
        )

    def add_vertex(self, label: "VertexLabel", **attr: Any) -> "Vertex":
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

    def deepcopy(self) -> "Graph":
        """Returns a deep copy of this graph."""
        return copy.deepcopy(self)

    @property
    def edge_count(self) -> int:
        """The number of edges."""
        return len(self._edges)

    def edges(self) -> ValuesView["Edge"]:
        """A view of graph edges."""
        return cast(ValuesView[Edge], self._edges.values())

    def get_edge(self, vertex1: "VertexType", vertex2: "VertexType") -> Edge:
        """Returns the :term:`edge` specified by the vertices, or None if no such edge exists.

        Args:
            vertex1: The first vertex.
            vertex2: The second vertex.

        Returns:
            Edge: The specified edge.

        Raises:
            KeyError: If the graph does not contain an edge with the specified vertex endpoints.
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        return cast(Edge, self._edges[edge_label])

    def get_random_edge(self) -> Optional["Edge"]:
        """Returns a randomly selected edge from the graph, or None if there are no edges."""
        if self._edges:
            return cast(Edge, random.choice(list(self._edges.values())))
        return None

    def vertices(self) -> ValuesView["Vertex"]:
        """A view of the graph vertices."""
        return self._vertices.values()


class DiGraph(GraphBase[DiVertex]):
    """A :term:`digraph` is a graph with :term:`directed edges <directed edge>`;
    :term:`parallel edges <parallel edge>` are not allowed.

    The ``DiGraph`` class contains :term:`vertices <vertex>` of type :class:`DiVertex
    <vertizee.classes.vertex.DiVertex>` and :term:`directed edges <directed edge>` of type
    :class:`DiEdge <vertizee.classes.edge.DiEdge>`.

    Args:
        edges_or_graph: An iterable container of directed edges or a graph object. If a multigraph
            is passed, only the first connection in each multiedge is added to the new digraph.
        allow_self_loops: Indicates if self loops are allowed. A self loop is an edge that
            connects a vertex to itself. Defaults to True.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary of the graph.

    See also:
        * :mod:`EdgeType <vertizee.classes.edge>`
        * :class:`Graph`
        * :class:`MultiGraph`
        * :class:`MultiDiGraph`
    """

    @overload
    def __init__(
        self,
        edges: Optional[Iterable["EdgeType"]] = None,
        allow_self_loops: bool = True,
        **attr: Any,
    ) -> None:
        ...

    @overload
    def __init__(
        self, graph: Optional["GraphBase[V_co]"] = None, allow_self_loops: bool = True, **attr: Any
    ) -> None:
        ...

    def __init__(  # type: ignore
        self,
        edges_or_graph: Optional[Union[Iterable["EdgeType"], "GraphBase[V_co]"]] = None,
        allow_self_loops: bool = True,
        **attr: Any,
    ) -> None:
        super().__init__(
            allow_self_loops=allow_self_loops,
            is_directed=True,
            is_multigraph=False,
            is_weighted_graph=False,
            **attr,
        )

        if edges_or_graph and isinstance(edges_or_graph, GraphBase):
            _init_graph_from_graph(
                cast(GraphBase[VertexBase], self), cast(GraphBase[VertexBase], edges_or_graph)
            )

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(cast(Iterable["EdgeType"], edges_or_graph))

        elif edges_or_graph:
            raise TypeError(
                f"expected GraphBase or Iterable instance; {type(edges_or_graph).__name__} found"
            )

    def add_edge(
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        **attr: Any,
    ) -> "DiEdge":
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
        return cast(
            DiEdge,
            _add_edge_to_graph(
                self, edge_module._DiEdge, vertex1=vertex1, vertex2=vertex2, weight=weight, **attr
            ),
        )

    def add_vertex(self, label: "VertexLabel", **attr: Any) -> "DiVertex":
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

    def deepcopy(self) -> "DiGraph":
        """Returns a deep copy of this graph."""
        return copy.deepcopy(self)

    @property
    def edge_count(self) -> int:
        """The number of edges."""
        return len(self._edges)

    def edges(self) -> ValuesView["DiEdge"]:
        """A view of graph edges."""
        return cast(ValuesView[DiEdge], self._edges.values())

    def get_edge(self, vertex1: "VertexType", vertex2: "VertexType") -> "DiEdge":
        """Returns the :term:`diedge` specified by the vertices, or None if no such edge exists.

        Args:
            vertex1: The first vertex (the :term:`tail` in :term:`directed graphs
                <directed graph>`).
            vertex2: The second vertex (the :term:`head` in directed graphs).

        Returns:
            DiEdge: The specified edge.

        Raises:
            KeyError: If the graph does not contain an edge with the specified vertex endpoints.
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        return cast(DiEdge, self._edges[edge_label])

    def get_random_edge(self) -> Optional["DiEdge"]:
        """Returns a randomly selected edge from the graph, or None if there are no edges."""
        if self._edges:
            return cast(DiEdge, random.choice(list(self._edges.values())))
        return None

    def vertices(self) -> ValuesView["DiVertex"]:
        """A view of the graph vertices."""
        return self._vertices.values()


class MultiGraph(GraphBase[MultiVertex]):
    """A :term:`multigraph` is an :term:`undirected graph` that supports :term:`parallel edge`
    connections between each pair of :term:`vertices <vertex>`, as well as multiple
    :term:`self-loops <loop>` on a single vertex.

    The ``MultiGraph`` class contains :term:`vertices <vertex>` of type :class:`MultiVertex
    <vertizee.classes.vertex.MultiVertex>` and :term:`undirected edges <undirected edge>` of type
    :class:`MultiEdge <vertizee.classes.edge.MultiEdge>`.

    Args:
        edges_or_graph: An iterable container of edges or a graph object.
        allow_self_loops: Indicates if self loops are allowed. A self loop is an edge that
            connects a vertex to itself. Defaults to True.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary of the graph.

    See also:
        * :mod:`EdgeType <vertizee.classes.edge>`
        * :class:`Graph`
        * :class:`DiGraph`
        * :class:`MultiDiGraph`
    """

    @overload
    def __init__(
        self,
        edges: Optional[Iterable["EdgeType"]] = None,
        allow_self_loops: bool = True,
        **attr: Any,
    ) -> None:
        ...

    @overload
    def __init__(
        self, graph: Optional["GraphBase[V_co]"] = None, allow_self_loops: bool = True, **attr: Any
    ) -> None:
        ...

    def __init__(  # type: ignore
        self,
        edges_or_graph: Optional[Union[Iterable["EdgeType"], "GraphBase[V_co]"]] = None,
        allow_self_loops: bool = True,
        **attr: Any,
    ) -> None:
        super().__init__(
            allow_self_loops=allow_self_loops,
            is_directed=False,
            is_multigraph=True,
            is_weighted_graph=False,
            **attr,
        )

        if edges_or_graph and isinstance(edges_or_graph, GraphBase):
            _init_graph_from_graph(
                cast(GraphBase[VertexBase], self), cast(GraphBase[VertexBase], edges_or_graph)
            )

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(cast(Iterable["EdgeType"], edges_or_graph))

        elif edges_or_graph:
            raise TypeError(
                f"expected GraphBase or Iterable instance; {type(edges_or_graph).__name__} found"
            )

    def add_edge(
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        **attr: Any,
    ) -> "MultiEdge":
        """Adds a new :term:`multiedge` to the graph.

        If an existing edge matches the vertices, the existing edge is returned. To assign a custom
        key to the new edge connection, use :meth:`add_edge_with_key`. To add a parallel edge
        connection to an existing multiedge, see :meth:`MultiEdgeBase.add_connection
        <vertizee.classes.edge.MultiEdgeBase.add_connection>`.

        Args:
            vertex1: The first vertex.
            vertex2: The second vertex.
            weight: Optional; The edge weight. Defaults to ``edge.DEFAULT_WEIGHT`` (1.0).
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

        Returns:
            MultiEdge: The newly added edge, or an existing edge with matching vertices.
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        if edge_label in self._edges:
            multiedge: MultiEdge = cast(MultiEdge, self._edges[edge_label])
            multiedge.add_connection(weight=weight, **attr)
            return multiedge

        return cast(
            MultiEdge,
            _add_edge_to_graph(
                self,
                edge_module._MultiEdge,
                vertex1=vertex1,
                vertex2=vertex2,
                weight=weight,
                **attr,
            ),
        )

    def add_edge_with_key(
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        key: ConnectionKey = edge_module.DEFAULT_CONNECTION_KEY,
        **attr: Any,
    ) -> "MultiEdge":
        """Adds a new :term:`multiedge` to the graph and provides the option to set a the `key`
        used to index the new edge connection (among possibly many parallel edges).

        If an existing edge matches the vertices, the existing edge is returned. To add a parallel
        edge connection to an existing multiedge, see :meth:`MultiEdgeBase.add_connection
        <vertizee.classes.edge.MultiEdgeBase.add_connection>`.

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
            multiedge: MultiEdge = cast(MultiEdge, self._edges[edge_label])
            multiedge.add_connection(weight=weight, key=key, **attr)
            return multiedge

        return cast(
            MultiEdge,
            _add_edge_to_graph(
                self,
                edge_module._MultiEdge,
                vertex1=vertex1,
                vertex2=vertex2,
                weight=weight,
                key=key,
                **attr,
            ),
        )

    def add_vertex(self, label: "VertexLabel", **attr: Any) -> "MultiVertex":
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

    def deepcopy(self) -> "MultiGraph":
        """Returns a deep copy of this graph."""
        return copy.deepcopy(self)

    @property
    def edge_count(self) -> int:
        """The number of edges, including parallel edge connections."""
        return sum(e.multiplicity for e in cast(ValuesView[MultiEdge], self._edges.values()))

    def edges(self) -> ValuesView["MultiEdge"]:
        """A view of the graph edges."""
        return cast(ValuesView[MultiEdge], self._edges.values())

    def get_edge(self, vertex1: "VertexType", vertex2: "VertexType") -> "MultiEdge":
        """Returns the :term:`multiedge` specified by the vertices, or None if no such edge exists.

        Args:
            vertex1: The first vertex.
            vertex2: The second vertex.

        Returns:
            MultiEdge: The specified edge.

        Raises:
            KeyError: If the graph does not contain an edge with the specified vertex endpoints.
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        return cast(MultiEdge, self._edges[edge_label])

    def get_random_edge(self) -> Optional["MultiEdge"]:
        """Returns a randomly chosen multiedge from the graph, or None if there are no edges.

        Note:
            This method ignores the multiplicity of the multiedges. Hence, a multiedge with ten
            parallel edge connections is equally likely to be chosen as a multiedge with only one
            edge connection.

        Returns:
            Optional[MultiEdge]: A randomly chosen multiedge from the graph.

        See Also:
            :meth:`get_random_edge_weighted_by_multiplicity`
        """
        if self._edges:
            return cast(MultiEdge, random.choice(list(self._edges.values())))
        return None

    def get_random_edge_weighted_by_multiplicity(self) -> Optional["MultiEdge"]:
        """Returns a randomly chosen multiedge from the graph, where the multiplicity of each
        multiedge is used as a sample weighting. For example, a multiedge with ten parallel
        edge connections would be ten times more likely to be chosen than a multiedge with only
        one edge connection.

        Returns:
            Optional[MultiEdge]: A randomly chosen multiedge from the graph, where the edges are
            weighted by the multiplicity.

        See Also:
            :meth:`get_random_edge`
        """
        if self._edges:
            sample_weights = []
            for e in self._edges.values():
                sample_weights.append(cast(MultiEdge, e).multiplicity)

            sample = random.choices(
                population=list(self._edges.values()), weights=sample_weights, k=1
            )
            return cast(MultiEdge, sample[0])

        return None

    def vertices(self) -> ValuesView["MultiVertex"]:
        """A view of the graph vertices."""
        return self._vertices.values()


class MultiDiGraph(GraphBase[MultiDiVertex]):
    """A directed :term:`multigraph` that supports :term:`parallel edge` connections between each
    pair of :term:`vertices <vertex>`.

    The ``MultiDiGraph`` class contains :term:`vertices <vertex>` of type :class:`MultiDiVertex
    <vertizee.classes.vertex.MultiDiVertex>` and :term:`directed edges <directed edge>` of type
    :class:`MultiDiEdge <vertizee.classes.edge.MultiDiEdge>`.

    Args:
        edges_or_graph: An iterable container of edges or a graph object.
        allow_self_loops: Indicates if self loops are allowed. A self loop is an edge that
            connects a vertex to itself. Defaults to True.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary of the graph.

    See also:
        * :mod:`EdgeType <vertizee.classes.edge>`
        * :class:`Graph`
        * :class:`DiGraph`
        * :class:`MultiGraph`
    """

    @overload
    def __init__(
        self,
        edges: Optional[Iterable["EdgeType"]] = None,
        allow_self_loops: bool = True,
        **attr: Any,
    ) -> None:
        ...

    @overload
    def __init__(
        self, graph: Optional["GraphBase[V_co]"] = None, allow_self_loops: bool = True, **attr: Any
    ) -> None:
        ...

    def __init__(  # type: ignore
        self,
        edges_or_graph: Optional[Union[Iterable["EdgeType"], "GraphBase[V_co]"]] = None,
        allow_self_loops: bool = True,
        **attr: Any,
    ) -> None:
        super().__init__(
            allow_self_loops=allow_self_loops,
            is_directed=True,
            is_multigraph=True,
            is_weighted_graph=False,
            **attr,
        )

        if edges_or_graph and isinstance(edges_or_graph, GraphBase):
            _init_graph_from_graph(
                cast(GraphBase[VertexBase], self), cast(GraphBase[VertexBase], edges_or_graph)
            )

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(cast(Iterable["EdgeType"], edges_or_graph))

        elif edges_or_graph:
            raise TypeError(
                f"expected GraphBase or Iterable instance; {type(edges_or_graph).__name__} found"
            )

    def add_edge(
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        **attr: Any,
    ) -> "MultiDiEdge":
        """Adds a new :term:`multidiedge` to the graph.

        If an existing edge matches the vertices, the existing edge is returned. To assign a custom
        key to the new edge connection, use :meth:`add_edge_with_key`. To add a parallel edge
        connection to an existing multiedge, see :meth:`MultiEdgeBase.add_connection
        <vertizee.classes.edge.MultiEdgeBase.add_connection>`.

        Args:
            vertex1: The first vertex.
            vertex2: The second vertex.
            weight: Optional; The edge weight. Defaults to ``edge.DEFAULT_WEIGHT`` (1.0).
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

        Returns:
            MultiDiEdge: The newly added edge, or an existing edge with matching vertices.
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        if edge_label in self._edges:
            multiedge: MultiDiEdge = cast(MultiDiEdge, self._edges[edge_label])
            multiedge.add_connection(weight=weight, **attr)
            return multiedge

        return cast(
            MultiDiEdge,
            _add_edge_to_graph(
                self,
                edge_module._MultiDiEdge,
                vertex1=vertex1,
                vertex2=vertex2,
                weight=weight,
                **attr,
            ),
        )

    def add_edge_with_key(
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        key: ConnectionKey = edge_module.DEFAULT_CONNECTION_KEY,
        **attr: Any,
    ) -> "MultiDiEdge":
        """Adds a new :term:`multidiedge` to the graph and provides the option to set a the `key`
        used to index the new edge connection (among possibly many parallel edges).

        If an existing edge matches the vertices, the existing edge is returned. To add a parallel
        edge connection to an existing multidiedge, see :meth:`MultiEdgeBase.add_connection
        <vertizee.classes.edge.MultiEdgeBase.add_connection>`.

        Args:
            vertex1: The first vertex.
            vertex2: The second vertex.
            weight: Optional; The edge weight. Defaults to ``edge.DEFAULT_WEIGHT`` (1.0).
            key: Optional; The key to associate with the new connection that distinguishes it from
                other parallel connections in the multidiedge.
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

        Returns:
            MultiDiEdge: The newly added edge, or an existing edge with matching vertices.
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        if edge_label in self._edges:
            multiedge: MultiDiEdge = cast(MultiDiEdge, self._edges[edge_label])
            multiedge.add_connection(weight=weight, key=key, **attr)
            return multiedge

        return cast(
            MultiDiEdge,
            _add_edge_to_graph(
                self,
                edge_module._MultiDiEdge,
                vertex1=vertex1,
                vertex2=vertex2,
                weight=weight,
                key=key,
                **attr,
            ),
        )

    def add_vertex(self, label: "VertexLabel", **attr: Any) -> "MultiDiVertex":
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

    def deepcopy(self) -> "MultiDiGraph":
        """Returns a deep copy of this graph."""
        return copy.deepcopy(self)

    @property
    def edge_count(self) -> int:
        """The number of edges, including parallel edge connections."""
        return sum(e.multiplicity for e in cast(ValuesView[MultiDiEdge], self._edges.values()))

    def edges(self) -> ValuesView["MultiDiEdge"]:
        """A view of the graph multiedges."""
        return cast(ValuesView[MultiDiEdge], self._edges.values())

    def get_edge(self, vertex1: "VertexType", vertex2: "VertexType") -> "MultiDiEdge":
        """Returns the :term:`multidiedge` specified by the vertices, or None if no such edge
        exists.

        Args:
            vertex1: The first vertex (the :term:`tail` in :term:`directed graphs
                <directed graph>`).
            vertex2: The second vertex (the :term:`head` in directed graphs).

        Returns:
            MultiDiEdge: The specified edge.

        Raises:
            KeyError: If the graph does not contain an edge with the specified vertex endpoints.
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        return cast(MultiDiEdge, self._edges[edge_label])

    def get_random_edge(self) -> Optional["MultiDiEdge"]:
        """Returns a randomly chosen multiedge from the graph, or None if there are no edges.

        Note:
            This method ignores the multiplicity of the multiedges. Hence, a multiedge with ten
            parallel edge connections is equally likely to be chosen as a multiedge with only one
            edge connection.

        Returns:
            Optional[MultiDiEdge]: A randomly chosen multiedge from the graph.

        See Also:
            :meth:`get_random_edge_weighted_by_multiplicity`
        """
        if self._edges:
            return cast(MultiDiEdge, random.choice(list(self._edges.values())))
        return None

    def get_random_edge_weighted_by_multiplicity(self) -> Optional["MultiDiEdge"]:
        """Returns a randomly chosen multiedge from the graph, where the multiplicity of each
        multiedge is used as a sample weighting. For example, a multiedge with ten parallel
        edge connections would be ten times more likely to be chosen than a multiedge with only
        one edge connection.

        Returns:
            Optional[MultiDiEdge]: A randomly chosen multiedge from the graph, where the edges are
            weighted by the multiplicity.

        See Also:
            :meth:`get_random_edge`
        """
        if self._edges:
            sample_weights = []
            for e in self._edges.values():
                sample_weights.append(cast(MultiEdge, e).multiplicity)

            sample = random.choices(
                population=list(self._edges.values()), weights=sample_weights, k=1
            )
            return cast(MultiDiEdge, sample[0])

        return None

    def vertices(self) -> ValuesView["MultiDiVertex"]:
        """A view of the graph vertices."""
        return self._vertices.values()
