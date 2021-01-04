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

The graph module defines data types supporting directed and undirected :term:`graphs <graph>`
as well as directed and undirected :term:`multigraphs <multigraph>`.

**Recommended Tutorial**: :doc:`Getting Started <../../tutorials/getting_started>` - |image-colab-getting-started|

.. |image-colab-getting-started| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/cpeisert/vertizee/blob/master/docs/source/tutorials/getting_started.ipynb


All Vertizee :term:`graph` classes derive from abstract base class
:class:`G[V, E] <vertizee.classes.graph.G>`.

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

* :class:`V` - a generic type parameter defined as ``TypeVar("V", bound="VertexBase")``. See
  :class:`VertexBase <vertizee.classes.vertex.VertexBase>`.
* :class:`E` - a generic type parameter defined as
  ``TypeVar("E", bound=Union["Connection", "MultiConnection"])``. See :class:`Connection
  <vertizee.classes.edge.Connection>` and :class:`MultiConnection
  <vertizee.classes.edge.MultiConnection>`.
* :class:`G[V, E] <G>` - Generic abstract base class from which all graph classes derive.
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
# Due to pylint bug. See pylint issue #2981: https://github.com/PyCQA/pylint/issues/2981


# Note: In order to prevent Sphinx from unfolding type aliases, future
# annotations must be imported and type aliases that should not be unfolded must be quoted.
from __future__ import annotations
from abc import ABC, abstractmethod
import collections.abc
import copy
import random
from typing import (
    Any, cast, Dict, Generic, Iterable, Iterator, Optional, overload, Type, TYPE_CHECKING, Union,
    ValuesView
)

from vertizee import exception
from vertizee.classes import edge as edge_module
from vertizee.classes import primitives_parsing
from vertizee.classes import vertex as vertex_module

from vertizee.classes.edge import (
    _Edge, _DiEdge, _MultiEdge, _MultiDiEdge, Attributes,
    ConnectionKey, DiEdge, E, Edge, EdgeBase, EdgeClass, MultiDiEdge, MultiEdge, MultiEdgeBase
)
from vertizee.classes.primitives_parsing import (
    EdgeData, GraphPrimitive, ParsedEdgeAndVertexData, VertexData
)
from vertizee.classes.vertex import (
    _Vertex, _DiVertex, _MultiVertex, _MultiDiVertex,
    DiVertex, MultiDiVertex, MultiVertex, V, Vertex, VertexClass, VertexLabel, VertexType
)

if TYPE_CHECKING:
    from vertizee.classes.edge import EdgeType

ConcreteEdgeClass = Union[_Edge, _DiEdge, _MultiEdge, _MultiDiEdge]
ConcreteVertexClass = Union[_Vertex, _DiVertex, _MultiVertex, _MultiDiVertex]


def _add_edge_obj_to_graph(graph: G[VertexClass, EdgeClass], edge: EdgeClass) -> EdgeClass:
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
    if isinstance(edge, MultiEdgeBase):
        for key, connection in edge.connection_items():
            attr = connection.attr if connection.has_attributes_dict() else {}
            if new_edge:
                assert isinstance(new_edge, MultiEdgeBase)
                new_edge.add_connection(weight=connection.weight, key=key, **copy.deepcopy(attr))
            else:
                new_edge = graph.add_edge(
                    edge.vertex1,
                    edge.vertex2,
                    weight=connection.weight,
                    key=key,
                    **copy.deepcopy(attr)
                )
                if not graph.is_multigraph():
                    # Multiedges are not supported, so ignore additional parallel connections.
                    break
    elif isinstance(edge, (Edge, DiEdge)):
        attr = edge.attr if edge.has_attributes_dict() else {}
        new_edge = graph.add_edge(
            edge.vertex1, edge.vertex2, weight=edge.weight, **copy.deepcopy(attr)
        )
    else:
        raise TypeError("expected one of Edge, DiEdge, MultiEdge, or MultiDiEdge; "
            f"{type(edge).__name__} found")
    assert new_edge is not None
    return new_edge


def _add_edge_to_graph(
    graph: G[VertexClass, EdgeClass],
    edge_class: Type[ConcreteEdgeClass],
    vertex1: "VertexType",
    vertex2: "VertexType",
    weight: float,
    key: Optional[ConnectionKey] = None,
    **attr: Any
) -> EdgeClass:
    """Adds a new edge to the graph. If an existing edge matches the vertices, the existing edge is
    returned.

    NOTE:
        This function does not handle adding parallel edge connections to multiedges. If there is no
        connection between a pair of vertices in a multigraph, then this function will create a new
        multiedge with exactly one connection. Additional parallel connections must be handled
        separately.

    Args:
        graph: The graph to which the new edge is to be added.
        EdgeClass: The class to use to instantiate a new edge object.
        vertex1: The first vertex.
        vertex2: The second vertex.
        weight: The edge weight.
        key: Optional; If ``EdgeClass`` is an instance of ``MultiConnection``, then the key is
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

    # Union[Vertex, DiVertex, MultiVertex, MultiDiVertex]
    v1_obj = graph._add_vertex_from_vertex_data(vertex1_data)
    v2_obj = graph._add_vertex_from_vertex_data(vertex2_data)

    new_edge: Optional[EdgeClass] = None
    if graph.is_multigraph():
        if not key:
            key = edge_module.DEFAULT_CONNECTION_KEY
        assert key is not None
        if issubclass(edge_class, MultiEdge):
            new_edge = edge_class(
                cast(MultiVertex, v1_obj), cast(MultiVertex, v2_obj), weight=weight, key=key,
                **attr
            )
        elif issubclass(edge_class, MultiDiEdge):
            new_edge = edge_class(
                cast(MultiDiVertex, v1_obj), cast(MultiDiVertex, v2_obj), weight=weight, key=key,
                **attr
            )
        else:
            raise TypeError(f"expected MultiEdge or MultiDiEdge; {type(edge_class).__name__} found")
    else:
        if issubclass(edge_class, Edge):
            new_edge = edge_class(cast(Vertex, v1_obj), cast(Vertex, v2_obj), weight=weight, **attr)
        elif issubclass(edge_class, DiEdge):
            new_edge = edge_class(
                cast(DiVertex, v1_obj), cast(DiVertex, v2_obj), weight=weight, **attr)
        else:
            raise TypeError(f"expected Edge or DiEdge; {type(edge_class).__name__} found")

    assert new_edge is not None
    graph._edges[edge_label] = new_edge
    # Handle vertex bookkeeping.
    new_edge.vertex1._add_edge(new_edge)
    new_edge.vertex2._add_edge(new_edge)
    return new_edge


def _add_vertex_to_graph(
    graph: G[VertexClass, EdgeClass],
    vertex_class: Type[ConcreteVertexClass],
    label: VertexLabel,
    **attr: Any
) -> VertexClass:
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


def _init_graph_from_graph(
    new_graph: G[VertexClass, EdgeClass], other: G[VertexClass, EdgeClass]
) -> None:
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


class G(ABC, Generic[V, E]):
    """Generic abstract base class from which all :term:`graph` classes derive.

    This class has generic type parameters ``V`` and ``E``, which enable the type-hint usage
    ``G[V, E]``.

    * ``V = TypeVar("V", bound="VertexBase")`` See :class:`VertexBase
      <vertizee.classes.vertex.VertexBase>`.
    * ``E = TypeVar("E", bound=Union["Connection", "MultiConnection"])`` See
      :class:`Connection <vertizee.classes.edge.Connection>` and
      :class:`MultiConnection <vertizee.classes.edge.MultiConnection>`.

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
        **attr: Any
    ) -> None:
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

        raise TypeError(
            "expected GraphPrimitive (i.e. EdgeType or VertexType) instance; "
            f"{type(edge_or_vertex).__name__} found"
        )

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "G[V, E]":
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
            _add_edge_obj_to_graph(new, edge)  # type: ignore
        return new

    @overload
    def __getitem__(self, vertex: "VertexType") -> V:
        ...

    @overload
    def __getitem__(self, edge: "EdgeType") -> E:
        ...

    def __getitem__(self, keys: Union["VertexType", "EdgeType"]) -> Union[V, E]:
        """Supports index accessor notation to retrieve vertices and edges.

        Args:
            keys: Usually one vertex (to retrieve a vertex) or two vertices (to retrieve an edge).
                However, any valid ``VertexType`` or ``EdgeType`` may be used.

        Returns:
            Union[V, E]: The specified vertex or edge.

        Raises:
            KeyError: If the graph does not contain a vertex or an edge matching ``keys``.
            VertizeeException: If ``keys`` is not a valid ``GraphPrimitive`` (that is a
                ``VertexType`` or an ``EdgeType``).

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
                data.edges[0].vertex1.label, data.edges[0].vertex2.label, self.__is_directed
            )
            return self._edges[edge_label]
        if data.vertices:
            return self._vertices[data.vertices[0].label]

        raise TypeError(
            "expected GraphPrimitive (i.e. EdgeType or VertexType) instance; "
            f"{type(keys).__name__} found"
        )

    def __iter__(self) -> Iterator[V]:
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
        **attr: Any
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
                connection = next(iter(new_edge.connections()))
            else:
                connection = new_edge  # type: ignore
            for k, v in attr.items():
                cast(Attributes, connection).attr[k] = v

    @abstractmethod
    def add_vertex(self, label: "VertexLabel", **attr: Any) -> V:
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
    def deepcopy(self) -> "G[V, E]":
        """Returns a deep copy of this graph."""

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """The number of edges (type ``int``)."""

    @abstractmethod
    def edges(self) -> ValuesView[E]:
        """A view of the graph edges."""
        return self._edges.values()

    @abstractmethod
    def get_random_edge(self, *args: Any, **kwargs: Any) -> Optional[E]:
        """Returns a randomly selected edge from the graph, or None if there are no edges.

        Note that ``*args`` and ``**kwargs`` are part of the method signature in order to provide
        flexibility for subclasses such as :class:`MultiGraph` and :class:`MultiDiGraph`.

        Returns:
            Optional[E]: The random edge, or None if there are no edges.
        """

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
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        remove_semi_isolated_vertices: bool = False,
    ) -> E:
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
        label = edge_module.create_edge_label(vertex1, vertex2, self.__is_directed)
        if not label in self._edges:
            raise exception.EdgeNotFound(f"graph does not have edge {label}")

        edge = self._edges[label]
        self._edges.pop(edge.label)
        edge.vertex1._remove_edge(edge)
        edge.vertex2._remove_edge(edge)
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

        See Also:
            :mod:`EdgeType <vertizee.classes.edge>`
        """
        deletion_count = 0

        for edge_type in edges:
            try:
                edge = self.__getitem__(edge_type)
                self.remove_edge(edge.vertex1, edge.vertex2)
                if isinstance(edge, MultiEdgeBase):
                    deletion_count += edge.multiplicity
                else:
                    deletion_count += 1
            except exception.EdgeNotFound:
                pass

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
            cast(EdgeBase[V], vertex_obj.loop_edge).remove()
        self._vertices.pop(label, None)

    @property
    def vertex_count(self) -> int:
        """The count (type ``int``) of vertices in the graph."""
        return len(self._vertices)

    @abstractmethod
    def vertices(self) -> ValuesView[V]:
        """A view of the graph vertices."""
        return self._vertices.values()

    @property
    def weight(self) -> float:
        """Returns the weight (type ``float``) of all edges."""
        return sum([edge.weight for edge in self._edges.values()])

    def _add_edge_from_edge_data(self, edge_data: EdgeData) -> E:
        """Helper method to a add a new edge from an EdgeData object.

        Args:
            edge_data: The edge to add.

        Returns:
            E: The new edge.
        """
        self._add_vertex_from_vertex_data(edge_data.vertex1)
        self._add_vertex_from_vertex_data(edge_data.vertex2)
        return self.add_edge(
            edge_data.vertex1.label, edge_data.vertex2.label, edge_data.weight, **edge_data.attr
        )

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
            return self.add_vertex(vertex_data.label, **vertex_data.attr)
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
        **attr: Any
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        graph: Optional["G[VertexClass, EdgeClass]"] = None,
        allow_self_loops: bool = True,
        **attr: Any
    ) -> None:
        ...

    def __init__(  # type: ignore
        self,
        edges_or_graph: Optional[Union[Iterable["EdgeType"], "G[VertexClass, EdgeClass]"]] = None,
        allow_self_loops: bool = True,
        **attr: Any
    ) -> None:
        super().__init__(
            allow_self_loops=allow_self_loops,
            is_directed=False,
            is_multigraph=False,
            is_weighted_graph=False,
            **attr
        )

        if edges_or_graph and isinstance(edges_or_graph, G):
            _init_graph_from_graph(
                cast(G[VertexClass, EdgeClass], self),
                cast(G[VertexClass, EdgeClass], edges_or_graph)
            )

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(cast(Iterable["EdgeType"], edges_or_graph))

        elif edges_or_graph:
            raise TypeError(
                f"expected G or Iterable instance; {type(edges_or_graph).__name__} found"
            )

    def add_edge(
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        **attr: Any
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
        return cast(Edge, _add_edge_to_graph(
            cast(G[VertexClass, EdgeClass], self),
            edge_module._Edge,
            vertex1=vertex1,
            vertex2=vertex2,
            weight=weight,
            **attr
        ))

    def add_vertex(self, label: "VertexLabel", **attr: Any) -> "Vertex":
        """Adds a vertex to the graph and returns the new Vertex object. If an existing vertex
        matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex.
            **attr: Optional; Keyword arguments to add to the vertex ``attr`` dictionary.

        Returns:
            Vertex: The new vertex (or an existing vertex matching the vertex label).
        """
        return cast(Vertex, _add_vertex_to_graph(
            cast(G[VertexClass, EdgeClass], self),
            vertex_module._Vertex,
            label=label,
            **attr
        ))

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
        return self._edges.values()

    def get_random_edge(self) -> Optional["Edge"]:  # type: ignore
        """Returns a randomly selected edge from the graph, or None if there are no edges."""
        if self._edges:
            return random.choice(list(self._edges.values()))
        return None

    def vertices(self) -> ValuesView["Vertex"]:
        """A view of the graph vertices."""
        return self._vertices.values()


class DiGraph(G[DiVertex, DiEdge]):
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
        **attr: Any
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        graph: Optional["G[VertexClass, EdgeClass]"] = None,
        allow_self_loops: bool = True,
        **attr: Any
    ) -> None:
        ...

    def __init__(  # type: ignore
        self,
        edges_or_graph: Optional[Union[Iterable["EdgeType"], "G[VertexClass, EdgeClass]"]] = None,
        allow_self_loops: bool = True,
        **attr: Any
    ) -> None:
        super().__init__(
            allow_self_loops=allow_self_loops,
            is_directed=True,
            is_multigraph=False,
            is_weighted_graph=False,
            **attr,
        )

        if edges_or_graph and isinstance(edges_or_graph, G):
            _init_graph_from_graph(
                cast(G[VertexClass, EdgeClass], self),
                cast(G[VertexClass, EdgeClass], edges_or_graph)
            )

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(cast(Iterable["EdgeType"], edges_or_graph))

        elif edges_or_graph:
            raise TypeError(
                f"expected G or Iterable instance; {type(edges_or_graph).__name__} found"
            )

    def add_edge(  # type: ignore
        self,
        tail: "VertexType",
        head: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        **attr: Any
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
        return cast(DiEdge, _add_edge_to_graph(
            cast(G[VertexClass, EdgeClass], self),
            edge_module._DiEdge,
            vertex1=tail,
            vertex2=head,
            weight=weight,
            **attr
        ))

    def add_vertex(self, label: "VertexLabel", **attr: Any) -> "DiVertex":
        """Adds a vertex to the graph and returns the new Vertex object. If an existing vertex
        matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex.
            **attr: Optional; Keyword arguments to add to the vertex ``attr`` dictionary.

        Returns:
            DiVertex: The new vertex (or an existing vertex matching the vertex label).
        """
        return cast(DiVertex, _add_vertex_to_graph(
            cast(G[VertexClass, EdgeClass], self),
            vertex_module._DiVertex,
            label=label,
            **attr
        ))

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
        return self._edges.values()

    def get_random_edge(self) -> Optional["DiEdge"]:  # type: ignore
        """Returns a randomly selected edge from the graph, or None if there are no edges."""
        if self._edges:
            return random.choice(list(self._edges.values()))
        return None

    def vertices(self) -> ValuesView["DiVertex"]:
        """A view of the graph vertices."""
        return self._vertices.values()


class MultiGraph(G[MultiVertex, MultiEdge]):
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
        **attr: Any
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        graph: Optional["G[VertexClass, EdgeClass]"] = None,
        allow_self_loops: bool = True,
        **attr: Any
    ) -> None:
        ...

    def __init__(  # type: ignore
        self,
        edges_or_graph: Optional[Union[Iterable["EdgeType"], "G[VertexClass, EdgeClass]"]] = None,
        allow_self_loops: bool = True,
        **attr: Any
    ) -> None:
        super().__init__(
            allow_self_loops=allow_self_loops,
            is_directed=False,
            is_multigraph=True,
            is_weighted_graph=False,
            **attr,
        )

        if edges_or_graph and isinstance(edges_or_graph, G):
            _init_graph_from_graph(
                cast(G[VertexClass, EdgeClass], self),
                cast(G[VertexClass, EdgeClass], edges_or_graph)
            )

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(cast(Iterable["EdgeType"], edges_or_graph))

        elif edges_or_graph:
            raise TypeError(
                f"expected G or Iterable instance; {type(edges_or_graph).__name__} found"
            )

    def add_edge(
        self,
        vertex1: "VertexType",
        vertex2: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        key: Optional[ConnectionKey] = None,
        **attr: Any
    ) -> "MultiEdge":
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

        return cast(MultiEdge, _add_edge_to_graph(
            cast(G[VertexClass, EdgeClass], self),
            edge_module._MultiEdge,
            vertex1=vertex1,
            vertex2=vertex2,
            weight=weight,
            key=key,
            **attr,
        ))

    def add_vertex(self, label: "VertexLabel", **attr: Any) -> "MultiVertex":
        """Adds a vertex to the graph and returns the new Vertex object. If an existing vertex
        matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex.
            **attr: Optional; Keyword arguments to add to the vertex ``attr`` dictionary.

        Returns:
            MultiVertex: The new vertex (or an existing vertex matching the vertex label).
        """
        return cast(MultiVertex, _add_vertex_to_graph(
            cast(G[VertexClass, EdgeClass], self),
            vertex_module._MultiVertex,
            label=label,
            **attr
        ))

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
        return sum(e.multiplicity for e in self._edges.values())

    def edges(self) -> ValuesView["MultiEdge"]:
        """A view of the graph edges."""
        return self._edges.values()

    def get_random_edge(  # type: ignore
        self, ignore_multiplicity: bool = False
    ) -> Optional["MultiEdge"]:
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

    def vertices(self) -> ValuesView["MultiVertex"]:
        """A view of the graph vertices."""
        return self._vertices.values()


class MultiDiGraph(G[MultiDiVertex, MultiDiEdge]):
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
        **attr: Any
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        graph: Optional["G[VertexClass, EdgeClass]"] = None,
        allow_self_loops: bool = True,
        **attr: Any
    ) -> None:
        ...

    def __init__(  # type: ignore
        self,
        edges_or_graph: Optional[Union[Iterable["EdgeType"], "G[VertexClass, EdgeClass]"]] = None,
        allow_self_loops: bool = True,
        **attr: Any
    ) -> None:
        super().__init__(
            allow_self_loops=allow_self_loops,
            is_directed=True,
            is_multigraph=True,
            is_weighted_graph=False,
            **attr,
        )

        if edges_or_graph and isinstance(edges_or_graph, G):
            _init_graph_from_graph(
                cast(G[VertexClass, EdgeClass], self),
                cast(G[VertexClass, EdgeClass], edges_or_graph)
            )

        elif edges_or_graph and isinstance(edges_or_graph, collections.abc.Iterable):
            self.add_edges_from(cast(Iterable["EdgeType"], edges_or_graph))

        elif edges_or_graph:
            raise TypeError(
                f"expected G or Iterable instance; {type(edges_or_graph).__name__} found"
            )

    def add_edge(  # type: ignore
        self,
        tail: "VertexType",
        head: "VertexType",
        weight: float = edge_module.DEFAULT_WEIGHT,
        key: Optional[ConnectionKey] = None,
        **attr,
    ) -> "MultiDiEdge":
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

        return cast(MultiDiEdge, _add_edge_to_graph(
            cast(G[VertexClass, EdgeClass], self),
            edge_module._MultiDiEdge,
            vertex1=tail,
            vertex2=head,
            weight=weight,
            key=key,
            **attr,
        ))

    def add_vertex(self, label: "VertexLabel", **attr: Any) -> "MultiDiVertex":
        """Adds a vertex to the graph and returns the new Vertex object. If an existing vertex
        matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex.
            **attr: Optional; Keyword arguments to add to the vertex ``attr`` dictionary.

        Returns:
            MultiDiVertex: The new vertex (or an existing vertex matching the vertex label).
        """
        return cast(MultiDiVertex, _add_vertex_to_graph(
            cast(G[VertexClass, EdgeClass], self),
            vertex_module._MultiDiVertex,
            label=label,
            **attr
        ))

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
        return sum(e.multiplicity for e in self._edges.values())

    def edges(self) -> ValuesView["MultiDiEdge"]:
        """A view of the graph multiedges."""
        return self._edges.values()

    def get_random_edge(  # type: ignore
        self, ignore_multiplicity: bool = False
    ) -> Optional["MultiDiEdge"]:
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

    def vertices(self) -> ValuesView["MultiDiVertex"]:
        """A view of the graph vertices."""
        return self._vertices.values()
