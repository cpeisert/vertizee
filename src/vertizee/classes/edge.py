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

"""Data types supporting directed and undirected :term:`edges <edge>` as well as directed and
undirected :term:`multiedges <multiedge>`.

Class and type-alias summary
============================

* :class:`V_co` - a generic covariant type parameter defined as
  ``TypeVar("V_co", bound="VertexBase", covariant=True)``. See
  :class:`VertexBase <vertizee.classes.vertex.VertexBase>`.
* :class:`EdgeType` - A type alias defined as ``Union[EdgeBase, EdgeLiteral]`` where ``EdgeLiteral``
  is an alias for various edge-tuple formats, such as ``Tuple[VertexType, VertexType]`` and
  ``Tuple[VertexType, VertexType, Weight]``. Weight is an alias for ``float``.
* :class:`Attributes` - An abstract class that defines an interface for any class containing an
  ``attr`` dictionary (that is, a dictionary of ad-hoc attributes).
* :class:`EdgeBase[V] <EdgeBase>` - An abstract class defining an immutable :term:`edge` interface.
  All edge classes derive from this class.
* :class:`MutableEdgeBase[V] <MutableEdgeBase>` - Abstract class that defines the mutable
  :term:`edge` interface.
* :class:`Edge(MutableEdgeBase[Vertex]) <Edge>` - An :term:`undirected edge` that does not allow
  :term:`parallel edge` connections between its :term:`endpoints <endpoint>`. This is the class to
  use when adding type hints for :term:`undirected edge` objects.
* :class:`DiEdge(MutableEdgeBase[DiVertex]) <DiEdge>` - A :term:`directed edge` that does not allow
  :term:`parallel edge` connections between its :term:`endpoints <endpoint>`. This is the class to
  use when adding type hints for :term:`directed edge` objects.
* :class:`EdgeConnectionData` - Unique data associated with an individual connection in a
  :term:`multiedge`, such as a weight and custom attributes.
* :class:`EdgeConnectionView(EdgeBase[V]) <EdgeConnectionView>` - A dynamic view of an
  :term:`edge` connection. Connection views provide an edge-like API for each of the
  :term:`parallel edge` connections in a :term:`multiedge`.
* :class:`MultiEdgeBase(MutableEdgeBase[V]) <MultiEdgeBase>` - Abstract base class from which all
  :term:`multiedge` classes inherit.
* :class:`MultiEdge(MultiEdgeBase[MultiVertex]) <MultiEdge>` - An term:`undirected edge
  <undirected edge>` that allows :term:`parallel edge` connections between its
  :term:`endpoints <endpoint>`.
* :class:`MultiDiEdge(MultiEdgeBase[MultiDiVertex]) <MultiDiEdge>` - A directed :term:`multiedge`
  that allows parallel, directed connections between its :term:`endpoints <endpoint>`.

Function summary
================

* :func:`create_edge_label` - Creates a consistent string representation of an :term:`edge`.
  :term:`Directed edges <directed edge>` have labels with the :term:`vertices <vertex>` ordered
  based on the initialization order. :term:`Undirected edges <undirected edge>` have labels with
  the vertices sorted :term:`lexicographically <lexicographical order>`. For example, in an
  :term:`undirected graph` both :math:`(1, 2)` and :math:`(2, 1)` refer to the same undirected edge,
  but the label would always be "(1, 2)".

Detailed documentation
======================
"""

# pylint: disable=cyclic-import
# See pylint issue #3525: https://github.com/PyCQA/pylint/issues/3525

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, cast, Dict, Final, Generic, Iterable, Optional
from typing import Tuple, Type, TYPE_CHECKING, TypeVar, Union

from vertizee.classes import vertex as vertex_module
from vertizee.classes.collection_views import ItemsView, ListView
from vertizee.classes.comparable import Comparable
from vertizee.classes.vertex import (
    DiVertex,
    MultiDiVertex,
    MultiVertex,
    V,
    V_co,
    Vertex,
    VertexBase,
    VertexType,
)
from vertizee.utils import abc_utils

if TYPE_CHECKING:
    from vertizee.classes.graph import GraphBase


# Type aliases
AttributesDict = dict
Weight = float
EdgeTuple = Tuple[VertexType, VertexType]
EdgeTupleWeighted = Tuple[VertexType, VertexType, Weight]
EdgeTupleAttr = Tuple[VertexType, VertexType, AttributesDict]
EdgeTupleWeightedAttr = Tuple[VertexType, VertexType, Weight, AttributesDict]
EdgeLiteral = Union[EdgeTuple, EdgeTupleWeighted, EdgeTupleAttr, EdgeTupleWeightedAttr]

EdgeType = Union["EdgeBase", EdgeLiteral]

#: **E** - A generic edge type parameter.
#: ``E = TypeVar("E", bound=EdgeBase)``
E = TypeVar("E", bound="EdgeBase")  # type: ignore

#: **E_co** - A generic covariant edge type parameter.
#: ``E_co = TypeVar("E_co", bound=EdgeBase, covariant=True)``
E_co = TypeVar("E_co", bound="EdgeBase", covariant=True)  # type: ignore

#: **ME** - A generic multiedge type parameter.
#: ``ME = TypeVar("ME", bound=MultiEdgeBase)``
ME = TypeVar("ME", bound="MultiEdgeBase")  # type: ignore

#: **ME_co** - A generic covariant multiedge type parameter.
#: ``ME_co = TypeVar("ME_co", bound=MultiEdgeBase, covariant=True)``
ME_co = TypeVar("ME_co", bound="MultiEdgeBase", covariant=True)  # type: ignore

ConnectionKey = str

DEFAULT_WEIGHT: Final[float] = 1.0
DEFAULT_CONNECTION_KEY: Final[str] = "0"


def create_edge_label(vertex1: "VertexType", vertex2: "VertexType", is_directed: bool) -> str:
    """Creates a consistent string representation of an :term:`edge`.

    :term:`Directed edges <directed edge>` have labels with the :term:`endpoints <endpoint>` in
    initialization order. :term:`Undirected edges <undirected edge>` have labels with the endpoints
    sorted :term:`lexicographically <lexicographical order>`. For example, in an :term:`undirected
    graph` both :math:`(1, 2)` and :math:`(2, 1)` refer to the same undirected edge, but the label
    would always be "(1, 2)".

    Args:
        vertex1: The first endpoint of the edge.
        vertex2: The second endpoint of the edge.
        is_directed (bool): True indicates a directed edge, False an undirected edge.

    Returns:
        str: The edge label.
    """
    v1_label = vertex_module.get_vertex_label(vertex1)
    v2_label = vertex_module.get_vertex_label(vertex2)

    if not is_directed and v1_label > v2_label:
        return f"({v2_label}, {v1_label})"
    return f"({v1_label}, {v2_label})"


class Attributes(ABC):
    """An interface defining a class that provides an attributes dictionary."""

    __slots__ = ()

    @property
    @abstractmethod
    def attr(self) -> Dict[str, Any]:
        """A dictionary of ad-hoc attributes."""
        return dict()

    @classmethod
    def __subclasshook__(cls, C: Type[object]) -> bool:
        if cls is Attributes:
            return abc_utils.check_methods(C, "attr")
        return NotImplemented


class EdgeBase(ABC, Comparable, Generic[V_co]):
    """Abstract class defining an immutable :term:`edge` interface. All edge classes derive
    from this class.

    This class has a covariant generic type parameter ``V_co``, which supports the type-hint usage
    ``E_co``.

    ``V_co = TypeVar("V_co", bound="VertexBase", covariant=True)`` See :class:`VertexBase
    <vertizee.classes.vertex.VertexBase>`.
    """

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __lt__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Supports adding edges to set-like collections."""

    @abstractmethod
    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are :term:`lexicographically <lexicographical order>` ordered.
        """

    @abstractmethod
    def is_loop(self) -> bool:
        """Returns True if this edge connects a vertex to itself."""

    @property
    @abstractmethod
    def label(self) -> str:
        """A consistent string representation that includes the vertex endpoints. Directed edges
        have labels with the vertices ordered based on the initialization order. Undirected edges
        have labels with vertices :term:`lexicographically <lexicographical order>` sorted to
        ensures consistency. For example, both :math:`(1, 2)` and :math:`(2, 1)` refer to the same
        undirected edge, but the edge label would always be "(1, 2)".
        """

    @property
    @abstractmethod
    def vertex1(self) -> V_co:
        """The first vertex. For DiEdge objects, this is a synonym for the ``tail`` property."""

    @property
    @abstractmethod
    def vertex2(self) -> V_co:
        """The second vertex. For DiEdge objects, this is a synonym for the ``head`` property."""

    @property
    @abstractmethod
    def weight(self) -> float:
        """The weight of the edge. For multiedges, the total weight of all parallel edge
        connections."""

    @property
    @abstractmethod
    def _parent_graph(self) -> GraphBase[V_co, EdgeBase[V_co]]:
        """A reference to the graph that contains this edge."""


class MutableEdgeBase(EdgeBase[V_co], Generic[V_co]):
    """Abstract class that defines the mutable :term:`edge` interface.

    This class has a generic type parameter ``V``, which supports the type-hint usage
    ``MutableEdgeBase[V]``.

    ``V = TypeVar("V", bound="VertexBase")`` See :class:`VertexBase
    <vertizee.classes.vertex.VertexBase>`.

    Args:
        vertex1: The first vertex. In undirected edges, the order of ``vertex1`` and ``vertex2``
            does not matter. For subclasses implementing directed edges, it is recommended to
            add aliases ``tail`` (for ``vertex1``) and ``head`` (for ``vertex2``).
        vertex2: The second vertex.
    """

    __slots__ = ("_label", "_vertex1", "_vertex2")

    def __init__(self, vertex1: V_co, vertex2: V_co) -> None:
        super().__init__()
        self._vertex1 = vertex1
        self._vertex2 = vertex2

        self._label = create_edge_label(
            vertex1, vertex2, is_directed=vertex1._parent_graph.is_directed()
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MutableEdgeBase):
            return _is_edge_equal_to(self, other)
        return NotImplemented  # Delegate equality check to the RHS.

    def __lt__(self, other: object) -> bool:
        """Returns True if ``self`` is less than ``other``."""
        if isinstance(other, MutableEdgeBase):
            return _is_edge_less_than(self, other)
        return NotImplemented  # Delegate equality check to the right-hand side.

    def __hash__(self) -> int:
        """Supports adding edges to set-like collections."""
        return hash(self.label)

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are :term:`lexicographically <lexicographical order>` sorted.
        """

    def contract(self, remove_loops: bool = False) -> None:
        r""":term:`Contracts <contraction>` this edge by removing it from the graph and
        merging its two incident vertices.

        Edge contraction is written as :math:`G/e`, which should not be confused with set
        difference, written as :math:`B \setminus A = \{ x\in B \mid x \notin A \}`.

        For a formal definition, see :term:`contraction`.

        For efficiency, only one of the two incident vertices is actually deleted. After the edge
        contraction:

           - Incident edges of ``vertex2`` are modified such that ``vertex2`` is replaced by
             ``vertex1``
           - Incident loops on ``vertex2`` become loops on ``vertex1``
           - ``vertex2`` is deleted from the graph

        When ``vertex2`` is deleted, its incident edges are also deleted.

        In some cases, an incident edge of ``vertex2`` will be modified such that by replacing
        ``vertex2`` with ``vertex1``, there exists an edge in the graph matching the new endpoints.
        In this case, the existing edge is not modified.

        If the graph does not contain an edge matching the new endpoints after replacing ``vertex2``
        with ``vertex1``, then a new edge object is added to the graph.

        Note that if either ``G._allow_self_loops`` is False or ``remove_loops`` is True,
        self loops will be deleted from the merged vertex (``vertex1``).

        Args:
            remove_loops: If True, loops on the merged vertices will be removed. Defaults to False.
        """
        _contract_edge(self, self._parent_graph, remove_loops)

    def is_loop(self) -> bool:
        """Returns True if this edge connects a vertex to itself."""
        return self._vertex1.label == self._vertex2.label

    @property
    def label(self) -> str:
        """A consistent string representation that includes the vertex endpoints. Directed edges
        have labels with the vertices ordered based on the initialization order. Undirected edges
        have labels with vertices :term:`lexicographically <lexicographical order>` sorted to
        ensures consistency. For example, both :math:`(1, 2)` and :math:`(2, 1)` refer to the same
        undirected edge, but the edge label would always be "(1, 2)".
        """
        return self._label

    def remove(self, remove_semi_isolated_vertices: bool = False) -> None:
        """Removes this edge from the graph.

        Args:
            remove_semi_isolated_vertices: If True, then vertices adjacent to ``edge`` that become
                :term:`semi-isolated` after the edge removal are also removed. Defaults to False.
        """
        self._parent_graph.remove_edge(self.vertex1, self.vertex2, remove_semi_isolated_vertices)

    @property
    def vertex1(self) -> V_co:
        """The first vertex. For DiEdge objects, this is a synonym for the ``tail`` property."""
        return self._vertex1

    @property
    def vertex2(self) -> V_co:
        """The second vertex. For DiEdge objects, this is a synonym for the ``head`` property."""
        return self._vertex2

    @property
    @abstractmethod
    def weight(self) -> float:
        """The weight of the edge. For multiedges, the total weight of all parallel edge
        connections."""

    @property
    def _parent_graph(self) -> GraphBase[V_co, MutableEdgeBase[V_co]]:
        """A reference to the graph that contains this edge."""
        return self._vertex1._parent_graph  # type: ignore


class Edge(MutableEdgeBase[Vertex]):
    """An :term:`undirected edge` that does not allow :term:`parallel edge` connections between
    its :term:`endpoints <endpoint>`.

    To help ensure the integrity of :term:`graphs <graph>`, the ``Edge`` class is abstract and
    cannot be instantiated directly. To create edges, use :meth:`Graph.add_edge
    <vertizee.classes.graph.Graph.add_edge>` and :meth:`Graph.add_edges_from
    <vertizee.classes.graph.G.add_edges_from>`.

    Note:
        In an :term:`undirected graph`, edges :math:`(s, t)` and :math:`(t, s)` represent the same
        edge. Therefore, attempting to add :math:`(s, t)` and :math:`(t, s)` would raise an
        exception, since ``Edge`` objects do not support :term:`parallel edges <parallel edge>`.
        For parallel edge support, see :class:`MultiEdge` and :class:`MultiDiEdge`.

    Args:
        vertex1: The first vertex. In undirected edges, the order of ``vertex1`` and ``vertex2``
            does not matter. For subclasses implementing directed edges, it is recommended to
            add aliases ``tail`` (for ``vertex1``) and ``head`` (for ``vertex2``).
        vertex2: The second vertex.
        weight: Optional; Edge weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.
    """

    __slots__ = ("_attr", "_weight")

    def __init__(
        self, vertex1: "Vertex", vertex2: "Vertex", weight: float = DEFAULT_WEIGHT, **attr: Any
    ) -> None:
        super().__init__(vertex1, vertex2)

        self._attr: Optional[Dict[str, Any]] = None  # Initialized lazily using getter.
        for k, v in attr.items():
            self.attr[k] = v

        self._weight = weight

    def __getitem__(self, key: str) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""
        return self.attr[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Supports index accessor notation to set values in the ``attr`` dictionary."""
        self.attr[key] = value

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge that shows the vertex labels, and for
        weighted graphs, includes the edge weight. The vertices are :term:`lexicographically
        <lexicographical order>` sorted.

        Example:
            "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]
        """

    @property
    def attr(self) -> Dict[str, Any]:
        """Attribute dictionary to store optional data associated with a connection. The attribute
        dictionary is only instantiated as needed, for example, the first time that the ``attr``
        property is accessed. See :meth:`has_attributes_dict`."""
        if not self._attr:
            self._attr = dict()
        return self._attr

    def has_attributes_dict(self) -> bool:
        """Returns True if this connection has an instantiated ``attr`` dictionary. Use this method
        to save memory by avoiding unnecessarily accessing the ``attr`` dictionary. See
        :attr:`attr`."""
        return self._attr is not None

    @property
    def vertex1(self) -> Vertex:
        """The first vertex (type :class:`Vertex <vertizee.classes.vertex.Vertex>`)."""
        return self._vertex1

    @property
    def vertex2(self) -> Vertex:
        """The second vertex (type :class:`Vertex <vertizee.classes.vertex.Vertex>`)."""
        return self._vertex2

    @property
    def weight(self) -> float:
        """The edge weight (type ``float``)."""
        return self._weight

    @property
    def _parent_graph(self) -> GraphBase[Vertex, Edge]:
        """A reference to the graph that contains this edge."""
        return self._vertex1._parent_graph  # type: ignore


class DiEdge(MutableEdgeBase[DiVertex]):
    """A :term:`directed edge` that does not allow :term:`parallel edge` connections between its
    :term:`endpoints <endpoint>`.

    To help ensure the integrity of graphs, the ``DiEdge`` class is abstract and cannot be
    instantiated directly. To create directed edges, use :meth:`DiGraph.add_edge
    <vertizee.classes.graph.DiGraph.add_edge>` and :meth:`DiGraph.add_edges_from
    <vertizee.classes.graph.G.add_edges_from>`.

    Note:
        In a :term:`directed graph`, edge :math:`(s, t)` is distinct from edge :math:`(t, s)`.
        Adding these two edges to a directed graph results in separate ``DiEdge`` objects. However,
        adding the edges to an :term:`undirected graph` (for example, :class:`Graph
        <vertizee.classes.graph.Graph>`), would raise an exception, since :term:`parallel edges
        <parallel edge>` are not allowed.

    Args:
        tail: The :term:`tail` of the directed edge.
        head: The :term:`head` of the directed edge.
        weight: Optional; Edge weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.
    """

    __slots__ = ("_attr", "_weight")

    def __init__(
        self, tail: "DiVertex", head: "DiVertex", weight: float = DEFAULT_WEIGHT, **attr: Any
    ) -> None:
        super().__init__(tail, head)

        self._attr: Optional[Dict[str, Any]] = None  # Initialized lazily using getter.
        for k, v in attr.items():
            self.attr[k] = v

        self._weight = weight

    def __getitem__(self, key: str) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""
        return self.attr[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Supports index accessor notation to set values in the ``attr`` dictionary."""
        self.attr[key] = value

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge that shows the vertex labels, and for
        weighted graphs, includes the edge weight. The vertices are :term:`lexicographically
        <lexicographical order>` ordered.

        Example:
            "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]
        """

    @property
    def attr(self) -> Dict[str, Any]:
        """Attribute dictionary to store optional data associated with a connection. The attribute
        dictionary is only instantiated as needed, for example, the first time that the ``attr``
        property is accessed. See :meth:`has_attributes_dict`."""
        if not self._attr:
            self._attr = dict()
        return self._attr

    def has_attributes_dict(self) -> bool:
        """Returns True if this connection has an instantiated ``attr`` dictionary. Use this method
        to save memory by avoiding unnecessarily accessing the ``attr`` dictionary. See
        :attr:`attr`."""
        return self._attr is not None

    @property
    def vertex1(self) -> DiVertex:
        """The tail vertex (type :class:`DiVertex <vertizee.classes.vertex.DiVertex>`), which is
        the origin of the :term:`directed edge`."""
        return self._vertex1

    @property
    def vertex2(self) -> DiVertex:
        """The head vertex (type :class:`DiVertex <vertizee.classes.vertex.DiVertex>`), which is
        the destination of the :term:`directed edge`."""
        return self._vertex2

    @property
    def weight(self) -> float:
        """The edge weight (type ``float``)."""
        return self._weight

    @property
    def _parent_graph(self) -> GraphBase[DiVertex, DiEdge]:
        """A reference to the graph that contains this edge."""
        return self._vertex1._parent_graph  # type: ignore


class EdgeConnectionData:
    """Unique data associated with an individual connection in a :term:`multiedge`, such as a
    weight and custom attributes.

    Args:
        weight: Optional; Edge weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.
    """

    __slots__ = ("_attr", "weight")

    def __init__(self, weight: float = DEFAULT_WEIGHT, **attr: Any):
        self._attr: Optional[Dict[str, Any]] = None  # Initialized lazily using getter.
        for k, v in attr.items():
            self.attr[k] = v

        self.weight = weight

    def __getitem__(self, key: str) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""
        return self.attr[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Supports index accessor notation to set values in the ``attr`` dictionary."""
        self.attr[key] = value

    @property
    def attr(self) -> Dict[str, Any]:
        """Attribute dictionary to store optional data associated with a connection. The attribute
        dictionary is only instantiated as needed, for example, the first time that the ``attr``
        property is accessed. See :meth:`has_attributes_dict`."""
        if not self._attr:
            self._attr = dict()
        return self._attr

    def has_attributes_dict(self) -> bool:
        """Returns True if this connection has an instantiated ``attr`` dictionary. Use this method
        to save memory by avoiding unnecessarily accessing the ``attr`` dictionary. See
        :attr:`attr`."""
        return self._attr is not None


class EdgeConnectionView(EdgeBase[V_co], Generic[V_co, ME_co]):
    """A dynamic view of an :term:`edge` connection. Connection views provide an edge-like API for
    each of the :term:`parallel edge` connections in a :term:`multiedge` or a :term:`multidiedge`.

    Args:
        multiconnection: A :class:`MultiConnection` object representing parallel edge connections.
        key: The key used to index the edge connection among the potentially multiple parallel
            edges.
    """

    __slots__ = ("_edge_data", "_multiconnection")

    def __init__(self, multiconnection: ME_co, key: "ConnectionKey"):
        self._multiconnection = multiconnection
        self._edge_data: EdgeConnectionData = multiconnection._connections[key]
        self._key: str = key

    def __eq__(self, other: object) -> bool:
        """Returns True if ``self`` is logically equal to ``other``."""
        if isinstance(other, EdgeBase):
            return _is_edge_equal_to(self, other)
        return NotImplemented  # Delegate equality check to the right-hand side.

    def __lt__(self, other: object) -> bool:
        """Returns True if ``self`` is less than ``other``."""
        if isinstance(other, EdgeBase):
            return _is_edge_less_than(self, other)
        return NotImplemented  # Delegate equality check to the right-hand side.

    def __getitem__(self, key: str) -> Any:
        """Supports index accessor notation to retrieve values from the ``attr`` dictionary."""
        return self._edge_data.attr[key]

    def __hash__(self) -> int:
        """Supports adding edges to set-like collections."""
        return self._multiconnection.__hash__()

    def __repr__(self) -> str:
        return self.__str__()

    def __setitem__(self, key: str, value: Any) -> None:
        """Supports index accessor notation to set values in the ``attr`` dictionary."""
        self._edge_data.attr[key] = value

    def __str__(self) -> str:
        return f"{self.__class__.__name__}{self._multiconnection.label}"

    @property
    def attr(self) -> Dict[str, Any]:
        """Attribute dictionary to store optional data associated with a connection. The attribute
        dictionary is only instantiated as needed, for example, the first time that the ``attr``
        property is accessed. See :meth:`has_attributes_dict`."""
        return self._edge_data.attr

    def has_attributes_dict(self) -> bool:
        """Returns True if this connection has an instantiated ``attr`` dictionary. Use this method
        to save memory by avoiding unnecessarily accessing the ``attr`` dictionary. See
        :attr:`attr`."""
        return self._edge_data.has_attributes_dict()

    def is_loop(self) -> bool:
        """Returns True if this edge connects a vertex to itself."""
        return self._multiconnection.is_loop()

    @property
    def key(self) -> "ConnectionKey":
        """The key used to index this parallel edge connection within the :term:`multiedge`."""
        return self._key

    @property
    def label(self) -> str:
        """A consistent string representation."""
        return self._multiconnection.label

    @property
    def vertex1(self) -> V_co:
        """The first vertex (type ``V_co``). For DiEdge objects, this
        is a synonym for the ``tail`` property."""
        return cast(V_co, self._multiconnection.vertex1)

    @property
    def vertex2(self) -> V_co:
        """The second vertex (type ``V_co``). For DiEdge objects, this
        is a synonym for the ``head`` property."""
        return cast(V_co, self._multiconnection.vertex2)

    @property
    def weight(self) -> float:
        """The edge weight (type ``float``)."""
        return self._edge_data.weight

    @property
    def _parent_graph(self) -> GraphBase[V_co, ME_co]:
        """A reference to the graph that contains this edge."""
        return self._multiconnection._parent_graph  # type: ignore


class MultiEdgeBase(MutableEdgeBase[V_co], Generic[V_co]):
    """Abstract base class from which all :term:`multiedge` classes inherit.

    Args:
        vertex1: The first vertex. In undirected multiedges, the order of ``vertex1`` and
            ``vertex2`` does not matter. For subclasses implementing directed edges, it is
            recommended to add aliases ``tail`` (for ``vertex1``) and ``head`` (for ``vertex2``).
        vertex2: The second vertex.
        weight: Optional; Edge weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
        key: Optional; The key to use within the multiedge to reference the new edge connection.
            Each of the potentially multiple parallel connections is assigned a key that is unique
            to the multiedge object. Defaults to ``DEFAULT_CONNECTION_KEY`` (0).
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary of the first
            edge connection in the new multiedge.
    """

    __slots__ = ("_connections", "_weight")

    def __init__(
        self,
        vertex1: V_co,
        vertex2: V_co,
        weight: float = DEFAULT_WEIGHT,
        key: ConnectionKey = DEFAULT_CONNECTION_KEY,
        **attr: Any,
    ) -> None:
        super().__init__(vertex1, vertex2)

        edge_connection = EdgeConnectionData(weight, **attr)
        self._connections: Dict[ConnectionKey, EdgeConnectionData] = dict()
        self._connections[key] = edge_connection

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the :term:`multiedge` showing the vertex labels, and
        for weighted graphs, the edge connection weights.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are :term:`lexicographically <lexicographical order>` ordered.
        """

    @abstractmethod
    def add_connection(
        self, weight: float = DEFAULT_WEIGHT, key: Optional["ConnectionKey"] = None, **attr: Any
    ) -> "EdgeConnectionView[V_co, MultiEdgeBase[V_co]]":
        """Adds a new edge connection to this multiedge. If the connection key already exists, then
        the existing connection key data is replaced with ``weight`` and ``**attr``.

        Args:
            weight: Optional; The connection weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
            key: Optional; The key to associate with the new connection that distinguishes it from
                other parallel connections in the multiedge.
            **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

        Returns:
            EdgeConnectionView[V_co, ME_co]: The newly added edge connection.
        """

    def change_connection_key(self, current_key: "ConnectionKey", new_key: "ConnectionKey") -> None:
        """Changes the dictionary key used to index an edge connection.

        Args:
            current_key: The connection key to be changed.
            new_key: The new connection key.

        Raises:
            KeyError: Raises `KeyError` if `new_key` is already being used.
        """
        if new_key in self._connections:
            raise KeyError(f"new_key '{new_key}' already in use")
        self._connections[new_key] = self._connections.pop(current_key)

    @abstractmethod
    def connections(self) -> "ListView[EdgeConnectionView[V_co, MultiEdgeBase[V_co]]]":
        """Returns a :class:`ListView <vertizee.classes.collection_views.ListView>` of the
        connections in the multiconnection, where each connection is represented as an
        :class:`EdgeConnectionView`."""

    @abstractmethod
    def connection_items(
        self,
    ) -> "ItemsView[ConnectionKey, EdgeConnectionView[V_co, MultiEdgeBase[V_co]]]":
        """Returns an :class:`ItemsView <vertizee.classes.collection_views.ItemsView>`, where each
        item is a tuple containing a connection key and the corresponding connection in the
        multiconnection. Each connection is represented as an :class:`EdgeConnectionView`."""

    @abstractmethod
    def get_connection(
        self, key: "ConnectionKey"
    ) -> "EdgeConnectionView[V_co, MultiEdgeBase[V_co]]":
        """Supports index accessor notation to retrieve a multiedge connection by its key."""

    @property
    def multiplicity(self) -> int:
        """The number (type ``int``) of connections within the multiedge.

        For multiedges without parallel connections, the multiplicity is 1. Each parallel edge adds
        1 to the multiplicity.
        """
        return len(self._connections)

    def remove_connection(self, key: "ConnectionKey") -> None:
        """Removes an edge connection from this multiedge based on its key. If the multiedge only
        has one connection, then removing the connection is the same as calling :meth:`remove`."""
        if key in self._connections:
            if self.multiplicity > 1:
                self._connections.pop(key)
            else:
                self.remove()
        else:
            raise KeyError(key)

    @property
    def weight(self) -> float:
        """The weight (type ``float``) of the multiedge, including parallel connections."""
        return sum(connection.weight for connection in self._connections.values())


class MultiEdge(MultiEdgeBase[MultiVertex]):
    """:term:`Undirected edge <undirected edge>` that allows :term:`parallel edge` connections
    between its :term:`endpoints <endpoint>`.

    To help ensure the integrity of graphs, the ``MultiEdge`` class is abstract and cannot be
    instantiated directly. To create ``MultiEdge`` objects, use :meth:`MultiGraph.add_edge
    <vertizee.classes.graph.MultiGraph.add_edge>` and :meth:`MultiGraph.add_edges_from
    <vertizee.classes.graph.G.add_edges_from>`.

    Note:
        In an undirected :term:`multigraph`, edges :math:`(s, t)` and :math:`(t, s)` represent the
        same :term:`multiedge`. If multiple edges are added with the same vertices, then a single
        ``MultiEdge`` instance is used to store the :term:`parallel edge` connections. When working
        with multiedges, use the :attr:`multiplicity` property to determine if the edge represents
        more than one edge connection.

    Args:
        vertex1: The first vertex. The order of ``vertex1`` and ``vertex2`` does not matter.
        vertex2: The second vertex.
        weight: Optional; Edge weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
        key: Optional; The key to use within the multiedge to reference the new edge connection.
            Each of the potentially multiple parallel connections is assigned a key that is unique
            to the multiedge object. Defaults to ``DEFAULT_CONNECTION_KEY`` (0).
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.
    """

    __slots__ = ()

    def __init__(
        self,
        vertex1: "MultiVertex",
        vertex2: "MultiVertex",
        weight: float = DEFAULT_WEIGHT,
        key: "ConnectionKey" = DEFAULT_CONNECTION_KEY,
        **attr: Any,
    ) -> None:
        super().__init__(vertex1, vertex2, weight=weight, key=key, **attr)

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the :term:`multiedge` showing the vertex labels, and
        for weighted graphs, the edge connection weights.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are :term:`lexicographically <lexicographical order>` ordered.
        """

    def add_connection(
        self, weight: float = DEFAULT_WEIGHT, key: Optional["ConnectionKey"] = None, **attr: Any
    ) -> "EdgeConnectionView[MultiVertex, MultiEdge]":
        return _add_connection(self, weight, key, **attr)

    def connections(self) -> "ListView[EdgeConnectionView[MultiVertex, MultiEdge]]":
        return ListView([EdgeConnectionView(self, key) for key in self._connections])

    def connection_items(
        self,
    ) -> "ItemsView[ConnectionKey, EdgeConnectionView[MultiVertex, MultiEdge]]":
        items = dict()
        for key in self._connections:
            view: EdgeConnectionView[MultiVertex, MultiEdge] = EdgeConnectionView(self, key)
            items[key] = view
        return ItemsView(items)

    def get_connection(self, key: "ConnectionKey") -> "EdgeConnectionView[MultiVertex, MultiEdge]":
        if key in self._connections:
            return EdgeConnectionView(self, key)
        raise KeyError(key)

    @property
    def vertex1(self) -> MultiVertex:
        """The first vertex (type :class:`MultiVertex <vertizee.classes.vertex.MultiVertex>`)."""
        return self._vertex1

    @property
    def vertex2(self) -> MultiVertex:
        """The second vertex (type :class:`MultiVertex <vertizee.classes.vertex.MultiVertex>`)."""
        return self._vertex2


class MultiDiEdge(MultiEdgeBase[MultiDiVertex]):
    """Directed :term:`multiedge` that allows parallel, directed connections between its endpoints.

    To help ensure the integrity of graphs, ``MultiDiEdge`` is abstract and cannot be instantiated
    directly. To create edges, use :meth:`MultiDiGraph.add_edge
    <vertizee.classes.graph.MultiDiGraph.add_edge>` and :meth:`MultiDiGraph.add_edges_from
    <vertizee.classes.graph.G.add_edges_from>`.

    Note:
        In a :term:`directed graph`, edge :math:`(s, t)` is distinct from edge :math:`(t, s)`.
        Adding these two edges to a directed :term:`multigraph` results in separate ``MultiDiEdge``
        objects. However, adding the edges to an undirected multigraph (for example,
        :class:`MultiGraph <vertizee.classes.graph.MultiGraph>`), results in one :class:`MultiEdge`
        object with a :term:`parallel edge` connection.

    Args:
        tail: The :term:`tail` of the directed edge.
        head: The :term:`head` of the directed edge.
        weight: Optional; Edge weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
        key: Optional; The key to use within the multiedge to reference the new edge connection.
            Each of the potentially multiple parallel connections is assigned a key that is unique
            to the multiedge object. Defaults to ``DEFAULT_CONNECTION_KEY`` (0).
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.
    """

    __slots__ = ("_connections",)

    def __init__(
        self,
        tail: "MultiDiVertex",
        head: "MultiDiVertex",
        weight: float = DEFAULT_WEIGHT,
        key: "ConnectionKey" = DEFAULT_CONNECTION_KEY,
        **attr: Any,
    ):
        super().__init__(vertex1=tail, vertex2=head, weight=weight, key=key, **attr)

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the :term:`multiedge` showing the vertex labels, and
        for weighted graphs, the edge connection weights.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are :term:`lexicographically <lexicographical order>` ordered.
        """

    def add_connection(
        self, weight: float = DEFAULT_WEIGHT, key: Optional["ConnectionKey"] = None, **attr: Any
    ) -> "EdgeConnectionView[MultiDiVertex, MultiDiEdge]":
        return _add_connection(self, weight, key, **attr)

    def connections(self) -> "ListView[EdgeConnectionView[MultiDiVertex, MultiDiEdge]]":
        return ListView([EdgeConnectionView(self, key) for key in self._connections])

    def connection_items(
        self,
    ) -> "ItemsView[ConnectionKey, EdgeConnectionView[MultiDiVertex, MultiDiEdge]]":
        items = dict()
        for key in self._connections:
            view: EdgeConnectionView[MultiDiVertex, MultiDiEdge] = EdgeConnectionView(self, key)
            items[key] = view
        return ItemsView(items)

    def get_connection(
        self, key: "ConnectionKey"
    ) -> "EdgeConnectionView[MultiDiVertex, MultiDiEdge]":
        if key in self._connections:
            return EdgeConnectionView(self, key)
        raise KeyError(key)

    @property
    def vertex1(self) -> MultiDiVertex:
        """The tail vertex (type :class:`MultiDiVertex <vertizee.classes.vertex.MultiDiVertex>`),
        which is the origin of the :term:`directed edge`."""
        return self._vertex1

    @property
    def vertex2(self) -> MultiDiVertex:
        """The head vertex (type :class:`MultiDiVertex <vertizee.classes.vertex.MultiDiVertex>`),
        which is the destination of the :term:`directed edge`."""
        return self._vertex2


class _Edge(Edge):
    """Concrete implementation of the abstract :class:`Edge` class."""

    __slots__ = ()

    def __str__(self) -> str:
        return _str_for_edge(self._vertex1, self._vertex2, self._weight)


class _DiEdge(DiEdge):
    """Concrete implementation of the abstract :class:`DiEdge` class."""

    __slots__ = ()

    def __str__(self) -> str:
        return _str_for_edge(self._vertex1, self._vertex2, self._weight)


class _MultiEdge(MultiEdge):
    """Concrete implementation of the abstract :class:`MultiEdge` class."""

    __slots__ = ()

    def __str__(self) -> str:
        return _str_for_multiedge(self._vertex1, self._vertex2, self._connections)


class _MultiDiEdge(MultiDiEdge):
    """Concrete implementation of the abstract :class:`MultiDiEdge` class."""

    __slots__ = ()

    def __str__(self) -> str:
        return _str_for_multiedge(self._vertex1, self._vertex2, self._connections)


def _add_connection(
    edge: ME, weight: float, key: Optional["ConnectionKey"] = None, **attr: Any
) -> EdgeConnectionView[V, ME]:
    """Adds a new edge connection to the multiedge. If the connection key already
    exists, then the existing connection key data is replaced with ``weight`` and ``**attr``.

    Args:
        edge: The multiedge to which a new connection is added.
        weight: Optional; The connection weight. Defaults to ``DEFAULT_WEIGHT`` (1.0).
        key: Optional; The key to associate with the new connection that distinguishes it from
            other parallel connections in the multiedge.
        **attr: Optional; Keyword arguments to add to the ``attr`` dictionary.

    Returns:
        EdgeConnectionView[V]: The newly added edge connection.
    """
    new_key = key
    if key is not None and key in edge._connections:
        edge_data: EdgeConnectionData = edge._connections[key]
        if edge_data.has_attributes_dict():
            assert edge_data._attr is not None
            edge_data._attr.clear()
        for k, v in attr.items():
            edge_data.attr[k] = v
        edge_data.weight = weight
    else:
        edge_data = EdgeConnectionData(weight, **attr)
        if key is None:
            new_key = _create_connection_key(edge._connections.keys())

    if weight != DEFAULT_WEIGHT:
        edge._parent_graph._is_weighted_graph = True
    if weight < 0:
        edge._parent_graph._has_negative_edge_weights = True

    assert new_key is not None
    edge._connections[new_key] = edge_data
    return EdgeConnectionView(edge, new_key)


def _is_edge_equal_to(edge: E, other: E) -> bool:
    """Returns True if ``edge`` is logically equal to ``other``."""
    if edge is other:
        return True
    v1 = edge.vertex1
    v2 = edge.vertex2
    o_v1 = other.vertex1
    o_v2 = other.vertex2

    if not v1._parent_graph.is_directed():
        if v1.label > v2.label:
            v1, v2 = v2, v1
        if o_v1.label > o_v2.label:
            o_v1, o_v2 = o_v2, o_v1

    if v1 != o_v1 or v2 != o_v2 or edge.weight != other.weight:
        return False
    if getattr(edge, "has_attributes_dict", None) and getattr(other, "has_attributes_dict", None):
        if cast(Edge, edge).has_attributes_dict() != cast(Edge, other).has_attributes_dict():
            return False
        if cast(Edge, edge).has_attributes_dict() and cast(Edge, other).has_attributes_dict():
            if cast(Edge, edge).attr != cast(Edge, other).attr:
                return False
    elif getattr(edge, "multiplicity", None) and getattr(other, "multiplicity", None):
        if cast(MultiEdge, edge).multiplicity != cast(MultiEdge, other).multiplicity:
            return False
    return True


def _is_edge_less_than(edge: E, other: E) -> bool:
    """Returns True if ``edge`` is less than ``other``."""
    if edge is other:
        return False
    v1 = edge.vertex1
    v2 = edge.vertex2
    o_v1 = other.vertex1
    o_v2 = other.vertex2

    if not v1._parent_graph.is_directed():
        if v1.label > v2.label:
            v1, v2 = v2, v1
        if o_v1.label > o_v2.label:
            o_v1, o_v2 = o_v2, o_v1

    if v1 < o_v1:
        return True
    if v1 == o_v1 and v2 < o_v2:
        return True
    if v1 == o_v1 and v2 == o_v2 and edge.weight < other.weight:
        return True
    if getattr(edge, "multiplicity", None) and getattr(other, "multiplicity", None):
        if (
            v1 == o_v1
            and v2 == o_v2
            and edge.weight == other.weight
            and cast(MultiEdge, edge).multiplicity < cast(MultiEdge, other).multiplicity
        ):
            return True
    return False


def _contract_edge(edge: E, graph: GraphBase[V, E], remove_loops: bool = False) -> None:
    """Contracts an edge by removing it from the graph and merging its two incident vertices.

    Args:
        edge: The edge to contract.
        graph: The graph to which the edge connection belongs.
        remove_loops: If True, loops on the merged vertex will be removed. Defaults to False.
    """
    v1 = edge.vertex1
    v2 = edge.vertex2

    edges_to_delete = []
    # Incident edges of vertex2, where vertex2 is to be replaced by vertex1.
    for incident in v2.incident_edges():
        edges_to_delete.append(incident)

        if incident.is_loop():
            vertex1 = v1
            vertex2 = v1
        elif incident.vertex1 == v2:
            vertex1 = v1
            vertex2 = incident.vertex2
        else:  # incident.vertex2 == v2
            vertex1 = incident.vertex1
            vertex2 = v1

        if graph.is_multigraph():
            assert isinstance(incident, MultiEdgeBase)
            multiedge: Optional[MultiEdgeBase[VertexBase]] = None
            if graph.has_edge(vertex1, vertex2):
                multiedge = cast(MultiEdgeBase[VertexBase], graph.get_edge(vertex1, vertex2))

            for connection in incident.connections():
                attr = connection.attr if connection.has_attributes_dict() else dict()
                if multiedge:
                    multiedge.add_connection(weight=connection.weight, **attr)
                else:
                    multiedge = cast(
                        MultiEdgeBase[VertexBase],
                        graph.add_edge(vertex1, vertex2, weight=connection.weight, **attr),
                    )
        else:
            assert isinstance(incident, (Edge, DiEdge))
            if not graph.has_edge(vertex1, vertex2):
                graph.add_edge(vertex1, vertex2, weight=incident.weight, **incident.attr)

    # Delete indicated edges after finishing loop iteration.
    for e in edges_to_delete:
        graph.remove_edge(e.vertex1, e.vertex2)
    if remove_loops or not graph.allows_self_loops():
        v1.remove_loops()
    graph.remove_vertex(v2)


def _create_connection_key(connection_keys: Iterable[ConnectionKey]) -> ConnectionKey:
    """Creates a new connection key that does not conflict with a collection of existing keys."""
    numeric_keys = []
    for key in connection_keys:
        try:
            int_key = int(key)
            numeric_keys.append(int_key)
        except ValueError:
            pass

    if numeric_keys:
        max_key = max(numeric_keys)
        new_key = max_key + 1
    else:
        new_key = int(DEFAULT_CONNECTION_KEY) + 1
    return str(new_key)


def _str_for_edge(vertex1: V, vertex2: V, weight: float) -> str:
    """A simple string representation of the edge that shows the vertex labels, and for weighted
    graphs, includes the edge weight."""
    edge_str = f"({vertex1.label}, {vertex2.label}"

    if not vertex1._parent_graph.is_directed():
        if vertex1.label > vertex2.label:
            edge_str = f"({vertex2.label}, {vertex1.label}"

    if vertex1._parent_graph.is_weighted():
        edge_str = f"{edge_str}, {weight})"
    else:
        edge_str = f"{edge_str})"

    return edge_str


def _str_for_multiedge(
    vertex1: V, vertex2: V, connections: Dict[ConnectionKey, EdgeConnectionData]
) -> str:
    """A simple string representation of the multiedge that shows the vertex labels, and for
    weighted graphs, includes the edge weight. The string will show separate vertex tuples for each
    parallel connection, such as "(a, b, 1.0), (a, b, 3.5), (a, b, 2.1)"."""
    connection_str = f"({vertex1.label}, {vertex2.label}"

    if not vertex1._parent_graph.is_directed():
        if vertex1.label > vertex2.label:
            connection_str = f"({vertex2.label}, {vertex1.label}"

    multiconnection_strings = list()
    if vertex1._parent_graph.is_weighted():
        for connection in connections.values():
            multiconnection_strings.append(f"{connection_str}, {connection.weight})")
    else:
        connection_str = f"{connection_str})"
        multiconnection_strings = [connection_str] * len(connections)

    return ", ".join(multiconnection_strings)
