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

"""Data types supporting directed and undirected graph edges.

* :class:`DiEdge` - A directed connection between two vertices that
  defines the ``tail`` as the starting vertex and the ``head`` as the destination vertex. Parallel
  connections are not allowed.
* :class:`Edge` - An undirected connection between two vertices. The order of the vertices does not
  matter. However, the string representation will always show vertices sorted in lexicographic
  order. Parallel connections are not allowed.
* :class:`EdgeType` - A type alias defined as
  Union[DiEdge, Edge, EdgeLiteral, MultiDiEdge, MultiEdge] and where EdgeLiteral is an alias for
  various edge-tuple formats.
* :class:`MultiDiEdge` - A directed connection between two vertices that defines the ``tail`` as
  the starting vertex and the ``head`` as the destination vertex. Multi-edges support parallel
  connections.
* :class:`MultiEdge` - An undirected connection between two vertices. The order of the vertices
  does not matter. However, the string representation will always show vertices sorted in
  lexicographic order. Multi-edges support parallel connections.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, TYPE_CHECKING, Union

from vertizee.utils import abc_utils

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.graph_base import GraphBase
    from vertizee.classes.vertex import Vertex, VertexType

# Type aliases
AttributesDict = dict
Weight = float
EdgeTuple = Tuple["VertexType", "VertexType"]
EdgeTupleWeighted = Tuple["VertexType", "VertexType", Weight]
EdgeTupleAttr = Tuple["VertexType", "VertexType", AttributesDict]
EdgeTupleWeightedAttr = Tuple["VertexType", "VertexType", Weight, AttributesDict]
EdgeLiteral = Union[EdgeTuple, EdgeTupleWeighted, EdgeTupleAttr, EdgeTupleWeightedAttr]

#: EdgeType: A type alias defined as Union[DiEdge, Edge, EdgeLiteral, MultiDiEdge, MultiEdge] and
# where EdgeLiteral is an alias for various edge-tuple formats.
EdgeType = Union["DiEdge", "Edge", EdgeLiteral, "MultiDiEdge", "MultiEdge"]


DEFAULT_WEIGHT = 1.0


def create_label(vertex1: "VertexType", vertex2: "VertexType", is_directed: bool) -> str:
    """Creates a string representation of the edge connecting ``vertex1`` and ``vertex2``, for
    example "(1, 2)".

    Directed edges have labels with the vertices ordered based on the instantiation order.
    Undirected edges have labels with vertices lexicographically sorted, which provides a consistent
    representation; for example, both :math:`(1, 2)` and :math:`(2, 1)` refer to the same undirected
    edge, but the edge label would always be "(1, 2)".

    Args:
        vertex1: The first vertex of the edge.
        vertex2: The second vertex of the edge.
        is_directed (bool): True indicates a directed edge, False an undirected edge.

    Returns:
        str: The edge key.
    """
    # Local import to avoid circular dependency.
    # pylint: disable=import-outside-toplevel
    from vertizee.classes import vertex

    v1_label = vertex.get_vertex_label(vertex1)
    v2_label = vertex.get_vertex_label(vertex2)

    if not is_directed and v1_label > v2_label:
        return f"({v2_label}, {v1_label})"
    return f"({v1_label}, {v2_label})"


class Edge(ABC):
    """An undirected edge that does not allow parallel connections between its vertices.

    To help ensure the integrity of graphs, the ``Edge`` class is abstract and cannot be
    instantiated directly. To create edges, use :meth:`Graph.add_edge
    <vertizee.classes.graph.Graph.add_edge>` and :meth:`Graph.add_edges_from
    <vertizee.classes.graph.Graph.add_edges_from>`.

    Edges may be assigned a weight as well as custom attributes using the :attr:`attr` dictionary.

    Note:
        In an undirected graph, edges :math:`(s, t)` and :math:`(t, s)` represent the same edge.
        Therefore, attempting to add :math:`(s, t)` and :math:`(t, s)` would raise an exception,
        since ``Edge`` objects do not support parallel connections. For parallel edge support,
        see :class:`MultiEdge` and :class:`MultiDiEdge`.

    Args:
        vertex1: The first vertex (the order does not matter, since this is an undirected edge).
        vertex2: The second vertex.
        weight: Optional; Edge weight. Defaults to 1.0.
        **attr: Optional; Keyword arguments to be added to the ``attr`` dictionary.

    Attributes:
        attr: Attribute dictionary to store ad hoc data associated with the edge.
    """
    def __init__(self, vertex1: Vertex, vertex2: Vertex, weight: float = DEFAULT_WEIGHT, **attr):
        # IMPORTANT: _vertex1 and _vertex2 are used in Edge.__hash__, and must therefore be
        # treated as immutable (read-only). If the vertices need to change, first delete the
        # edge instance and then create a new instance.
        self._vertex1 = vertex1
        self._vertex2 = vertex2

        if vertex1.label <= vertex2.label:
            self._label = f"({vertex1.label}, {vertex2.label})"
        else:
            self._label = f"({vertex2.label}, {vertex1.label})"

        self._attr: dict = {}
        for k, v in attr.items():
            self._attr[k] = v

        self._weight: float = float(weight)

        # Parallel edge count included as protected property (even on Edge classes that do not
        # support multi-edges) for efficiency when writing generic code that works on both standard
        # graphs and multigraphs.
        self._parallel_edge_count: int = 0
        self._parent_graph: GraphBase = vertex1._parent_graph

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    def __getitem__(self, key: Any) -> Any:
        """Supports index accessor notation to retrieve values from the `attr` dictionary.

        Example:
            >>> import vertizee as vz
            >>> g = vz.Graph()
            >>> g.add_edge(1, 2)
            (1, 2)
            >>> g[1, 2]["color"] = "blue"
            >>> g[1, 2]["color"]  # <== calls __getitem__()
            'blue'

        Args:
            key: The `attr` dictionary key.

        Returns:
            Any: The value indexed by `key`.
        """
        return self._attr[key]

    @abstractmethod
    def __hash__(self) -> int:
        """Creates a hash key using the edge's vertices."""

    def __repr__(self) -> str:
        return self.__str__()

    def __setitem__(self, key: Any, value: Any) -> None:
        """Supports index accessor notation to set values in the `attr` dictionary.

        Example:
            >>> import vertizee as vz
            >>> g = vz.Graph()
            >>> g.add_edge(1, 2)
            (1, 2)
            >>> g[1, 2]["color"] = "blue"  # <== calls __setitem__()
            >>> g[1, 2]["color"]
            "blue"

        Args:
            key: The `attr` dictionary key.
            value: The value to assign to `key` in the `attr` dictionary.
        """
        self._attr[key] = value

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted.
        """

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Edge:
            return abc_utils.check_methods(C, "__eq__", "__getitem__", "__hash__", "__setitem__",
                "__str__", "attr", "label", "is_loop", "vertex1", "vertex2", "weight")
        return NotImplemented

    @property
    def attr(self) -> dict:
        """Attribute dictionary to store ad hoc data associated with an edge."""
        return self._attr

    @property
    def label(self) -> str:
        """A string representation of the edge that includes the vertex endpoints, for example
        "(1, 2)". Directed edges have labels with the vertices ordered based on the instantiation
        order. Undirected edges have labels with vertices lexicographically sorted, which provides a
        consistent representation; for example, both :math:`(1, 2)` and :math:`(2, 1)` refer to the
        same undirected edge, but the edge label would always be "(1, 2)".
        """
        return self._label

    def is_loop(self) -> bool:
        """Returns True if this edge connects a vertex to itself."""
        return self._vertex1.label == self._vertex2.label

    @property
    def vertex1(self) -> Vertex:
        """The first vertex. For DiEdge objects, this is a synonym for the ``tail`` property."""
        return self._vertex1

    @property
    def vertex2(self) -> Vertex:
        """The second vertex. For DiEdge objects, this is a synonym for the ``head`` property."""
        return self._vertex2

    @property
    def weight(self) -> float:
        """The edge weight."""
        return self._weight


class DiEdge(Edge):
    """A directed edge that does not allow parallel connections between its vertices.

    To help ensure the integrity of graphs, the ``DiEdge`` class is abstract and cannot be
    instantiated directly. To create directed edges, use :meth:`DiGraph.add_edge
    <vertizee.classes.graph.DiGraph.add_edge>` and :meth:`DiGraph.add_edges_from
    <vertizee.classes.graph.DiGraph.add_edges_from>`.

    Edges may be assigned a weight as well as custom attributes using the :attr:`attr` dictionary.

    Note:
        In a directed graph, edge :math:`(s, t)` is distinct from edge :math:`(t, s)`. Therefore,
        it is legal for a ``DiGraph`` to contain parallel edges, as long as the edges are in the
        opposite direction. Each ``DiEdge`` object represents exactly one directed connection.

    Args:
        tail: The first vertex (the origin of the directed edge).
        head: The second vertex (the destination of the directed edge).
        weight: Optional; Edge weight. Defaults to 1.0.
        **attr: Optional; Keyword arguments to be added to the ``attr`` dictionary.

    Attributes:
        attr: Attribute dictionary to store ad hoc data associated with the edge.
    """
    def __init__(self, tail: Vertex, head: Vertex, weight: float = DEFAULT_WEIGHT, **attr):
        super().__init__(vertex1=tail, vertex2=head, weight=weight, **attr)
        self._label = f"({tail.label}, {head.label})"

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Creates a hash key using the edge's vertices."""

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted].
        """

    @classmethod
    def __subclasshook__(cls, C):
        if cls is DiEdge:
            return abc_utils.check_methods(C, "__eq__", "__getitem__", "__hash__", "__setitem__",
                "__str__", "attr", "head", "label", "is_loop", "tail", "vertex1", "vertex2",
                "weight")
        return NotImplemented

    @property
    def head(self) -> Vertex:
        """The head vertex, which is the destination of the directed edge."""
        return self._vertex2

    @property
    def tail(self) -> Vertex:
        """The tail vertex, which is the origin of the directed edge."""
        return self._vertex1


class MultiEdge(Edge):
    """An undirected edge that allows multiple parallel connections between its vertices.

    To help ensure the integrity of graphs, the ``MultiEdge`` class is abstract and cannot be
    instantiated directly. To create ``MultiEdge`` objects, use :meth:`MultiGraph.add_edge
    <vertizee.classes.graph.MultiGraph.add_edge>` and :meth:`MultiGraph.add_edges_from
    <vertizee.classes.graph.MultiGraph.add_edges_from>`.

    Edges may be assigned a weight as well as custom attributes using the :attr:`attr` dictionary.

    Note:
        In an undirected multigraph, edges :math:`(s, t)` and :math:`(t, s)` represent the same
        multi-edge. If multiple edges are added with the same vertices, then a single ``MultiEdge``
        instance is used to store the parallel edges. When working with multi-edges, use the
        ``parallel_edge_count`` or ``multiplicity`` properties to determine if the edge represents
        more than one edge connection.

    Args:
        vertex1: The first vertex (the order does not matter, since this is an undirected edge).
        vertex2: The second vertex.
        weight: Optional; Edge weight. Defaults to 1.0.
        **attr: Optional; Keyword arguments to be added to the ``attr`` dictionary.

    Attributes:
        attr: Attribute dictionary to store ad hoc data associated with the edge.
    """
    def __init__(self, vertex1: Vertex, vertex2: Vertex, weight: float = DEFAULT_WEIGHT, **attr):
        super().__init__(vertex1, vertex2, weight, **attr)

        if vertex1.label <= vertex2.label:
            self._label = f"({vertex1.label}, {vertex2.label})"
        else:
            self._label = f"({vertex2.label}, {vertex1.label})"

        self._parallel_edge_count: int = 0
        self._parallel_edge_weights: List[float] = []

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Creates a hash key using the edge's vertices."""

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted. In multigraphs, the string will show separate vertex
        tuples for each parallel edge, such as "(a, b), (a, b), (a, b)" for a multi-edge with
        multiplicity 3.
        """

    @classmethod
    def __subclasshook__(cls, C):
        if cls is MultiEdge:
            return abc_utils.check_methods(C, "__eq__", "__getitem__", "__hash__", "__setitem__",
                "__str__", "attr", "head", "is_loop", "label", "parallel_edge_count",
                "parallel_edge_weights", "tail", "vertex1", "vertex2", "weight",
                "weight_with_parallel_edges")
        return NotImplemented

    @property
    def multiplicity(self) -> int:
        """The multiplicity is the number of edges within the multi-edge.

        For edges without parallel connections, the multiplicity is 1. Each parallel edge adds 1 to
        the multiplicity. An edge with one parallel connection has multiplicity 2.
        """
        return self._parallel_edge_count + 1

    @property
    def parallel_edge_count(self) -> int:
        """The number of parallel edges.

        Note:
            The parallel edge count will always be one less than the total number of edges, since
            the initial edge connecting the two endpoints is not counted as a parallel edge.

        Returns:
            int: The number of parallel edges.
        """
        return self._parallel_edge_count

    @property
    def parallel_edge_weights(self) -> List[float]:
        """A list of parallel edge weights.

        The list of parallel edge weights does not contain the weight of the initial edge between
        the two vertices. Instead, the initial edge weight is stored in the ``weight`` property.

        Returns:
            List[float]: The list of parallel edge weights.
        """
        return self._parallel_edge_weights.copy()

    @property
    def weight_with_parallel_edges(self) -> float:
        """The total weight, including parallel edges.

        Returns:
            float: The total multi-edge weight, including parallel edges.
        """
        total = self._weight
        total += sum(self._parallel_edge_weights)
        return total


class MultiDiEdge(DiEdge, MultiEdge):
    """Edge that supports multiple directed edges between two vertices.

    To help ensure the integrity of graphs, ``MultiDiEdge`` is abstract and cannot be instantiated
    directly. To create edges, use :meth:`MultiDiGraph.add_edge
    <vertizee.classes.graph.MultiDiGraph.add_edge>` and :meth:`MultiDiGraph.add_edges_from
    <vertizee.classes.graph.MultiDiGraph.add_edges_from>`.

    Edges may be assigned a weight as well as custom attributes using the :attr:`attr` dictionary.

    Note:
        In a directed graph, edge :math:`(s, t)` is distinct from edge :math:`(t, s)`. Therefore,
        it is legal for a ``DiGraph`` to contain parallel edges, as long as the edges are in the
        opposite direction. Each ``DiEdge`` object represents exactly one directed connection.

    Args:
        tail: The first vertex (the origin of the directed edge).
        head: The second vertex (the destination of the directed edge).
        weight: Optional; Edge weight. Defaults to 1.0.
        **attr: Optional; Keyword arguments to be added to the ``attr`` dictionary.

    Attributes:
        attr: Attribute dictionary to store ad hoc data associated with the edge.
    """
    def __init__(self, tail: Vertex, head: Vertex, weight: float = DEFAULT_WEIGHT, **attr):
        super().__init__(vertex1=tail, vertex2=head, weight=weight, **attr)
        self._label = f"({tail.label}, {head.label})"

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        """Creates a hash key using the edge's vertices."""

    @abstractmethod
    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. In multigraphs, the string
        will show separate vertex tuples for each parallel edge, such as "(a, b), (a, b), (a, b)"
        for a multi-edge with multiplicity 3.
        """

    @classmethod
    def __subclasshook__(cls, C):
        if cls is MultiDiEdge:
            return abc_utils.check_methods(C, "__eq__", "__getitem__", "__hash__", "__setitem__",
                "__str__", "attr", "head", "is_loop", "label", "parallel_edge_count",
                "parallel_edge_weights", "tail", "vertex1", "vertex2", "weight",
                "weight_with_parallel_edges")
        return NotImplemented


#
# Protected concrete implementations.
#


class _Edge(Edge):
    """Concrete implementation of the abstract :class:`Edge` class."""

    def __eq__(self, other) -> bool:
        if not isinstance(other, Edge):
            return False

        v1 = self.vertex1
        v2 = self.vertex2
        o_v1 = other.vertex1
        o_v2 = other.vertex2
        if v1.label > v2.label:
            v1, v2 = v2, v1
        if o_v1.label > o_v2.label:
            o_v1, o_v2 = o_v2, o_v1
        if v1 != o_v1 or v2 != o_v2 or self._weight != other._weight:
            return False
        return True

    def __hash__(self) -> int:
        """Creates a hash key using the edge's vertices.

        Note that ``__eq__`` is defined to take ``_weight`` into consideration, whereas ``__hash__``
        does not. This is because ``_weight`` is not intended to be immutable throughout the
        lifetime of the object.

        From: "Python Hashes and Equality" <https://hynek.me/articles/hashes-and-equality/>:

        "Hashes can be less picky than equality checks. Since key lookups are always followed by
        an equality check, your hashes don’t have to be unique. That means that you can compute
        your hash over an immutable subset of attributes that may or may not be a unique
        "primary key" for the instance."
        """
        if self.vertex1.label > self.vertex2.label:
            return hash((self.vertex2, self.vertex1))

        return hash((self.vertex1, self.vertex2))

    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted.
        """
        if self.vertex1.label > self.vertex2.label:
            edge_str = f"({self.vertex2.label}, {self.vertex1.label}"
        else:
            edge_str = f"({self.vertex1.label}, {self.vertex2.label}"

        if self._parent_graph.is_weighted():
            edge_str = f"{edge_str}, {self._weight})"
        else:
            edge_str = f"{edge_str})"

        return edge_str


class _DiEdge(DiEdge):
    """Concrete implementation of the abstract :class:`DiEdge` class."""
    def __eq__(self, other) -> bool:
        if not isinstance(other, DiEdge):
            return False

        v1 = self.vertex1
        v2 = self.vertex2
        o_v1 = other.vertex1
        o_v2 = other.vertex2
        if (v1 != o_v1 or v2 != o_v2 or self._weight != other._weight):
            return False
        return True

    def __hash__(self) -> int:
        """Creates a hash key using the edge's vertices.

        Note that ``__eq__`` is defined to take ``_weight`` into consideration, whereas ``__hash__``
        does not. This is because ``_weight`` is not intended to be immutable throughout the
        lifetime of the object.

        From: "Python Hashes and Equality" <https://hynek.me/articles/hashes-and-equality/>:

        "Hashes can be less picky than equality checks. Since key lookups are always followed by
        an equality check, your hashes don’t have to be unique. That means that you can compute
        your hash over an immutable subset of attributes that may or may not be a unique
        "primary key" for the instance."
        """
        return hash((self.vertex1, self.vertex2))

    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted].
        """
        edge_str = f"({self.vertex1.label}, {self.vertex2.label}"

        if self._parent_graph.is_weighted():
            edge_str = f"{edge_str}, {self._weight})"
        else:
            edge_str = f"{edge_str})"

        return edge_str


class _MultiEdge(MultiEdge):
    """Concrete implementation of the abstract :class:`MultiEdge` class."""
    def __eq__(self, other) -> bool:
        if not isinstance(other, MultiEdge):
            return False

        v1 = self.vertex1
        v2 = self.vertex2
        o_v1 = other.vertex1
        o_v2 = other.vertex2
        if v1.label > v2.label:
            v1, v2 = v2, v1
        if o_v1.label > o_v2.label:
            o_v1, o_v2 = o_v2, o_v1
        if (
            v1 != o_v1
            or v2 != o_v2
            or self._parallel_edge_count != other._parallel_edge_count
            or self._weight != other._weight
        ):
            return False
        return True

    def __hash__(self) -> int:
        """Creates a hash key using the edge's vertices.

        Note that ``__eq__`` is defined to take ``_weight`` and ``_parallel_edge_count`` into
        consideration, whereas ``__hash__`` does not. This is because ``_weight`` and
        ``_parallel_edge_count``are not intended to be immutable throughout the lifetime of the
        object.

        From: "Python Hashes and Equality" <https://hynek.me/articles/hashes-and-equality/>:

        "Hashes can be less picky than equality checks. Since key lookups are always followed by
        an equality check, your hashes don’t have to be unique. That means that you can compute
        your hash over an immutable subset of attributes that may or may not be a unique
        "primary key" for the instance."
        """
        if self.vertex1.label > self.vertex2.label:
            return hash((self.vertex2, self.vertex1))

        return hash((self.vertex1, self.vertex2))

    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted. In multigraphs, the string will show separate vertex
        tuples for each parallel edge, such as "(a, b), (a, b), (a, b)" for a multi-edge with
        multiplicity 3.
        """
        if self.vertex1.label > self.vertex2.label:
            edge_str = f"({self.vertex2.label}, {self.vertex1.label}"
        else:
            edge_str = f"({self.vertex1.label}, {self.vertex2.label}"

        if self.vertex1._parent_graph.is_weighted():
            edge_str = f"{edge_str}, {self._weight})"
        else:
            edge_str = f"{edge_str})"

        edges = [edge_str for _ in range(self.multiplicity)]
        return ", ".join(edges)


class _MultiDiEdge(MultiDiEdge):
    """Concrete implementation of the abstract :class:`MultiDiEdge` class."""
    def __eq__(self, other) -> bool:
        if not isinstance(other, MultiDiEdge):
            return False

        v1 = self.vertex1
        v2 = self.vertex2
        o_v1 = other.vertex1
        o_v2 = other.vertex2
        if (
            v1 != o_v1
            or v2 != o_v2
            or self._parallel_edge_count != other._parallel_edge_count
            or self._weight != other._weight
        ):
            return False
        return True

    def __hash__(self) -> int:
        """Creates a hash key using the edge's vertices.

        Note that ``__eq__`` is defined to take ``_weight`` and ``_parallel_edge_count`` into
        consideration, whereas ``__hash__`` does not. This is because ``_weight`` and
        ``_parallel_edge_count``are not intended to be immutable throughout the lifetime of the
        object.

        From: "Python Hashes and Equality" <https://hynek.me/articles/hashes-and-equality/>:

        "Hashes can be less picky than equality checks. Since key lookups are always followed by
        an equality check, your hashes don’t have to be unique. That means that you can compute
        your hash over an immutable subset of attributes that may or may not be a unique
        "primary key" for the instance."
        """
        return hash((self.vertex1, self.vertex2))

    def __str__(self) -> str:
        """A simple string representation of the edge showing the vertex labels, and for weighted
        graphs, the edge weight.

        Examples: "(a, b)" [unweighted] and "(a, b, 4.5)" [weighted]. For undirected graphs, the
        vertices are lexicographically sorted. In multigraphs, the string will show separate vertex
        tuples for each parallel edge, such as "(a, b), (a, b), (a, b)" for a multi-edge with
        multiplicity 3.
        """
        edge_str = f"({self.vertex1.label}, {self.vertex2.label}"

        if self.vertex1._parent_graph.is_weighted():
            edge_str = f"{edge_str}, {self._weight})"
        else:
            edge_str = f"{edge_str})"

        edges = [edge_str for _ in range(self.multiplicity)]
        return ", ".join(edges)
