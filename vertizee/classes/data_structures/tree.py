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

"""
====
Tree
====

A :term:`tree <free tree>` is a :term:`graph` that is connected and does not contain
:term:`circuits <circuit>`.
"""

from __future__ import annotations
from typing import cast, Dict, Generic, Iterator, overload, TYPE_CHECKING, Union, ValuesView

from vertizee import exception
from vertizee.classes import primitives_parsing
from vertizee.classes import edge as edge_module
from vertizee.classes.comparable import Comparable
from vertizee.classes.primitives_parsing import GraphPrimitive, ParsedEdgeAndVertexData, VertexData
from vertizee.classes.edge import E, MultiEdgeBase
from vertizee.classes.vertex import V

if TYPE_CHECKING:
    from vertizee.classes.edge import EdgeType
    from vertizee.classes.vertex import VertexType


class Tree(Comparable, Generic[V, E]):
    """A :term:`tree <free tree>` is a :term:`graph` that is :term:`connected` and does not contain
    :term:`circuits <circuit>`.

    For any positive integer :math:`n`, a tree with :math:`n` vertices has :math:`n - 1` edges.

    This class has generic type parameters ``V`` and ``E``, which enable the type-hint usage
    ``Tree[V, E]``.

    * ``V = TypeVar("V", bound="VertexBase")`` See :class:`VertexBase
      <vertizee.classes.vertex.VertexBase>`.
    * ``E = TypeVar("E", bound=Union["Connection", "MultiConnection"])`` See
      :class:`Connection <vertizee.classes.edge.Connection>` and
      :class:`MultiConnection <vertizee.classes.edge.MultiConnection>`.

    Note:
        A tree is not intended to be used as a standalone graph, but rather as a container to
        organize trees. The edges and vertices of a tree belong to an instance of
        :class:`G <vertizee.classes.graph.G>` (or one of its subclasses). However, ``Tree``
        implements the basic graph API, such as ``contains``, ``iter``, and ``len`` as well as
        index notation to get vertices and edges.

    Args:
        root: The root vertex of the tree.
    """

    __slots__ = ("_edges", "_root", "_vertices")

    def __init__(self, root: V) -> None:
        super().__init__()
        self._root: V = root
        self._edges: Dict[str, E] = dict()
        """A dictionary mapping edge labels to edge objects. See :func:`create_edge_label
        <vertizee.classes.edge.create_edge_label>`."""

        self._vertices: Dict[str, V] = dict()
        """A dictionary mapping vertex labels to vertex objects."""

        self._vertices[root.label] = root

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tree):
            return NotImplemented
        if self._root != other._root:
            return False
        if self._vertices.keys() != other._vertices.keys():
            return False
        if self._edges.keys() != other._edges.keys():
            return False

        return True

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Tree):
            return NotImplemented
        if self._root < other._root:
            return True
        if self._root == other._root and self._vertices.keys() < other._vertices.keys():
            return True
        if (self._root == other._root and self._vertices.keys() == other._vertices.keys()
            and self._edges.keys() < other._edges.keys()):
            return True

        return False


    def __contains__(self, edge_or_vertex: GraphPrimitive) -> bool:
        graph = self._root._parent_graph
        data: ParsedEdgeAndVertexData = primitives_parsing.parse_graph_primitive(edge_or_vertex)

        if data.edges:
            if graph.has_edge(data.edges[0].vertex1.label, data.edges[0].vertex2.label):
                return self.has_edge(data.edges[0].vertex1.label, data.edges[0].vertex2.label)
            return False
        if data.vertices:
            return data.vertices[0].label in self._vertices

        raise TypeError(
            "expected GraphPrimitive (i.e. EdgeType or VertexType) instance; "
            f"{type(edge_or_vertex).__name__} found"
        )

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
            Union[V, E, None]: The vertex specified by the vertex label or the edge specified by
            two vertices. If no matching vertex or edge found, returns None.

        Raises:
            KeyError: If the graph does not contain a vertex or an edge matching ``keys``.
            VertizeeException: If ``keys`` is not a valid ``GraphPrimitive`` (that is a
                ``VertexType`` or an ``EdgeType``).

        See Also:
            * :mod:`EdgeType <vertizee.classes.edge>`
            * :mod:`VertexType <vertizee.classes.vertex>`
        """
        graph = self._root._parent_graph
        data: ParsedEdgeAndVertexData = primitives_parsing.parse_graph_primitive(keys)
        if data.edges:
            edge_label = edge_module.create_edge_label(
                data.edges[0].vertex1.label, data.edges[0].vertex2.label, self.is_directed()
            )
            return self._edges[edge_label]
        if data.vertices:
            vertex = graph._vertices[data.vertices[0].label]
            if vertex.label in self._vertices:
                return self._vertices[vertex.label]
            raise KeyError(keys)

        raise TypeError(
            "expected GraphPrimitive (i.e. EdgeType or VertexType) instance; "
            f"{type(keys).__name__} found"
        )

    def __hash__(self) -> int:
        """Supports adding trees to set-like collections."""
        return hash(self._vertices)

    def __iter__(self) -> Iterator[V]:
        """Iterates over the vertices of the tree."""
        yield from self._vertices.values()

    def __len__(self) -> int:
        """Returns the number of vertices in the tree when the built-in Python function
        ``len`` is used."""
        return len(self._vertices)

    def add_edge(self, edge: E) -> None:
        """Adds a new edge to the tree.

        Exactly one of the edge's vertices must already be in the tree. If neither of the edge's
        endpoints are in the tree, then the edge is unreachable from the existing tree, and by
        definition, trees must be connected. If both of the edge's endpoints are in the tree, it
        forms a cycle, and by definition, trees are :term:`acyclic`.

        Args:
            edge: The edge to add.

        Raises:
            Unfeasible: An ``Unfeasible`` exception is raised if the edge is not already in the
                tree and either both of its endpoints are in the tree (which would create a cycle)
                or neither of its endpoints are in the tree (which would make the edge unreachable).
        """
        if edge.label in self._edges:
            return
        if edge.vertex1.label not in self._vertices and edge.vertex2.label not in self._vertices:
            raise exception.Unfeasible(
                f"neither of the edge endpoints {edge} were found in the "
                "tree; exactly one of the endpoints must already be in the tree"
            )
        if edge.vertex1.label in self._vertices and edge.vertex2.label in self._vertices:
            raise exception.Unfeasible(
                f"both of the edge endpoints {edge} were found in the "
                "tree, which would create a cycle; trees are acyclic"
            )

        self._edges[edge.label] = edge
        self._vertices[edge.vertex1.label] = edge.vertex1
        self._vertices[edge.vertex2.label] = edge.vertex2

    @property
    def edge_count(self) -> int:
        """The number of edges, including parallel edge connections."""
        if self.is_multigraph():
            return sum(cast(MultiEdgeBase, e).multiplicity for e in self._edges.values())
        return len(self._edges)

    def edges(self) -> ValuesView[E]:
        """A view of the tree edges."""
        return self._edges.values()

    def has_edge(self, vertex1: VertexType, vertex2: VertexType) -> bool:
        """Returns True if the tree contains the edge.

        Instead of using this method, it is also possible to use the ``in`` operator:

            >>> if ("s", "t") in tree:

        or with objects:

            >>> edge_st = graph.add_edge("s", "t")
            >>> if edge_st in tree:

        Args:
            vertex1: The first endpoint of the edge.
            vertex2: The second endpoint of the edge.

        Returns:
            bool: True if there is a matching edge in the tree, otherwise False.

        See Also:
            :mod:`VertexType <vertizee.classes.vertex>`
        """
        label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        return label in self._edges

    def has_vertex(self, vertex: VertexType) -> bool:
        """Returns True if the tree contains the specified vertex.

        See Also:
            :mod:`VertexType <vertizee.classes.vertex>`
        """
        vertex_data: VertexData = primitives_parsing.parse_vertex_type(vertex)
        return vertex_data.label in self._vertices

    def is_directed(self) -> bool:
        """Returns True if this is a directed tree (i.e. each edge points from a tail vertex to a
        head vertex)."""
        return self._root._parent_graph.is_directed()

    def is_multigraph(self) -> bool:
        """Returns True if this is a multigraph (i.e. a graph that allows parallel edges)."""
        return self._root._parent_graph.is_multigraph()

    def merge(self, other: "Tree[V, E]") -> None:
        """Merges ``other`` into this tree."""
        self._edges.update(other._edges)
        self._vertices.update(other._vertices)

    @property
    def root(self) -> V:
        """The root vertex of the tree."""
        return self._root

    @property
    def vertex_count(self) -> int:
        """The count of vertices in the tree."""
        return len(self._vertices)

    def vertices(self) -> ValuesView[V]:
        """A ValuesView of the tree vertices."""
        return self._vertices.values()

    @property
    def weight(self) -> float:
        """Returns the weight of all tree edges."""
        return sum([edge.weight for edge in self._edges.values()])
