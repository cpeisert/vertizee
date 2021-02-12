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
from typing import cast, Dict, Generic, Iterator, TYPE_CHECKING, ValuesView

from vertizee import exception
from vertizee.classes import primitives_parsing
from vertizee.classes import edge as edge_module
from vertizee.classes.comparable import Comparable
from vertizee.classes.primitives_parsing import GraphPrimitive, ParsedEdgeAndVertexData, VertexData
from vertizee.classes.edge import E, E_co, MultiEdgeBase
from vertizee.classes.vertex import V_co

if TYPE_CHECKING:
    from vertizee.classes.vertex import VertexType


class Tree(Comparable, Generic[V_co, E_co]):
    """A :term:`tree <free tree>` is a :term:`graph` that is :term:`connected` and does not contain
    :term:`circuits <circuit>`.

    For any positive integer :math:`n`, a tree with :math:`n` vertices has :math:`n - 1` edges.

    This class has generic type parameter ``V_co``, which enable the type-hint usage
    ``Tree[V_co]``.

    * ``V_co = TypeVar("V", bound="VertexBase", covariant=True)`` See :class:`VertexBase
      <vertizee.classes.vertex.VertexBase>`.

    Note:
        A tree is not intended to be used as a standalone graph, but rather as a container to
        organize trees. The edges and vertices of a tree belong to an instance of
        :class:`GraphBase <vertizee.classes.graph.GraphBase>` (or one of its subclasses). However,
        ``Tree`` implements the basic graph API, such as ``contains``, ``iter``, and ``len`` as
        well as index notation to get vertices and edges.

    Args:
        root: The root vertex of the tree.
    """

    __slots__ = ("_edges", "_root", "_vertices")

    def __init__(self, root: V_co) -> None:
        super().__init__()
        self._root = root
        self._edges: Dict[str, E_co] = dict()
        """A dictionary mapping edge labels to edge objects. See :func:`create_edge_label
        <vertizee.classes.edge.create_edge_label>`."""

        self._vertices: Dict[str, V_co] = dict()
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
        if (
            self._root == other._root
            and self._vertices.keys() == other._vertices.keys()
            and self._edges.keys() < other._edges.keys()
        ):
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

    def __getitem__(self, vertex: "VertexType") -> V_co:
        """Supports index accessor notation to retrieve vertices.

        Args:
            vertex: The vertex to retrieve, usually using the vertex label.

        Returns:
            V: The specified vertex.

        Raises:
            KeyError: If the tree does not contain a vertex or an edge matching ``keys``.
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
        """Iterates over the vertices of the tree."""
        yield from self._vertices.values()

    def __len__(self) -> int:
        """Returns the number of vertices in the tree when the built-in Python function
        ``len`` is used."""
        return len(self._vertices)

    def add_edge(self, edge: E) -> E:
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
            return edge
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

        self._edges[edge.label] = cast(E_co, edge)
        self._vertices[edge.vertex1.label] = edge.vertex1
        self._vertices[edge.vertex2.label] = edge.vertex2
        return edge

    @property
    def edge_count(self) -> int:
        """The number of edges, including parallel edge connections."""
        if self.is_multigraph():
            return sum(cast(MultiEdgeBase[V_co], e).multiplicity for e in self._edges.values())
        return len(self._edges)

    def edges(self) -> ValuesView[E_co]:
        """A view of the tree edges."""
        return self._edges.values()

    def get_edge(self, vertex1: "VertexType", vertex2: "VertexType") -> E_co:
        """Returns the :term:`edge` specified by the vertices, or None if no such edge exists.

        Args:
            vertex1: The first vertex (the :term:`tail` in :term:`directed graphs
                <directed graph>`).
            vertex2: The second vertex (the :term:`head` in directed graphs).

        Returns:
            EdgeBase[V]: The specified edge.

        Raises:
            KeyError: If the tree does not contain an edge with the specified vertex endpoints.
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self.is_directed())
        return self._edges[edge_label]

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

    def merge(self, other: "Tree[V_co, E_co]") -> None:
        """Merges ``other`` into this tree."""
        self._edges.update(other._edges)
        self._vertices.update(other._vertices)

    @property
    def root(self) -> V_co:
        """The root vertex of the tree."""
        return self._root

    @property
    def vertex_count(self) -> int:
        """The count of vertices in the tree."""
        return len(self._vertices)

    def vertices(self) -> ValuesView[V_co]:
        """A ValuesView of the tree vertices."""
        return self._vertices.values()

    @property
    def weight(self) -> float:
        """Returns the weight of all tree edges."""
        return sum([edge.weight for edge in self._edges.values()])
