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

"""The default graph data type representing an undirected graph without parallel edges.

Undirected graph types:
    * :class:`Graph` - Undirected graph without parallel edges.
    * :class:`MultiGraph` - Undirected graph that allows parallel edges.
    * :class:`SimpleGraph` - Undirected graph containing no parallel edges and no self loops.

See Also:
    * :class:`DiGraph <vertizee.classes.digraph.DiGraph>`
    * :class:`Edge <vertizee.classes.edge.Edge>`
    * :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
    * :class:`MultiDiGraph <vertizee.classes.multidigraph.MultiDiGraph>`
    * :class:`MultiGraph <vertizee.classes.multigraph.MultiGraph>`
    * :class:`Vertex <vertizee.classes.vertex.Vertex>`

Note:
    All graph types except :class:`SimpleGraph` allow self loops.

Example:
    >>> import vertizee as vz
    >>> g = vz.Graph()
    >>> edge01 = g.add_edge(0, 1)
    >>> edge12 = g.add_edge(1, 2)
    >>> edge20 = g.add_edge(2, 0)
    >>> g[0].degree
    2
"""

# Note: In Python < 3.10, in order to prevent Sphinx from unfolding type aliases, future
# annotations must be imported and type aliases that should not be unfolded must be quoted.
from __future__ import annotations
from abc import ABC, abstractmethod
import collections.abc
import copy
import random
from typing import (
    Dict, Iterable, Iterator, List, Optional, overload, Set, Tuple, TYPE_CHECKING, Union, ValuesView
)

from vertizee import exception
from vertizee.classes import edge as edge_module
from vertizee.classes import primitives_parsing
from vertizee.classes import vertex as vertex_module

if TYPE_CHECKING:
    from vertizee.classes.edge import (
        DiEdge, Edge, EdgeClass, EdgeTuple, EdgeType, MultiDiEdge, MultiEdge
    )
    from vertizee.classes.primitives_parsing import (
        _EdgeData, _VertexData, GraphPrimitive, ParsedEdgeAndVertexData
    )
    from vertizee.classes.vertex import DiVertex, Vertex, VertexClass, VertexType

# Type aliases
GraphType = "GraphBase"


class GraphBase(ABC):
    """Abstract base class from which all graph classes inherit.

    Args:
        allow_self_loops: Indicates if self loops are allowed. A self loop is an edge that
            connects a vertex to itself. Defaults to True.
        is_directed_graph: True indicates that the graph has directed edges. Defaults to False.
        is_multigraph: True indicates that the graph is a multigraph (i.e. there can be multiple
             parallel edges between a pair of vertices). Defaults to False.
        is_weighted_graph: True indicates that the graph is weighted (i.e. there are edges with
            weights other than the default 1.0). Defaults to False.
    """
    def __init__(self, allow_self_loops: bool = True, is_directed_graph: bool = False,
            is_multigraph: bool = False, is_weighted_graph: bool = False):
        self._allow_self_loops = allow_self_loops

        self._is_directed_graph = is_directed_graph
        self._is_multigraph = is_multigraph
        self._is_weighted_graph = is_weighted_graph
        """If an edge is added with a weight that is not equal to
        ``vertizee.classes.edge.DEFAULT_WEIGHT``, then this flag is set to True."""

        self._edges: Dict[str, EdgeClass] = dict()
        """A dictionary mapping edge labels to edge objects. See :func:`create_edge_label
        <vertizee.classes.edge.create_edge_label>`."""

        self._vertices: Dict[str, VertexClass] = {}
        """A dictionary mapping vertex labels to vertex objects."""


    def __contains__(self, edge_or_vertex: GraphPrimitive) -> bool:
        data: ParsedEdgeAndVertexData = primitives_parsing.parse_graph_primitive(edge_or_vertex)
        if data.edges:
            return self._get_edge(data.edges[0].vertex1, data.edges[0].vertex2) is not None
        if data.vertices:
            return data.vertices[0].label in self._vertices

        raise ValueError("expected GraphPrimitive (EdgeType or VertexType); found "
            f"{type(edge_or_vertex)}")

    def __deepcopy__(self, memo) -> "GraphBase":
        new = self.__class__()
        new._allow_self_loops = self._allow_self_loops
        new._is_directed_graph = self._is_directed_graph
        new._is_multigraph = self._is_multigraph
        new._is_weighted_graph = self._is_weighted_graph
        for vertex in self._vertices.values():
            new.add_vertex(vertex)
        for edge in self._edges.values():
            new._add_edge_from_edge(edge)
        return new

    @overload
    def __getitem__(self, vertex: "VertexType") -> VertexClass:
        ...

    @overload
    def __getitem__(self, edge_tuple: "EdgeTuple") -> EdgeClass:
        ...

    def __getitem__(self, keys):
        """Supports index accessor notation to retrieve vertices and edges.

        Example:
            >>> import vertizee as vz
            >>> g = vz.Graph()
            >>> g.add_edge(1, 2)
            (1, 2)
            >>> g[1]  # <-- __getitem__(1)
            1
            >>> g[1, 2]  # <-- __getitem__([1, 2])
            (1, 2)

        Args:
            keys: A vertex label, vertex tuple, vertex object, or edge tuple. Specifying one vertex
                will retrieve the ``Vertex`` object from the graph. Specifying two vertices will
                retrieve the associated edge object from the graph.

        Returns:
            Union[EdgeClass, VertexClass, None]: The vertex specified by the vertex label or the
                edge specified by two vertices. If no matching vertex or edge found, returns None.

        Raises:
            IndexError: If ``keys`` is not exactly 1 or 2 ``VertexType`` keys.
            KeyError: If the graph does not contain a vertex or an edge matching ``keys``.
        """
        return_value: Union[EdgeClass, VertexClass, None] = None
        if isinstance(keys, tuple):
            if len(keys) > 2:
                raise IndexError(
                    f"graph index lookup supports one or two vertices; found {len(keys)} keys"
                )
            if len(keys) == 1:
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
        """Returns the number of vertices in the graph when the built-in Python function ``len`` is
        used."""
        return len(self._vertices)

    @abstractmethod
    def add_edge(
        self, vertex1: "VertexType", vertex2: "VertexType", weight: float, **attr
    ) -> "EdgeClass":
        """Adds a new edge to the graph.

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
        if not isinstance(edge_container, collections.abc.Iterable):
            raise TypeError("edge_container must be iterable")

        for e in edge_container:
            edge_data: _EdgeData = primitives_parsing.parse_edge_type(e)
            if edge_data.edge_object:
                edge_obj = edge_data.edge_object
                new_edge = self.add_edge(edge_obj.vertex1, edge_obj.vertex2, weight=edge_obj.weight,
                    **edge_obj.attr)
            else:
                new_edge = self.add_edge(edge_data.vertex1, edge_data.vertex2,
                    weight=edge_data.weight, **edge_data.attr)
            for k, v in attr.items():
                new_edge[k] = v

    @abstractmethod
    def add_vertex(self, label: "VertexLabel", **attr) -> "VertexClass":
        """Adds a vertex to the graph and returns the new Vertex object.

        If an existing vertex matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex. In order for a new
                vertex to be added, the label must not match an existing vertex in the graph.
            **attr: Optional; Keyword arguments to be added to the ``attr`` dictionary.

        Returns:
            VertexClass: The new vertex (or an existing vertex matching the vertex label).
        """

    def add_vertices_from(self, vertex_container: Iterable["VertexType"], **attr) -> None:
        """Adds vertices from a container, where the vertices are most often specified as strings
        or integers, but may also be tuples of the form ``Tuple[VertexLabel, AttributesDict]``.

        Args:
            vertex_container: Sequence of vertices to add.
            **attr: Optional; Keyword arguments to be added to the ``attr`` dictionaries of each
                vertex.

        Example:
            >>> graph.add_vertices_from([0, 1, 2, 3])

        See Also:
            :mod:`VertexType <vertizee.classes.vertex>`
        """
        if not isinstance(vertex_container, collections.abc.Iterable):
            raise TypeError("vertex_container must be iterable")

        for vertex in vertex_container:
            vertex_data: _VertexData = primitives_parsing.parse_vertex_type(vertex)
            new_vertex = self.add_vertex(vertex_data.label, **vertex_data.attr)
            for k, v in attr.items():
                new_vertex[k] = v

    def allows_self_loops(self) -> bool:
        """Returns True if this graph allows self loops, otherwise False."""
        return self._allow_self_loops

    @abstractmethod
    def clear(self) -> None:
        """Removes all edges and vertices from the graph."""

    @abstractmethod
    def contract_edge(self, edge: EdgeType, remove_loops: bool = False) -> None:
        """Removes ``edge`` from the graph and merges its two incident vertices.

        Formal Definitions [WEC2020]_:

            * Set difference - :math:`B \setminus A = \{ x\in B \mid x \notin A \}`
            * Edge contraction (written as :math:`G/e`):

            Let :math:`G = (V, E)` be a graph (or directed graph) containing an edge
            :math:`e = (u, v)` with :math:`u \neq v`. Let :math:`f` be a function which maps every
            vertex in :math:`V \setminus\{u, v\}` to itself, and otherwise, maps it to a new vertex
            :math:`w`. The contraction of :math:`e` results in a new graph :math:`G' = (V', E')`,
            where :math:`V' = (V \setminus\{u, v\})\cup\{w\}`, :math:`E' = E \setminus \{e\}`, and
            for every :math:`x \in V`, :math:`x' = F(x)\in V'` is incident to an edge
            :math:`e' \in E'` if and only if, the corresponding edge, :math:`e \in E` is incident
            to :math:`x` in :math:`G`.

        For efficiency, only one of the two incident vertices is actually deleted. After the edge
        contraction:

           - Incident edges of ``edge.vertex2`` are modified such that ``vertex2`` is replaced by
             ``edge.vertex1``
           - Incident loops on ``vertex2`` become loops on ``vertex1``
           - ``edge.vertex2`` is deleted from the graph
           - If loops are not deleted, then :math:`degree(vertex1)` [post-merge]
             :math:`\\Longleftrightarrow degree(vertex1) + degree(vertex2)` [pre-merge]

        Since an Edge's vertices are used in its hash function, they must be treated as immutable
        for the lifetime of the object. Therefore, when ``vertex2`` is deleted, its incident edges
        must also be deleted.

        In some cases, an incident edge of ``vertex2`` will be modified such that by replacing
        ``vertex2`` with ``vertex1``, there exists an edge in the graph matching the new endpoints.
        In this case, if the graph is a multigraph, then the existing edge object is updated by
        incrementing its ``parallel_edge_count`` and appending to ``parallel_edge_weights`` as
        needed. If the graph is not a multigraph, then the existing edge is not modified.

        If the graph does not contain an edge matching the new endpoints after replacing ``vertex2``
        with ``vertex1``, then a new edge object is added to the graph.

        Note that if either ``GraphBase._allow_self_loops`` is False or ``remove_loops`` is True,
        self loops will be deleted from the merged vertex (``vertex1``).

        Args:
            edge: The edge to contract.
            remove_loops: If True, loops on the merged vertices will be removed. Defaults to False.

        References:
         .. [WEC2020] Wikipedia contributors. "Edge contraction." Wikipedia, The Free
                   Encyclopedia. Available from: https://en.wikipedia.org/wiki/Edge_contraction.
                   Accessed 19 October 2020.
        """

    @abstractmethod
    def deepcopy(self) -> "GraphBase":
        """Returns a deep copy of this graph."""

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """The number of edges."""

    @property
    @abstractmethod
    def edges(self) -> ValuesView["EdgeClass"]:
        """The set of graph edges."""
        return self._edges.values()

    @abstractmethod
    def get_random_edge(self) -> Optional["EdgeClass"]:
        """Returns a randomly selected edge from the graph, or None if there are no edges.

        Returns:
            EdgeClass: The random edge, or None if there are no edges.
        """

    def has_edge(self, edge: "EdgeType") -> bool:
        """Returns True if the graph contains the edge.

        Instead of using this method, it is also possible to use the ``in`` operator:

            >>> if ["s", "t"] in graph:

        or with objects:

            >>> edge_st = graph.add_edge("s", "t")
            >>> if edge_st in graph:

        Args:
            edge: The edge to verify.

        Returns:
            bool: True if there is a matching edge in the graph, otherwise False.

        See Also:
            :mod:`EdgeType <vertizee.classes.edge>`
        """
        edge_data: _EdgeData = primitives_parsing.parse_edge_type(edge)
        return edge_data.get_label() in self._edges

    def has_vertex(self, vertex: "VertexType") -> bool:
        """Returns True if the graph contains the specified vertex."""
        vertex_data: _VertexData = primitives_parsing.parse_vertex_type(vertex)
        return vertex_data.get_label() in self._vertices

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

    @abstractmethod
    def remove_edge(self, edge: "EdgeType", remove_isolated_vertices: bool = False) -> None:
        """Removes an edge from the graph.

        Args:
            edge: The edge to remove.
            remove_isolated_vertices: If True, then vertices adjacent to ``edge`` that become
                isolated after the edge removal are also removed. Defaults to False.

        Raises:
            EdgeNotFound: If the edge is not in the graph.
        """

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
                self.remove_edge(edge_type)
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
                self.remove_edge(self._vertices[label].loop_edge)
            self._vertices.pop(label)
        return len(vertex_labels_to_remove)

    def remove_vertex(self, vertex: VertexType) -> None:
        """Removes the indicated vertex.

        For a vertex to be removed, it must be isolated, that is, no incident edges (except self
        loops). Any incident edges must be deleted prior to vertex removal.

        Args:
            vertex: The vertex to remove.

        Raises:
            VertizeeException: If the vertex has non-loop incident edges.
            VertexNotFound: If the vertex is not in the graph.
        """
        vertex_data: _VertexData = primitives_parsing.parse_vertex_type(vertex)
        label = vertex_data.get_label()
        if label not in self._vertices:
            raise exception.VertexNotFound(f"vertex {{{label}}} not found")

        vertex_obj = self._vertices[label]
        if not vertex_obj.is_isolated():
            raise exception.VertizeeException(f"cannot remove vertex {vertex_obj} due to "
                "adjacent non-loop edges")

        if vertex_obj.loop_edge:
            self.remove_edge(vertex_obj.loop_edge, remove_isolated_vertices=False)
        self._vertices.pop(label, None)

    @property
    def vertex_count(self) -> int:
        """The count of vertices in the graph."""
        return len(self._vertices)

    @property
    @abstractmethod
    def vertices(self) -> ValuesView["VertexClass"]:
        """The set of graph vertices."""
        return self._vertices.values()

    @property
    @abstractmethod
    def weight(self) -> float:
        """Returns the weight of all edges."""

    @abstractmethod
    def _add_edge_from_edge(self, edge: EdgeClass) -> EdgeClass:
        """Protected method for adding edges by copying data from an existing Edge object. This
        method is motivated to simplify graph copying. If the edge object is already a member
        of this graph instance, the edge is returned and no additional edge is added."""

    def _get_edge(self, vertex1: VertexType, vertex2: VertexType) -> Optional[EdgeClass]:
        """Gets the edge specified by the vertices, or None if no such edge exists.

        Args:
            vertex1: The first vertex (the *tail* in directed graphs).
            vertex2: The second vertex (the *head* in directed graphs).

        Returns:
            Edge: The edge or None if not found.

        See Also:
            :mod:`GraphPrimitive <vertizee.classes.primitives_parsing>`
        """
        edge_label = edge_module.create_edge_label(vertex1, vertex2, self._is_directed_graph)
        if edge_label in self._edges:
            return self._edges[edge_label]
        return None

    def _get_or_add_vertex(self, vertex_data: _VertexData) -> VertexClass:
        """Helper method to get a vertex, or if not found, add a new vertex.

        Args:
            vertex_data: The vertex to get or add.

        Returns:
            VertexClass: The vertex that was either found or added.
        """
        if vertex_data.label in self._vertices:
            return self._vertices[vertex_data.label]

        if vertex_data.vertex_object:
            return self.add_vertex(vertex_data.vertex_object)
        return self.add_vertex([vertex_data.label, vertex_data.attr])

    def _get_vertex(self, vertex: VertexType) -> Optional[VertexClass]:
        """Returns the specified vertex or None if not found."""
        vertex_data: _VertexData = primitives_parsing.parse_vertex_type(vertex)
        label = vertex_data.get_label()
        if label in self._vertices:
            return self._vertices[label]
        return None


class Graph(GraphBase):
    """An undirected graph without parallel edges.

    Args:
        allow_self_loops: Indicates if self loops are allowed. A self loop is an edge that
            connects a vertex to itself. Defaults to True.

    See also:
        * :class:`DiGraph`
        * :class:`MultiDiGraph`
        * :class:`MultiGraph`
    """
    def __init__(self, allow_self_loops: bool = True):
        super().__init__(allow_self_loops=allow_self_loops, is_directed_graph=False,
            is_multigraph=False, is_weighted_graph=False)

    def add_edge(
        self, vertex1: "VertexType", vertex2: "VertexType", weight: float, **attr
    ) -> "EdgeClass":
        """Adds a new edge to the graph.

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
        vertex1_data: _VertexData = primitives_parsing.parse_vertex_type(vertex1)
        vertex2_data: _VertexData = primitives_parsing.parse_vertex_type(vertex1)
        v1_label = vertex1_data.get_label()
        v2_label = vertex2_data.get_label()
        existing_edge = self._get_edge(v1_label, v2_label)

        if v1_label == v2_label and not self._allow_self_loops:
            raise exception.SelfLoopsNotAllowed(
                f"attempted to add loop edge ({v1_label}, {v2_label})")
        if existing_edge:
            raise exception.ParallelEdgesNotAllowed("attempted to add parallel edge "
                f"{existing_edge}; graph is not a multigraph")

        if weight != edge_module.DEFAULT_WEIGHT:
            self._is_weighted_graph = True

        v1_obj = self._get_or_add_vertex(vertex1_data)
        v2_obj = self._get_or_add_vertex(vertex2_data)
        new_edge = edge_module._Edge(v1_obj, v2_obj, weight=weight, **attr)

        self._edges[new_edge.label] = new_edge
        # Handle vertex bookkeeping.
        new_edge.vertex1._add_edge(new_edge)
        new_edge.vertex2._add_edge(new_edge)
        return new_edge

    def add_vertex(self, label: "VertexLabel", **attr) -> "VertexClass":
        """Adds a vertex to the graph and returns the new Vertex object.

        If an existing vertex matches the vertex label, the existing vertex is returned.

        Args:
            label: The label (``str`` or ``int``) to use for the new vertex. For a new vertex
                to be added, the label must not match an existing vertex in the graph.
            **attr: Optional; Keyword arguments to be added to the ``attr`` dictionary.

        Returns:
            VertexClass: The new vertex (or an existing vertex matching the vertex label).
        """
        if not isinstance(label, (str, int)):
            raise TypeError(f"label must be a string or integer; found {type(label)}")
        vertex_label = str(label)

        if vertex_label in self._vertices:
            return self._vertices[vertex_label]

        new_vertex = vertex_module._Vertex(vertex_label, parent_graph=self, **attr)
        self._vertices[vertex_label] = new_vertex
        return new_vertex

    def clear(self) -> None:
        """Removes all edges and vertices from the graph."""
        self._edges.clear()
        self._vertices.clear()

    def contract_edge(self, edge: EdgeType, remove_loops: bool = False) -> None:
        """Removes ``edge`` from the graph and merges its two incident vertices.

        Upon merging the incident vertices, if a new edge is formed that would result in a parallel
        edge, then the new edge is discarded; the original edge is kept. If parallel edges are
        needed, use a :class:`MultiGraph` or class:`MultiDiGraph`.

        For more information about edge contraction, see :meth:`GraphBase.contract_edge`.

        Args:
            edge: The edge to contract.
            remove_loops: If True, loops on the merged vertex will be removed. Defaults to False.
        """
        edge_data: _EdgeData = primitives_parsing.parse_edge_type(edge)
        label = edge_data.get_label()
        edge_obj: Edge = self._edges[label]

        v1 = edge_obj.vertex1
        v2 = edge_obj.vertex2

        edges_to_delete: List[Edge] = []
        # Incident edges of vertex2, where vertex2 is to be replaced by vertex1.
        for incident in v2.incident_edges:
            edges_to_delete.append(incident)
            if incident.is_loop():
                if self._allow_self_loops and not remove_loops:
                    if not self._get_edge(v1, v1):
                        self.add_edge(v1, v1, weight=incident.weight, **incident.attr)
            elif incident.vertex1 == v2:
                if not self._get_edge(incident.vertex2, v1):
                    self.add_edge(incident.vertex2, v1, weight=incident.weight, **incident.attr)
            else:  # incident.vertex2 == v2
                if not self._get_edge(incident.vertex1, v1):
                    self.add_edge(incident.vertex1, v1, weight=incident.weight, **incident.attr)

        # Delete indicated edges after finishing loop iteration.
        for e in edges_to_delete:
            self.remove_edge(e)
        # Delete v2 from the graph.
        self.remove_vertex(v2)

    def deepcopy(self) -> "Graph":
        """Returns a deep copy of this graph."""
        return copy.deepcopy(self)

    @property
    def edge_count(self) -> int:
        """The number of edges."""
        return len(self._edges)

    @property
    def edges(self) -> ValuesView["Edge"]:
        """The set of graph edges."""
        return self._edges.values()

    def get_random_edge(self) -> Optional["Edge"]:
        """Returns a randomly selected edge from the graph, or None if there are no edges.

        Returns:
            EdgeClass: The random edge, or None if there are no edges.
        """
        if self._edges:
            return random.choice(self._edges.values())
        return None

    def remove_edge(self, edge: "EdgeType", remove_isolated_vertices: bool = False) -> None:
        """Removes an edge from the graph.

        Args:
            edge: The edge to remove.
            remove_isolated_vertices: If True, then vertices adjacent to ``edge`` that become
                isolated after the edge removal are also removed. Defaults to False.

        Raises:
            EdgeNotFound: If the edge is not in the graph.

        See Also:
            :mod:`EdgeType <vertizee.classes.edge>`
        """
        edge_data: _EdgeData = primitives_parsing.parse_edge_type(edge)
        label = edge_data.get_label()
        if label not in self._edges:
            raise exception.EdgeNotFound(f"graph does not have edge {edge_data.get_label()}")

        edge_obj = self._edges[label]
        self._edges.pop(label)
        v1: vertex_module._Vertex = self._vertices[edge_data.vertex1.get_label()]
        v2: vertex_module._Vertex = self._vertices[edge_data.vertex2.get_label()]
        v1._remove_edge(edge_obj)
        v2._remove_edge(edge_obj)

    @property
    def vertices(self) -> ValuesView["Vertex"]:
        """The set of graph vertices."""
        return self._vertices.values()

    @property
    def weight(self) -> float:
        """Returns the weight of all edges."""
        return sum([edge.weight for edge in self._edges.values()])

    def _add_edge_from_edge(self, edge: Edge) -> Edge:
        """Protected method for adding edges by copying data from an existing Edge object. This
        method is motivated to simplify graph copying. If the edge object is already a member
        of this graph instance, the edge is returned and no additional edge is added."""
        if edge.label in self._edges:
            return edge
        return self.add_edge(edge.vertex1, edge.vertex2, edge.weight, **edge.attr)

        # For multigraphs, rather checking if the edge label in graph, check if actual object in
        # graph: if edge._parent_graph = self
        #
        # new_edge._parallel_edge_count += edge.parallel_edge_count
        # new_edge._parallel_edge_weights += edge.parallel_edge_weights
        # while len(new_edge.parallel_edge_weights) > new_edge.parallel_edge_count:
        #     new_edge._parallel_edge_weights.pop()
        # self._edges_with_freq_weight[new_edge] = new_edge.multiplicity
