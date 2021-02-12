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
========================
Connectivity: components
========================

Algorithms for :term:`connected components <connected component>`.

**Recommended Tutorial**: :doc:`Connected Components <../../tutorials/connected_components>` - |image-colab-components|

.. |image-colab-components| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/cpeisert/vertizee/blob/master/docs/source/tutorials/connected_components.ipynb

Class summary
=============

* :class:`Component` - A :term:`component <connected component>` in a :term:`graph`.

Function summary
================

* :func:`connected_components` - Returns an iterator over the
  :term:`connected components <connected component>`; if the :term:`graph` is directed, then the
  components are the :term:`strongly-connected <strongly connected>` components of the
  :term:`digraph`.
* :func:`strongly_connected_components` - Returns an iterator over the
  :term:`strongly-connected <strongly connected>` components of the :term:`digraph`.
* :func:`weakly_connected_components` - Returns an iterator over the
  :term:`weakly-connected <weakly connected>` components of the graph.

Detailed documentation
======================
"""

from __future__ import annotations
from typing import (
    Callable,
    cast,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Union,
    ValuesView,
)

from vertizee import exception
from vertizee.algorithms.algo_utils import search_utils
import vertizee.algorithms.search.depth_first_search as dfs_module
from vertizee.classes import primitives_parsing
from vertizee.classes.collection_views import SetView
from vertizee.classes.edge import E_co
from vertizee.classes.graph import DiGraph, MultiDiGraph
from vertizee.classes.primitives_parsing import GraphPrimitive, ParsedEdgeAndVertexData
from vertizee.classes.vertex import DiVertex, MultiDiVertex, V, V_co

if TYPE_CHECKING:
    from vertizee.classes.edge import DiEdge, MultiDiEdge
    from vertizee.classes.graph import GraphBase


class Component(Generic[V_co, E_co]):
    """A :term:`component <connected component>` in a :term:`graph`.

    Args:
        initial_vertex: The initial vertex comprising the component.
    """

    def __init__(self, initial_vertex: V_co) -> None:
        self._edge_set: Optional[Set[E_co]] = None
        self._parent_graph = initial_vertex._parent_graph
        self._vertices: Dict[str, V_co] = dict()
        self._vertices[initial_vertex.label] = initial_vertex

    def __contains__(self, edge_or_vertex: GraphPrimitive) -> bool:
        data: ParsedEdgeAndVertexData = primitives_parsing.parse_graph_primitive(edge_or_vertex)
        graph = self._parent_graph

        if data.edges:
            if graph.has_edge(data.edges[0].vertex1.label, data.edges[0].vertex2.label):
                edge = graph.get_edge(data.edges[0].vertex1.label, data.edges[0].vertex2.label)
                if not self._edge_set:
                    self.edges()  # Executed for side effects to initialize self._edges.
                assert self._edge_set is not None
                return edge in self._edge_set
            return False
        if data.vertices:
            return data.vertices[0].label in self._vertices

        raise TypeError(
            "expected GraphPrimitive (i.e. EdgeType or VertexType) instance; "
            f"{type(edge_or_vertex).__name__} found"
        )

    def __iter__(self) -> Iterator[V_co]:
        """Iterates over the vertices in the component."""
        yield from self._vertices.values()

    def __len__(self) -> int:
        """Returns the number of vertices in the component when the built-in Python function
        ``len`` is used."""
        return len(self._vertices)

    def edges(self) -> "SetView[E_co]":
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the component
        edges."""
        if self._edge_set:
            return SetView(self._edge_set)

        self._edge_set = set()
        vertices = self._vertices.values()
        for vertex in vertices:
            for edge in vertex.incident_edges():
                if edge.vertex1 in vertices and edge.vertex2 in vertices:
                    if edge not in self._edge_set:
                        self._edge_set.add(cast(E_co, edge))
        return SetView(self._edge_set)

    def vertices(self) -> ValuesView[V_co]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the component
        vertices."""
        return self._vertices.values()


def connected_components(graph: "GraphBase[V_co, E_co]") -> Iterator["Component[V_co, E_co]"]:
    """Returns an iterator over the :term:`connected components <connected component>`; if the
    :term:`graph` is directed, then the components are the
    :term:`strongly-connected <strongly connected>` components of the graph.

    Note:
        For :term:`directed graphs <digraph>`, this function uses Kosaraju's algorithm to find the
        strongly connected components (SCC), with the caveat that the SCCs are returned in reverse
        :term:`topological order <topological ordering>`. This ordering refers to
        :term:`topologically sorting <topological sorting>` the :term:`condensation graph
        <condensation>`.

    Args:
        graph (G): The graph to analyze.

    Yields:
        Component: An iterator of :class:`Component` objects.

    Note:
        This implementation of Kosaraju's algorithm is based on the treatment in Roughgarden.
        :cite:`2018:roughgarden`
    """
    if len(graph) == 0:
        raise exception.Unfeasible("components are undefined for an empty graph")
    if graph.is_directed():
        assert isinstance(graph, (DiGraph, MultiDiGraph))
        return cast(Iterator[Component[V_co, E_co]], strongly_connected_components(graph))
    return _plain_depth_first_search(graph, adjacency_function=_get_adjacent_to_child)


def strongly_connected_components(
    graph: Union["DiGraph", "MultiDiGraph"]
) -> Iterator["Component[Union[DiVertex, MultiDiVertex], Union[DiEdge, MultiDiEdge]]"]:
    """Returns an iterator over the :term:`strongly-connected <strongly connected>` components of
    the :term:`digraph`.

    Note:
        For :term:`directed graphs <digraph>`, this function uses Kosaraju's algorithm to find the
        strongly connected components (SCC), with the caveat that the SCCs are returned in reverse
        :term:`topological order <topological ordering>`. This ordering refers to
        :term:`topologically sorting <topological sorting>` the :term:`condensation graph
        <condensation>`.

    Args:
        graph: The graph to analyze.

    Yields:
        Component: An iterator of :class:`Component` objects.

    Note:
        This implementation of Kosaraju's algorithm is based on the treatment in Roughgarden.
        :cite:`2018:roughgarden`
    """
    if len(graph) == 0:
        raise exception.Unfeasible("components are undefined for an empty graph")
    if not graph.is_directed():
        raise exception.GraphTypeNotSupported("graph must be directed")

    postorder = cast(
        List[Union[DiVertex, MultiDiVertex]],
        list(dfs_module.dfs_postorder_traversal(graph, reverse_graph=True)),  # type: ignore
    )
    return _plain_depth_first_search(
        graph, adjacency_function=_get_adjacent_to_child, vertices=reversed(postorder)
    )


def weakly_connected_components(
    graph: Union["DiGraph", "MultiDiGraph"]
) -> Iterator["Component[Union[DiVertex, MultiDiVertex], Union[DiEdge, MultiDiEdge]]"]:
    """Returns an iterator over the :term:`weakly-connected <weakly connected>` components of the
    graph.

    Args:
        graph (G): The graph to analyze.

    Yields:
        Component: An iterator of :class:`Component` objects.
    """
    if len(graph) == 0:
        raise exception.Unfeasible("components are undefined for an empty graph")
    if not graph.is_directed():
        raise exception.GraphTypeNotSupported(
            "weakly-connected components are only defined for directed graphs"
        )

    return _plain_depth_first_search(graph, adjacency_function=_get_adjacent_to_child_undirected)


def _get_adjacent_to_child(child: V, parent: Optional[V]) -> Iterator[V]:
    if child._parent_graph.is_directed():
        assert isinstance(child, (DiVertex, MultiDiVertex))
        return cast(Iterator[V], iter(child.adj_vertices_outgoing()))
    return _get_adjacent_to_child_undirected(child, parent)


def _get_adjacent_to_child_undirected(child: V, parent: Optional[V]) -> Iterator[V]:
    """Gets adjacent vertices to ``child`` (excluding ``parent``), treating all graphs as if they
    were undirected."""
    adj_vertices = set(child.adj_vertices())
    if parent:
        adj_vertices = adj_vertices - {parent}
    return cast(Iterator[V], iter(adj_vertices))


def _plain_depth_first_search(
    graph: GraphBase[V_co, E_co],
    adjacency_function: Callable[[V_co, Optional[V_co]], Iterator[V_co]],
    vertices: Optional[Iterable[V_co]] = None,
) -> Iterator[Component[V_co, E_co]]:
    """Performs a plain depth-first search over the specified ``vertices``.

    Args:
        vertices: The graph vertices to be searched.
        adjacency_function: The function for retrieving the adjacent vertices of each vertex during
            the depth-first search.

    Yields:
        Component: An iterator of :class:`Component` objects.
    """
    if vertices:
        vertex_iterable = vertices
    else:
        vertex_iterable = graph.vertices()

    seen = set()
    for vertex in vertex_iterable:
        if vertex in seen:
            continue

        component: Component[V_co, E_co] = Component(initial_vertex=vertex)
        children = adjacency_function(vertex, None)
        stack = [search_utils.VertexSearchState(vertex, children)]

        while stack:
            parent = stack[-1].parent
            children = stack[-1].children
            try:
                child = next(children)
            except StopIteration:
                stack.pop()
                continue

            if child not in seen:
                seen.add(child)
                component._vertices[child.label] = child
                grandchildren = adjacency_function(child, parent)
                stack.append(search_utils.VertexSearchState(child, grandchildren))

        yield component
