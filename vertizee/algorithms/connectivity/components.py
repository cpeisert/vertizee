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

"""Algorithms for connected components."""

from __future__ import annotations
from typing import Callable, Generic, Iterable, Iterator, Optional, Set, Union

import vertizee.algorithms.search.depth_first_search as dfs_module
from vertizee.classes import primitives_parsing
from vertizee.classes.collection_views import SetView
from vertizee.classes.graph import DiGraph, E, GraphBase, MultiDiGraph, V
from vertizee.classes.primitives_parsing import GraphPrimitive, ParsedEdgeAndVertexData
from vertizee import exception


class Component(Generic[V, E]):
    """A component in a graph.

    Args:
        initial_vertex: The initial vertex comprising the component.
    """

    def __init__(self, initial_vertex: V) -> None:
        self._edge_set = None
        self._vertex_set: Set[V] = set()
        self._vertex_set.add(initial_vertex)

    def __contains__(self, edge_or_vertex: GraphPrimitive) -> bool:
        if not self._vertex_set:
            return False

        vertex = next(iter(self._vertex_set))
        graph = vertex._parent_graph
        data: ParsedEdgeAndVertexData = primitives_parsing.parse_graph_primitive(edge_or_vertex)

        if data.edges:
            if graph.has_edge(data.edges[0].vertex1.label, data.edges[0].vertex2.label):
                edge = graph[data.edges[0].vertex1.label, data.edges[0].vertex2.label]
                if not self._edge_set:
                    self.edges()  # Executed for side effects to initialize self._edges.
                return edge in self._edge_set
            return False
        if data.vertices:
            return data.vertices[0].label in self._vertex_set

        raise ValueError("expected GraphPrimitive (EdgeType or VertexType); found "
            f"{type(edge_or_vertex).__name__}")

    def __iter__(self) -> Iterator[V]:
        """Iterates over the vertices in the component."""
        yield from self._vertex_set

    def __len__(self) -> int:
        """Returns the number of vertices in the component when the built-in Python function
        ``len`` is used."""
        return len(self._vertex_set)

    def edges(self) -> SetView[E]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the component
        edges."""
        if self._edge_set:
            return SetView(self._edge_set)

        self._edge_set = set()
        for vertex in self._vertex_set:
            for edge in vertex.incident_edges():
                if edge.vertex1 in self._vertex_set and edge.vertex2 in self._vertex_set:
                    if edge not in self._edge_set:
                        self._edge_set.add(edge)
        return SetView(self._edge_set)


    def vertices(self) -> SetView[V]:
        """Returns a :class:`SetView <vertizee.classes.collection_views.SetView>` of the component
        vertices."""
        yield from self._vertex_set


def connected_components(graph: GraphBase[V, E]) -> Iterator[Component[V, E]]:
    """Returns an iterator over the connected components; if the graph is directed, then the
    components are the strongly-connected components of the graph.

    For directed graphs, this function uses Kosaraju's algorithm [R2018]_ to find the strongly
    connected components (SCC), with the caveat that the SCCs are returned in reverse topological
    order. This ordering refers to topologically sorting the condensation graph (i.e. the graph
    created by representing each SCC as a vertex).

    Args:
        graph (GraphBase): The graph to analyze.

    Yields:
        Component: An iterator of :class:`Component` objects.

    See Also:
        * :class:`Component`
        * :func:`strongly_connected_components`
        * :func:`weakly_connected_components`
    """
    if len(graph) == 0:
        raise exception.Unfeasible("components are undefined for an empty graph")
    if graph.is_directed():
        return strongly_connected_components(graph)
    return _plain_depth_first_search(graph.vertices(), adjacency_function=_get_adjacent_to_child)


def strongly_connected_components(graph: Union[DiGraph, MultiDiGraph]) -> Iterator[Component]:
    """Returns an iterator over the strongly-connected components of the graph.

    This function uses Kosaraju's algorithm [R2018]_, with the caveat that the strongly-connected
    components (SCC) are returned in reverse topological order. This ordering refers to
    topologically sorting the condensation graph (i.e. the graph created by representing each
    SCC as a vertex).

    Args:
        graph (GraphBase): The graph to analyze.

    Yields:
        Component: An iterator of :class:`Component` objects.

    See Also:
        * :class:`Component <vertizee.algorithms.algo_utils.search_utils.Component>`
        * :func:`connected_components`
        * :func:`weakly_connected_components`
    """
    if len(graph) == 0:
        raise exception.Unfeasible("components are undefined for an empty graph")
    if not graph.is_directed():
        raise exception.GraphTypeNotSupported("graph must be directed")

    postorder = list(dfs_module.dfs_postorder_traversal(graph, reverse_graph=True))
    return _plain_depth_first_search(reversed(postorder), adjacency_function=_get_adjacent_to_child)


def weakly_connected_components(graph: Union[DiGraph, MultiDiGraph]) -> Iterator[Component]:
    """Returns an iterator over the weakly-connected components of the graph.

    A weakly connected component is a component that is connected when the direction of the edges
    is ignored. All strongly connected components are also weakly connected, but *not all*
    weakly connected components are strongly connected.

    Args:
        graph (GraphBase): The graph to analyze.

    Yields:
        Component: An iterator of :class:`Component` objects.

    See Also:
        * :class:`Component <vertizee.algorithms.algo_utils.search_utils.Component>`
        * :func:`strongly_connected_components`

    References:
     .. [R2018] Algorithms Illuminated (Part 2): Graph Algorithms and Data Structures.
                Tim Roughgarden. Soundlikeyourself Publishing LLC, 2018. (pages 57-63)
    """
    if len(graph) == 0:
        raise exception.Unfeasible("components are undefined for an empty graph")
    if not graph.is_directed():
        raise exception.GraphTypeNotSupported(
            "weakly-connected components are only defined for directed graphs")

    return _plain_depth_first_search(
        graph.vertices(), adjacency_function=_get_adjacent_to_child_undirected)


def _get_adjacent_to_child(child: V, parent: Optional[V]) -> Iterator[V]:
    if child._parent_graph.is_directed():
        return iter(child.adj_vertices_outgoing())
    return _get_adjacent_to_child_undirected(child, parent)


def _get_adjacent_to_child_undirected(child: V, parent: Optional[V]) -> Iterator[V]:
    """Gets adjacent vertices to ``child`` (excluding ``parent``), treating all graphs as if they
    were undirected."""
    adj_vertices = child.adj_vertices()
    if parent:
        adj_vertices = adj_vertices - {parent}
    return iter(adj_vertices)


def _plain_depth_first_search(
    vertices: Iterable[V],
    adjacency_function: Callable[[V, Optional[V]], Iterator[V]]
) -> Iterator[Component]:
    """Performs a plain depth-first search over the specified ``vertices``.

    Args:
        vertices: The graph vertices to be searched.
        adjacency_function: The function for retrieving the adjacent vertices of each vertex during
            the depth-first search.

    Yields:
        Component: An iterator of :class:`Component` objects.
    """
    seen: Set[V] = set()

    for vertex in vertices:
        if vertex in seen:
            continue

        component = Component(initial_vertex=vertex)
        children = adjacency_function(vertex, parent=None)
        stack = [dfs_module._StackFrame(vertex, children)]

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
                component._vertex_set.add(child)
                grandchildren = adjacency_function(child=child, parent=parent)
                stack.append(dfs_module._StackFrame(child, grandchildren))

        yield component
