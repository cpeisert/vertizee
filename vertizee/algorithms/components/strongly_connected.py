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

"""Algorithms for strongly connected components."""

from __future__ import annotations
from typing import List, TYPE_CHECKING

from vertizee.algorithms.algo_utils.search_utils import SearchTree
from vertizee.algorithms.search.depth_first_search import (
    _initialize_dfs_graph,
    _StackFrame,
    BLACK,
    COLOR,
    dfs_postorder_traversal,
    GRAY,
    PREDECESSOR,
    WHITE,
)
from vertizee.exception import GraphTypeNotSupported

if TYPE_CHECKING:
    from vertizee.classes.graph_base import GraphBase
    from vertizee.classes.vertex import Vertex


def kosaraju_strongly_connected_components(graph: "GraphBase") -> List["SearchTree"]:
    """Returns strongly connected components, where each component is a SearchTree.

    This function uses Kosaraju's algorithm [R2018]_, with the caveat that the strongly-connected
    components (SCC) are returned in reverse topological order. This ordering refers to
    topologically sorting the condensation graph (i.e. the graph created by representing each
    SCC as a vertex).

    Args:
        graph (GraphBase): The graph to search.

    Returns:
        List[SearchTree]: A list of strongly-connected components, where each component
        is stored in a SearchTree object.

    References:
     .. [R2018] Algorithms Illuminated (Part 2): Graph Algorithms and Data Structures.
                Tim Roughgarden. Soundlikeyourself Publishing LLC, 2018. (pages 57-63)
    """
    if not graph.is_directed_graph():
        raise GraphTypeNotSupported("graph must be directed")
    postorder = list(dfs_postorder_traversal(graph, reverse_graph=True))
    # Mark all vertices of graph as unexplored.
    _initialize_dfs_graph(graph)
    strongly_connected_components = []

    while postorder:
        v = postorder.pop()
        if v.attr[COLOR] == WHITE:  # Unexplored
            strongly_connected_components.append(_dfs_scc(graph, v))
    return strongly_connected_components


def _dfs_scc(graph: "GraphBase", source: "Vertex") -> SearchTree:
    """Helper function Depth-First Search for Strongly Connected Components to implement
    Kosaraju's algorithm.

    Args:
        graph (GraphBase): The graph to search.
        source (VertexType, optional): The source vertex from which to discover reachable
            vertices.

    Returns:
        SearchTree: A strongly-connected component stored as a depth-first-search tree.
    """
    scc = SearchTree(root=source)
    stack: List[_StackFrame] = [_StackFrame(source, source.adj_vertices_outgoing)]

    while stack:
        v = stack[-1].vertex
        adj_vertices = stack[-1].adj_vertices

        if v.attr[COLOR] == WHITE:  # Discovered new vertex v.
            v.attr[COLOR] = GRAY  # Mark as explored.
            scc.vertices.add(v)

            if v.attr[PREDECESSOR]:
                parent_v = v.attr[PREDECESSOR]
                edge = graph[parent_v, v]
                scc.edges_in_discovery_order.append(edge)

        if adj_vertices:  # Continue depth-first search with next adjacent vertex.
            w = adj_vertices.pop()

            if w.attr[COLOR] == WHITE:  # Undiscovered vertex w adjacent to v.
                w.attr[PREDECESSOR] = v
                stack.append(_StackFrame(w, w.adj_vertices_outgoing))
        elif v.attr[COLOR] != BLACK:  # FINISHED visiting vertex v.
            stack.pop()
            v.attr[COLOR] = BLACK
    return scc
