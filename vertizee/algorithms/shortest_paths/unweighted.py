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

"""Algorithms for calculating shortest paths for unweighted graphs."""

from __future__ import annotations
from collections import deque
from typing import Set, TYPE_CHECKING

from vertizee import VertexNotFound
from vertizee.algorithms.algo_utils.shortest_path_utils import ShortestPath
from vertizee.classes.data_structures.vertex_dict import VertexDict
from vertizee.classes.vertex import Vertex

if TYPE_CHECKING:
    from vertizee.classes.graph import GraphBase
    from vertizee.classes.vertex import VertexType

INFINITY = float("inf")


def shortest_paths_breadth_first_search(
    graph: "GraphBase", source: "VertexType", save_paths: bool = False
) -> "VertexDict[ShortestPath]":
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices in an unweighted graph.

    Running time: :math:`O(|V| + |E|)`

    Unreachable vertices will have an empty list of vertices for their path and a length of
    infinity (``float("inf")``). In additional, ``ShortestPath.is_unreachable()`` will return True.

    Note:
        This is adapted from BFS [CLRS2009_11]_, but with the generalization of updating shortest
        paths and predecessor vertices using the concept of edge relaxation.

    Args:
        graph: The graph to search.
        source: The source vertex from which to find shortest paths to all other reachable vertices.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path`.
            Defaults to False.

    Returns:
        VertexDict[ShortestPath]: A dictionary mapping vertices to their shortest paths relative to
        the ``source`` vertex.

    See Also:
        * :class:`ShortestPath
          <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath>`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`

    Example:
        >>> g = Graph()
        >>> g.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (6, 7)])
        >>> paths = shortest_paths_breadth_first_search(g, 0)
        >>> paths[4].path
        [0, 1, 3, 4]
        >>> paths[4].length
        3
        >>> paths[6].path
        []
        >>> paths[6].is_unreachable()
        True

    References:
     .. [CLRS2009_11] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                      Introduction to Algorithms: Third Edition, page 595. The MIT Press, 2009.
    """
    try:
        s: Vertex = graph[source]
    except KeyError:
        raise VertexNotFound("source vertex was not found in the graph")
    vertex_to_path_map: VertexDict[ShortestPath] = VertexDict()

    for v in graph:
        vertex_path = ShortestPath(s, v, initial_length=INFINITY, save_paths=save_paths)
        vertex_to_path_map[v] = vertex_path
    vertex_to_path_map[s].reinitialize(initial_length=0)

    # pylint: disable=unused-argument
    def weight_function(v1: Vertex, v2: Vertex, reverse_graph: bool) -> float:
        return 1

    seen: Set[Vertex] = {s}
    queue = deque({s})
    while len(queue) > 0:
        u = queue.popleft()
        u_adj = u.get_adj_for_search()
        for w in u_adj:
            if w not in seen:
                seen.add(w)
                u_path: ShortestPath = vertex_to_path_map[u]
                w_path: ShortestPath = vertex_to_path_map[w]
                w_path.relax_edge(u_path, weight_function=weight_function)
                queue.append(w)
    return vertex_to_path_map
