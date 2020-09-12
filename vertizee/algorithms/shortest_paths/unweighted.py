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

from collections import deque
from typing import Set

from vertizee.classes.collections.vertex_dict import VertexDict
from vertizee.classes.shortest_path import ShortestPath
from vertizee.classes.graph_base import GraphBase
from vertizee.classes.vertex import Vertex, VertexKeyType

INFINITY = float("inf")


def breadth_first_search_shortest_paths(
    graph: GraphBase, source: VertexKeyType, find_path_lengths_only: bool = True
) -> VertexDict[ShortestPath]:
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices.

    Unreachable vertices will have an empty list of vertices for their path and a length of
    infinity (`math.inf`). In additional, `ShortestPath.is_unreachable` will return True.

    Args:
        graph (GraphBase): The graph to search.
        source (VertexKeyType): The source vertex from which to find shortest paths to all other
            reachable vertices.
        find_path_lengths_only(bool, optional): If True, only calculates the shortest path lengths,
            but does not determine the actual vertex sequences comprising each path. To reconstruct
            specific shortest paths, see `~shortest_path.reconstruct_path`. If set to False, then
            the ShortestPath.path property will contain the sequence of vertices comprising the
            shortest path. Defaults to True.

    Returns:
        VertexDict[VertexKeyType, ShortestPath]: A dictionary mapping vertices to their shortest
            paths and associated path lengths.

    See Also:
        `~shortest_path.ShortestPath`
        `~vertex_dict.VertexDict`

    Example::

        >>> g = Graph()
        >>> g.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (6, 7)])
        >>> paths = breadth_first_search_shortest_paths(g, 0)
        >>> paths[4].path
        [0, 1, 3, 4]
        >>> paths[4].length
        3
        >>> paths[6].path
        []
        >>> paths[6].is_unreachable()
        True
    """
    s: Vertex = graph[source]
    if s is None:
        raise ValueError("source vertex was not found in the graph")
    vertex_to_path_map: VertexDict[ShortestPath] = VertexDict()
    store_paths = not find_path_lengths_only

    for v in graph:
        vertex_path = ShortestPath(s, v, initial_length=INFINITY, store_full_paths=store_paths)
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
