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

"""Algorithms for the all-pairs-shortest-paths problem.

Note:
    :math:`m = |E|` (the number of edges) and :math:`n = |V|` (the number of vertices)

* :func:`all_shortest_paths` - Finds the shortest paths between all pairs of vertices in a graph.
  This function chooses the fastest available all-pairs-shortest-paths algorithm depending on the
  properties of the graph.
* :func:`floyd_warshall` - Finds the shortest paths between all pairs of vertices in a graph using
  the Floyd-Warshall algorithm. Running time: :math:`O(n^3)`
* :func:`johnson` - Finds the shortest paths between all pairs of vertices in a graph using
  Donald Johnson's algorithm. Running time: :math:`O(mn(\\log{n}))`
* :func:`johnson_fibonacci` - Finds the shortest paths between all pairs of vertices in a graph
  using Donald Johnson's algorithm implemented with a Fibonacci heap version of Dijkstra's
  algorithm. Running time: :math:`O((n^2)\\log{n} + mn)`
"""

from __future__ import annotations
import math
from typing import Callable, Final

from vertizee import exception
from vertizee.algorithms.algo_utils.path_utils import ShortestPath
from vertizee.algorithms.paths import single_source_shortest_paths
from vertizee.classes.data_structures.vertex_dict import VertexDict
from vertizee.classes.edge import E
from vertizee.classes.graph import G
from vertizee.classes.vertex import V, VertexType

INFINITY: Final = float("inf")


def get_weight_function(weight: str = "Edge__weight") -> Callable[[E], float]:
    """Returns a function that accepts an edge and returns the corresponding edge weight.

    If there is no edge weight, then the edge weight is assumed to be one.

    Note:
        For multigraphs, the minimum edge weight among the parallel edge connections is returned.

    Args:
        weight: Optional; The key to use to retrieve the weight from the `Edge.attr`
            dictionary. The default value ('Edge__weight') uses the property `Edge.weight`.

    Returns:
        Callable[[E], float]: A function that accepts an edge and returns the
        corresponding edge weight.
    """

    def default_weight_function(edge: E) -> float:
        if edge._parent_graph.is_multigraph():
            return min(c.weight for c in edge.connections())
        return edge.weight

    def attr_weight_function(edge: E) -> float:
        if edge._parent_graph.is_multigraph():
            return min(c.attr.get(weight, 1.0) for c in edge.connections())
        return edge.attr.get(weight, 1.0)

    if weight == "Edge__weight":
        return default_weight_function
    return attr_weight_function

#
# TODO(cpeisert): run benchmarks to determine under which real-world circumstances Floyd-Warshall is
# faster than Johnson's algorithm.
#
def all_shortest_paths(
    graph: G[V, E], save_paths: bool = False, weight: str = "Edge__weight"
) -> VertexDict[VertexDict[ShortestPath[V]]]:
    """Finds the shortest paths between all pairs of vertices in a graph.

    This function chooses the fastest available all-pairs-shortest-paths algorithm depending on the
    properties of the graph. Note that :math:`m = |E|` (the number of edges) and :math:`n = |V|`
    (the number of vertices):

        * If the graph is sufficient dense that :math:`m > (n^2)/\\log{n}`, then the
          :func:`Floyd-Warshall algorithm <floyd_warshall>` is used.
        * Otherwise, :func:`Johnson's algorithm <johnson>` is used.

    Pairs of vertices for which there is no connecting path will have path length infinity. In
    addition, :meth:`ShortestPath.is_destination_reachable()
    <vertizee.algorithms.algo_utils.path_utils.ShortestPath.is_destination_reachable>`
    will return False.

    Args:
        graph: The graph to search.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.path_utils.reconstruct_path`.
            Defaults to False.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Returns:
        VertexDict[VertexDict[ShortestPath[V]]]: A dictionary mapping source vertices to
        dictionaries mapping destination vertices to :class:`ShortestPath
        <vertizee.algorithms.algo_utils.path_utils.ShortestPath>` objects.

    Raises:
        NegativeWeightCycle: If the graph contains a negative weight cycle.

    See Also:
        * :func:`floyd_warshall`
        * :func:`johnson`
        * :func:`reconstruct_path
          <vertizee.algorithms.algo_utils.path_utils.reconstruct_path>`
        * :class:`ShortestPath
          <vertizee.algorithms.algo_utils.path_utils.ShortestPath>`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`

    Example:
        >>> import vertizee as vz
        >>> from vertizee import DiVertex, VertexDict
        >>> from vertizee.algorithms.paths import all_shortest_paths, ShortestPath
        >>> g = vz.DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])
        >>> all_paths: VertexDict[VertexDict[ShortestPath[DiVertex]]] = \
            all_shortest_paths(g, save_paths=True)
        >>> len(all_paths)
        5
        >>> all_paths['s']['s'].length
        0
        >>> all_paths['s']['z'].length
        7
        >>> all_paths['s']['z'].path()
        [s, y, z]
        >>> all_paths['s']['x'].path()
        [s, y, t, x]
    """
    m = len(graph.edges())
    n = graph.vertex_count

    if m > (n ** 2) / math.log10(n):
        return floyd_warshall(graph, save_paths=save_paths, weight=weight)
    return johnson(graph, save_paths=save_paths, weight=weight)


def floyd_warshall(
    graph: G[V, E], save_paths: bool = False, weight: str = "Edge__weight"
) -> VertexDict[VertexDict[ShortestPath[V]]]:
    """Finds the shortest paths between all pairs of vertices in a graph using the Floyd-Warshall
    algorithm.

    Running time: :math:`O(n^3)` where :math:`n = |V|`

    Running space:

    * if ``save_paths`` is False: :math:`O(n^2)`
    * if ``save_paths`` is True: :math:`O(n^3)`

    When the number of edges is less than :math:`(n^2)/\\log{n}`, then the graph is sufficiently
    sparse that Johnson's algorithm will provide better asymptotic running time. See
    :func:`johnson`.

    Pairs of vertices for which there is no connecting path will have path length infinity. In
    addition, :meth:`ShortestPath.is_destination_reachable()
    <vertizee.algorithms.algo_utils.path_utils.ShortestPath.is_destination_reachable>`
    will return False.

    Note:
        This is loosely adapted from FLOYD-WARSHALL [CLRS2009]_, with the novel addition of subpath
        relaxation. For more information about subpath relaxation, see :class:`ShortestPath
        <vertizee.algorithms.algo_utils.path_utils.ShortestPath>`.

    Args:
        graph: The graph to search.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.path_utils.reconstruct_path`.
            Defaults to False.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Returns:
        VertexDict[VertexDict[ShortestPath[V]]]: A dictionary mapping source vertices to
        dictionaries mapping destination vertices to :class:`ShortestPath
        <vertizee.algorithms.algo_utils.path_utils.ShortestPath>` objects.

    Raises:
        NegativeWeightCycle: If the graph contains a negative weight cycle.

    See Also:
        * :func:`johnson`
        * :func:`reconstruct_path
          <vertizee.algorithms.algo_utils.path_utils.reconstruct_path>`
        * :class:`ShortestPath
          <vertizee.algorithms.algo_utils.path_utils.ShortestPath>`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`

    Example:
        >>> import vertizee as vz
        >>> from vertizee import DiVertex, VertexDict
        >>> from vertizee.algorithms.paths import floyd_warshall, ShortestPath
        >>> g = vz.DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])
        >>> all_paths: VertexDict[VertexDict[ShortestPath[DiVertex]]] = \
            floyd_warshall(g, save_paths=True)
        >>> len(all_paths)
        5
        >>> all_paths['s']['s'].length
        0
        >>> all_paths['s']['z'].length
        7
        >>> all_paths['s']['z'].path()
        [s, y, z]
        >>> all_paths['s']['x'].path()
        [s, y, t, x]

    References:
     .. [CLRS2009] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                   Introduction to Algorithms: Third Edition, pages 685-699. The MIT Press, 2009.
    """
    weight_function = get_weight_function(weight)
    source_and_destination_to_path: VertexDict[VertexDict[ShortestPath]] = VertexDict()

    # Initialize the default path lengths for all vertex combinations.
    for i in graph:
        source_and_destination_to_path[i] = VertexDict()
        for j in graph:
            if i == j:
                source_and_destination_to_path[i][j] = ShortestPath(
                    i, j, initial_length=0, save_path=save_paths
                )
                continue

            edge = graph._get_edge(i, j)
            if edge is None:
                source_and_destination_to_path[i][j] = ShortestPath(
                    i, j, initial_length=INFINITY, save_path=save_paths
                )
            else:
                w = weight_function(edge)
                source_and_destination_to_path[i][j] = ShortestPath(
                    i, j, initial_length=w, save_path=save_paths
                )

    for k in graph:
        for i in graph:
            for j in graph:
                path_i_j: ShortestPath = source_and_destination_to_path[i][j]
                path_i_k: ShortestPath = source_and_destination_to_path[i][k]
                path_k_j: ShortestPath = source_and_destination_to_path[k][j]
                path_i_j.relax_subpaths(path_i_k, path_k_j)

    for v in graph:
        if source_and_destination_to_path[v][v].length < 0:
            raise exception.NegativeWeightCycle("found a negative weight cycle")

    return source_and_destination_to_path


def johnson(
    graph: G[V, E], save_paths: bool = False, weight: str = "Edge__weight"
) -> VertexDict[VertexDict[ShortestPath]]:
    """Finds the shortest paths between all pairs of vertices in a graph using Donald Johnson's
    algorithm.

    Running time: :math:`O(mn(\\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    For a theoretically faster implementation with running time :math:`O((n^2)\\log{n} + mn)`, see
    :func:`shortest_paths.weighted.johnson_fibonacci`.

    When :math:`m > (n^2)/\\log{n}`, then the graph is sufficiently dense that the Floyd-Warshall
    algorithm will provide better asymptotic running time. See
    :func:`floyd_warshall`.

    Pairs of vertices for which there is no connecting path will have path length infinity. In
    additional, `ShortestPath.is_destination_reachable()` will return False.

    Note:
        This implementation is based on JOHNSON [CLRS2009_2]_.

    Args:
        graph: The graph to search.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.path_utils.reconstruct_path`. Defaults
            to False.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Returns:
        VertexDict[VertexDict[ShortestPath]]: A dictionary mapping source vertices to dictionaries
        mapping destination vertices to :class:`ShortestPath
        <vertizee.algorithms.algo_utils.path_utils.ShortestPath>` objects.

    Raises:
        NegativeWeightCycle: If the graph contains a negative weight cycle.

    See Also:
        * :func:`floyd_warshall`
        * :func:`johnson_fibonacci`
        * :func:`reconstruct_path
          <vertizee.algorithms.algo_utils.path_utils.reconstruct_path>`
        * :class:`ShortestPath <vertizee.algorithms.algo_utils.path_utils.ShortestPath>`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`

    References:
     .. [CLRS2009_2] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                     Introduction to Algorithms: Third Edition, pages 700-704. The MIT Press, 2009.
    """
    weight_function = get_weight_function(weight)

    g_prime: G = graph.deepcopy()
    G_PRIME_SOURCE = "__g_prime_src"
    for v in graph.vertices():
        g_prime.add_edge(G_PRIME_SOURCE, v, weight=0)

    bellman_paths: VertexDict[ShortestPath] = single_source_shortest_paths.bellman_ford(
        g_prime, G_PRIME_SOURCE)

    # pylint: disable=unused-argument
    def new_weight(v1: VertexType, v2: VertexType, reverse_graph: bool = False) -> float:
        edge: E = graph[v1, v2]
        return weight_function(edge) + bellman_paths[v1].length - bellman_paths[v2].length

    source_and_destination_to_path: VertexDict[VertexDict[ShortestPath]] = VertexDict()

    for i in graph:
        source_and_destination_to_path[i] = VertexDict()
        dijkstra_paths: VertexDict[ShortestPath] = single_source_shortest_paths.dijkstra(
            graph, source=i, weight=new_weight, save_paths=save_paths
        )
        for j in graph:
            source_and_destination_to_path[i][j] = dijkstra_paths[j]
            source_and_destination_to_path[i][j]._length += (
                bellman_paths[j].length - bellman_paths[i].length
            )

    return source_and_destination_to_path


def johnson_fibonacci(
    graph: G[V, E], save_paths: bool = False, weight: str = "Edge__weight"
) -> "VertexDict[VertexDict[ShortestPath]]":
    """Finds the shortest paths between all pairs of vertices in a graph using Donald Johnson's
    algorithm implemented with a Fibonacci heap version of Dijkstra's algorithm.

    Running time: :math:`O((n^2)\\log{n} + mn)` where :math:`m = |E|` and :math:`n = |V|`

    Pairs of vertices for which there is no connecting path will have path length infinity. In
    additional, `ShortestPath.is_destination_reachable()` will return False.

    Note:
        This implementation is based on JOHNSON [CLRS2009_2]_.

    Args:
        graph: The graph to search.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.path_utils.reconstruct_path`. Defaults to False.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Returns:
        VertexDict[VertexDict[ShortestPath]]: A dictionary mapping source vertices to dictionaries
        mapping destination vertices to :class:`ShortestPath
        <vertizee.algorithms.algo_utils.path_utils.ShortestPath>` objects.

    Raises:
        NegativeWeightCycle: If the graph contains a negative weight cycle.

    See Also:
        * :func:`floyd_warshall`
        * :func:`johnson`
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :func:`reconstruct_path <vertizee.algorithms.algo_utils.path_utils.reconstruct_path>`
        * :class:`ShortestPath <vertizee.algorithms.algo_utils.path_utils.ShortestPath>`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`
    """
    weight_function = get_weight_function(weight)

    g_prime: G = graph.deepcopy()
    G_PRIME_SOURCE = "__g_prime_src"
    for v in graph.vertices():
        g_prime.add_edge(G_PRIME_SOURCE, v, weight=0)

    bellman_paths: VertexDict[ShortestPath] = single_source_shortest_paths.bellman_ford(
        g_prime, G_PRIME_SOURCE)

    # pylint: disable=unused-argument
    def new_weight(v1: VertexType, v2: VertexType, reverse_graph: bool = False) -> float:
        edge: E = graph[v1, v2]
        return weight_function(edge) + bellman_paths[v1].length - bellman_paths[v2].length

    source_and_destination_to_path: VertexDict[VertexDict[ShortestPath]] = VertexDict()

    for i in graph:
        source_and_destination_to_path[i] = VertexDict()
        dijkstra_paths: VertexDict[ShortestPath] = single_source_shortest_paths.dijkstra_fibonacci(
            graph, source=i, weight=new_weight, save_paths=save_paths
        )
        for j in graph:
            source_and_destination_to_path[i][j] = dijkstra_paths[j]
            source_and_destination_to_path[i][j]._length += (
                bellman_paths[j].length - bellman_paths[i].length
            )

    return source_and_destination_to_path
