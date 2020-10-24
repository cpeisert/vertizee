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

"""Algorithms for calculating shortest paths for weighted graphs."""

from __future__ import annotations
from typing import Callable, Union

import vertizee
from vertizee import VertexNotFound
from vertizee.algorithms.algo_utils.shortest_path_utils import ShortestPath
from vertizee.classes.data_structures.fibonacci_heap import FibonacciHeap
from vertizee.classes.data_structures.priority_queue import PriorityQueue
from vertizee.classes.data_structures.vertex_dict import VertexDict
from vertizee.classes.edge import Edge
from vertizee.classes.graph_base import GraphBase
from vertizee.classes.vertex import Vertex, VertexType

INFINITY = float("inf")


def get_weight_function(
    weight: Union[Callable, str] = "Edge__weight"
) -> Callable[["Vertex", "Vertex", bool], float]:
    """Returns a function that accepts two vertices and a boolean indicating if the graph should be
    treated as if it were reversed (i.e. edges of directed graphs in the opposite direction) and
    returns the corresponding edge weight.

    If there is no edge weight, then the edge weight is assumed to be one.  If `graph` is a
    multigraph, the minimum edge weight over all parallel edges is returned.

    Notes:
        To support reversed graphs, custom weight functions should implement the following pattern:

        .. code-block:: python

            def get_min_weight(v1: Vertex, v2: Vertex, reverse_graph: bool) -> float:
                graph = v1._parent_graph
                if reverse_graph:
                    edge: Edge = graph[v2, v1]
                    edge_str = f'({v2.label}, {v1.label})'
                else:
                    edge: Edge = graph[v1, v2]
                    edge_str = f'({v1.label}, {v2.label})'
                if edge is None:
                    raise AlgorithmError(f'graph does not have edge {edge_str}')

                <YOUR CODE HERE>

                return min_weight

        The weight function may also serve as a filter by returning None for any edge that should
        be excluded from the shortest path search.  For example, adding the following would
        exclude blue edges:

        .. code-block:: python

            if edge.attr.get('color', 'red') == 'blue':
                return None

    Args:
        weight: Optional; If callable, then ``weight`` itself is returned. If
            a string is specified, it is the key to use to retrieve the weight from an ``Edge.attr``
            dictionary. The default value (``Edge__weight``) returns a function that accesses the
            ``Edge.weight`` property.

    Returns:
        Callable[[VertexType, VertexType, bool], float]: A function that accepts two vertices
        and a boolean indicating if the graph is reversed (i.e. edges of directed graphs in the
        opposite direction) and returns the corresponding edge weight.
    """
    if callable(weight):
        return weight

    if not isinstance(weight, str):
        raise ValueError("`weight` must be a callable function or a string")

    def get_min_weight(v1: Vertex, v2: Vertex, reverse_graph: bool) -> float:
        graph = v1._parent_graph
        if reverse_graph:
            edge: Edge = graph[v2, v1]
            edge_str = f"({v2.label}, {v1.label})"
        else:
            edge = graph[v1, v2]
            edge_str = f"({v1.label}, {v2.label})"
        if edge is None:
            raise vertizee.AlgorithmError(f"graph does not have edge {edge_str}")
        if weight == "Edge__weight":
            min_weight = edge.weight
        else:
            min_weight = edge.attr.get(weight, 1.0)

        if len(edge.parallel_edge_weights) > 0:
            min_parallel = min(edge.parallel_edge_weights)
            min_weight = min(min_weight, min_parallel)
        return min_weight

    return get_min_weight


def get_weight_function_all_pairs_shortest_paths(
    weight: str = "Edge__weight",
) -> Callable[["Edge"], float]:
    """Returns a function that accepts an edge and returns the corresponding edge weight.

    If there is no edge weight, then the edge weight is assumed to be 1.  If ``graph`` is a
    multigraph, the minimum edge weight over all parallel edges is returned.

    Args:
        weight: Optional; The key to use to retrieve the weight from the `Edge.attr`
            dictionary. The default value ('Edge__weight') uses the property `Edge.weight`.

    Returns:
        Callable[[Edge], float]: A function that accepts an edge and returns the
        corresponding edge weight.
    """

    def default_weight_function(edge: Edge) -> float:
        w = edge.weight
        if len(edge.parallel_edge_weights) > 0:
            min_parallel = min(edge.parallel_edge_weights)
            w = min(w, min_parallel)
        return w

    def attr_weight_function(edge: Edge) -> float:
        w = edge.attr.get(weight, 1.0)
        if len(edge.parallel_edge_weights) > 0:
            min_parallel = min(edge.parallel_edge_weights)
            w = min(w, min_parallel)
        return w

    if weight == "Edge__weight":
        return default_weight_function
    else:
        return attr_weight_function


def all_pairs_shortest_paths_floyd_warshall(
    graph: "GraphBase", weight: str = "Edge__weight", save_paths: bool = False
) -> "VertexDict[VertexDict[ShortestPath]]":
    """Finds the shortest paths between all pairs of vertices in a graph using the Floyd-Warshall
    algorithm.

    Running time: :math:`O(n^3)` where :math:`n = |V|`

    Running space:

    * if ``save_paths`` is False: :math:`O(n^2)`
    * if ``save_paths`` is True: :math:`O(n^3)`

    When the number of edges is less than :math:`(n^2)/log(n)`, then the graph is sufficiently
    sparse that Johnson's algorithm will provide better asymptotic running time. See
    :func:`all_pairs_shortest_paths_johnson`.

    Pairs of vertices for which there is no connecting path will have path length infinity. In
    addition, :meth:`ShortestPath.is_destination_reachable()
    <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath.is_destination_reachable>`
    will return False.

    Note:
        This is loosely adapted from FLOYD-WARSHALL [CLRS2009]_, with the novel addition of
        conceptualizing the process of finding intermediate vertices of paths as subpath relaxation.

    Args:
        graph: The graph to search.
        weight: Optional; The key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge__weight``) uses the property ``Edge.weight``.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path`.
            Defaults to False.

    Returns:
        VertexDict[VertexDict[ShortestPath]]: A dictionary mapping source vertices to dictionaries
        mapping destination vertices to :class:`ShortestPath
        <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath>` objects.

    Raises:
        NegativeWeightCycle: If the graph contains a negative weight cycle. **Note that for
            undirected graphs, any negative weight edge is a negative weight cycle.**

    See Also:
        * :func:`all_pairs_shortest_paths_johnson`
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :func:`reconstruct_path
          <vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path>`
        * :class:`ShortestPath
          <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath>`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`

    Example:
        >>> g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])
        >>> paths: VertexDict[VertexDict[ShortestPath]] = \
            all_pairs_shortest_paths_floyd_warshall(g, save_paths=True)
        >>> len(paths)
        5
        >>> paths['s']['s'].length
        0
        >>> paths['s']['z'].length
        7
        >>> paths['s']['z'].path
        [s, y, z]
        >>> paths['s']['x'].path
        [s, y, t, x]

    References:
     .. [CLRS2009] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                   Introduction to Algorithms: Third Edition, pages 685-699. The MIT Press, 2009.
    """
    weight_function = get_weight_function_all_pairs_shortest_paths(weight)
    source_and_destination_to_path: VertexDict[VertexDict[ShortestPath]] = VertexDict()

    # Initialize the default path lengths for all vertex combinations.
    for i in graph:
        source_and_destination_to_path[i] = VertexDict()
        for j in graph:
            if i == j:
                source_and_destination_to_path[i][j] = ShortestPath(
                    i, j, initial_length=0, save_paths=save_paths
                )
                continue

            edge = graph._get_edge(i, j)
            if edge is None:
                source_and_destination_to_path[i][j] = ShortestPath(
                    i, j, initial_length=INFINITY, save_paths=save_paths
                )
            else:
                w = weight_function(edge)
                source_and_destination_to_path[i][j] = ShortestPath(
                    i, j, initial_length=w, save_paths=save_paths
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
            raise vertizee.NegativeWeightCycle("found a negative weight cycle")

    return source_and_destination_to_path


def all_pairs_shortest_paths_johnson(
    graph: "GraphBase", weight: str = "Edge__weight", save_paths: bool = False
) -> "VertexDict[VertexDict[ShortestPath]]":
    """Finds the shortest paths between all pairs of vertices in a graph using Donald Johnson's
    algorithm.

    Running time: :math:`O(mn(log(n)))` where :math:`m = |E|` and :math:`n = |V|`

    For a theoretically faster implementation with running time :math:`O((n^2)log(n) + mn)`, see
    :func:`shortest_paths.weighted.all_pairs_shortest_paths_johnson_fibonacci`.

    When :math:`m > (n^2)/log(n)`, then the graph is sufficiently dense that the Floyd-Warshall
    algorithm will provide better asymptotic running time. See
    :func:`all_pairs_shortest_paths_floyd_warshall`.

    Pairs of vertices for which there is no connecting path will have path length infinity. In
    additional, `ShortestPath.is_destination_reachable()` will return False.

    Note:
        This implementation is based on JOHNSON [CLRS2009_2]_.

    Args:
        graph: The graph to search.
        weight: Optional; The key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge__weight``) uses the property ``Edge.weight``.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path`. Defaults to False.

    Returns:
        VertexDict[VertexDict[ShortestPath]]: A dictionary mapping source vertices to dictionaries
        mapping destination vertices to :class:`ShortestPath
        <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath>` objects.

    Raises:
        NegativeWeightCycle: If the graph contains a negative weight cycle. **Note that for
            undirected graphs, any negative weight edge is a negative weight cycle.**

    See Also:
        * :func:`all_pairs_shortest_paths_floyd_warshall`
        * :func:`all_pairs_shortest_paths_johnson_fibonacci`
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :func:`reconstruct_path <vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path>`
        * :class:`ShortestPath <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath>`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`

    Example:
        >>> g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])
        >>> paths: VertexDict[VertexDict[ShortestPath]] = \
            all_pairs_shortest_paths_johnson(g, save_paths=True)
        >>> len(paths)
        5
        >>> paths['s']['s'].length
        0
        >>> paths['s']['z'].length
        7
        >>> paths['s']['z'].path
        [s, y, z]
        >>> paths['s']['x'].path
        [s, y, t, x]

    References:
     .. [CLRS2009_2] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                     Introduction to Algorithms: Third Edition, pages 700-704. The MIT Press, 2009.
    """
    weight_function = get_weight_function_all_pairs_shortest_paths(weight)

    g_prime: GraphBase = graph.deepcopy()
    G_PRIME_SOURCE = "__g_prime_src"
    for v in g_prime.vertices:
        g_prime.add_edge(G_PRIME_SOURCE, v, weight=0)

    bellman_paths: VertexDict[ShortestPath] = shortest_paths_bellman_ford(g_prime, G_PRIME_SOURCE)

    # pylint: disable=unused-argument
    def new_weight(v1: VertexType, v2: VertexType, reverse_graph: bool = False) -> float:
        edge: Edge = graph[v1, v2]
        return weight_function(edge) + bellman_paths[v1].length - bellman_paths[v2].length

    source_and_destination_to_path: VertexDict[VertexDict[ShortestPath]] = VertexDict()

    for i in graph:
        source_and_destination_to_path[i] = VertexDict()
        dijkstra_paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(
            graph, source=i, weight=new_weight, save_paths=save_paths
        )
        for j in graph:
            source_and_destination_to_path[i][j] = dijkstra_paths[j]
            source_and_destination_to_path[i][j]._length += (
                bellman_paths[j].length - bellman_paths[i].length
            )

    return source_and_destination_to_path


def all_pairs_shortest_paths_johnson_fibonacci(
    graph: "GraphBase", weight: str = "Edge__weight", save_paths: bool = False
) -> "VertexDict[VertexDict[ShortestPath]]":
    """Finds the shortest paths between all pairs of vertices in a graph using Donald Johnson's
    algorithm implemented with a Fibonacci heap version of Dijkstra's algorithm.

    Running time: :math:`O((n^2)log(n) + mn)` where :math:`m = |E|` and :math:`n = |V|`

    Pairs of vertices for which there is no connecting path will have path length infinity. In
    additional, `ShortestPath.is_destination_reachable()` will return False.

    Note:
        This implementation is based on JOHNSON [CLRS2009_2]_.

    Args:
        graph: The graph to search.
        weight: Optional; The key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge__weight``) uses the property ``Edge.weight``.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path`. Defaults to False.

    Returns:
        VertexDict[VertexDict[ShortestPath]]: A dictionary mapping source vertices to dictionaries
        mapping destination vertices to :class:`ShortestPath
        <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath>` objects.

    Raises:
        NegativeWeightCycle: If the graph contains a negative weight cycle. **Note that for
            undirected graphs, any negative weight edge is a negative weight cycle.**

    See Also:
        * :func:`all_pairs_shortest_paths_floyd_warshall`
        * :func:`all_pairs_shortest_paths_johnson`
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :func:`reconstruct_path <vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path>`
        * :class:`ShortestPath <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath>`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`
    """
    weight_function = get_weight_function_all_pairs_shortest_paths(weight)

    g_prime: GraphBase = graph.deepcopy()
    G_PRIME_SOURCE = "__g_prime_src"
    for v in g_prime.vertices:
        g_prime.add_edge(G_PRIME_SOURCE, v, weight=0)

    bellman_paths: VertexDict[ShortestPath] = shortest_paths_bellman_ford(g_prime, G_PRIME_SOURCE)

    # pylint: disable=unused-argument
    def new_weight(v1: VertexType, v2: VertexType, reverse_graph: bool = False) -> float:
        edge: Edge = graph[v1, v2]
        return weight_function(edge) + bellman_paths[v1].length - bellman_paths[v2].length

    source_and_destination_to_path: VertexDict[VertexDict[ShortestPath]] = VertexDict()

    for i in graph:
        source_and_destination_to_path[i] = VertexDict()
        dijkstra_paths: VertexDict[ShortestPath] = shortest_paths_dijkstra_fibonacci(
            graph, source=i, weight=new_weight, save_paths=save_paths
        )
        for j in graph:
            source_and_destination_to_path[i][j] = dijkstra_paths[j]
            source_and_destination_to_path[i][j]._length += (
                bellman_paths[j].length - bellman_paths[i].length
            )

    return source_and_destination_to_path


def shortest_paths_bellman_ford(
    graph: "GraphBase",
    source: "VertexType",
    weight: Union[Callable, str] = "Edge__weight",
    reverse_graph: bool = False,
    save_paths: bool = False,
) -> "VertexDict[ShortestPath]":
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices of a weighted graph using the Bellman-Ford algorithm.

    Running time: :math:`O(mn)` where :math:`m = |E|` and :math:`n = |V|`

    The Bellman-Ford algorithm is not as fast as Dijkstra, but it can handle negative edge weights.

    Unreachable vertices will have a path length of infinity. In additional,
    :func:`ShortestPath.is_destination_reachable()
    <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath.is_destination_reachable>`
    will return False.

    The :class:`Edge <vertizee.classes.edge.Edge>` class has a built-in ``weight`` property, which
    is used by default to determine edge weights (i.e. edge lengths). Alternatively, a weight
    function may be specified that accepts two vertices and returns the weight of the connecting
    edge. See :func:`get_weight_function`.

    Note:
        This implementation is based on BELLMAN-FORD [CLRS2009_3]_.

    Args:
        graph (GraphBase): The graph to search.
        source (VertexType): The source vertex from which to find shortest paths to all other
            reachable vertices.
        weight: Optional; If callable, then `weight` must be a function
            accepting two Vertex objects (edge endpoints) that returns an edge weight (or length).
            If a string is specified, it is the key to use to retrieve the weight from the
            ``Edge.attr`` dictionary. The default value (``Edge__weight``) uses the property
            ``Edge.weight``.
        reverse_graph (bool, optional): For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path`.
            Defaults to False.

    Returns:
        VertexDict[ShortestPath]: A dictionary mapping vertices to their shortest paths relative to
        the ``source`` vertex.

    Raises:
        NegativeWeightCycle: If the graph contains a negative weight cycle. **Note that for
            undirected graphs, any negative weight edge is a negative weight cycle.**

    See Also:
        * :func:`get_weight_function`
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :func:`reconstruct_path
          <vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path>`
        * :class:`ShortestPath
          <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath>`
        * :func:`shortest_paths_dijkstra`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`

    Example:
        >>> g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])
        >>> paths: VertexDict[ShortestPath] = \
            shortest_paths_bellman_ford(g, 's', save_paths=True)
        >>> len(paths)
        5
        >>> paths['s'].length
        0
        >>> paths['y'].path
        [s, y]
        >>> paths['y'].length
        5
        >>> paths['x'].path
        [s, y, t, x]
        >>> paths['x'].length
        9

    References:
     .. [CLRS2009_3] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                     Introduction to Algorithms: Third Edition, page 651. The MIT Press, 2009.
    """
    try:
        s: Vertex = graph[source]
    except KeyError:
        raise VertexNotFound("source vertex was not found in the graph")
    weight_function = get_weight_function(weight)
    vertex_to_path_map: VertexDict[ShortestPath] = VertexDict()

    for v in graph:
        vertex_to_path_map[v] = ShortestPath(s, v, initial_length=INFINITY, save_paths=save_paths)
    vertex_to_path_map[s].reinitialize(initial_length=0)

    u_path: ShortestPath
    w_path: ShortestPath
    for _ in range(graph.vertex_count):
        for e in graph.edges:
            u_path = vertex_to_path_map[e.vertex1]
            w_path = vertex_to_path_map[e.vertex2]

            if reverse_graph:
                u_path, w_path = w_path, u_path
            w_path.relax_edge(u_path, weight_function=weight_function, reverse_graph=reverse_graph)
            if not graph.is_directed_graph():
                u_path.relax_edge(
                    w_path, weight_function=weight_function, reverse_graph=reverse_graph
                )

    for e in graph.edges:
        u = e.vertex1
        w = e.vertex2
        u_path = vertex_to_path_map[u]
        w_path = vertex_to_path_map[w]
        if reverse_graph:
            u_path, w_path = w_path, u_path
            u, w = w, u
        weight_u_w = weight_function(u, w, reverse_graph)
        if w_path.length > u_path.length + weight_u_w:
            raise vertizee.NegativeWeightCycle("found a negative weight cycle")

    return vertex_to_path_map


def shortest_paths_dijkstra(
    graph: "GraphBase",
    source: "VertexType",
    weight: Union[Callable, str] = "Edge__weight",
    reverse_graph: bool = False,
    save_paths: bool = False,
) -> "VertexDict[ShortestPath]":
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices of a graph with positive edge weights using Dijkstra's algorithm.

    Running time: :math:`O((m + n)log(n))` where :math:`m = |E|` and :math:`n = |V|`. Running time
    is due to implementation using a minimum priority queue based on a binary heap. For an
    implementation built using a Fibonacci heap and corresponding running time of
    :math:`O(n(log(n)) + m)`, see :func:`shortest_paths_dijkstra_fibonacci`.

    This algorithm is not guaranteed to work if edge weights are negative or are floating point
    numbers (overflows and roundoff errors can cause problems). To handle negative edge weights,
    see :func:`shortest_paths_bellman_ford`.

    Unreachable vertices will have a path length of infinity. In additional,
    :func:`ShortestPath.is_destination_reachable()
    <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath.is_destination_reachable>`
    will return False.

    The :class:`Edge <vertizee.classes.edge.Edge>` class has a built-in ``weight`` property, which
    is used by default to determine edge weights (i.e. edge lengths). Alternatively, a weight
    function may be specified that accepts two vertices and returns the weight of the connecting
    edge. See :func:`get_weight_function`.

    Note:
        This implementation is based on DIJKSTRA [CLRS2009_4]_.

    Args:
        graph: The graph to search.
        source: The source vertex from which to find shortest paths to all other
            reachable vertices.
        weight: Optional; If callable, then `weight` must be a function
            accepting two Vertex objects (edge endpoints) that returns an edge weight (or length).
            If a string is specified, it is the key to use to retrieve the weight from the
            ``Edge.attr`` dictionary. The default value (``Edge__weight``) uses the property
            ``Edge.weight``.
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path`.
            Defaults to False.

    Returns:
        VertexDict[ShortestPath]: A dictionary mapping vertices to their shortest paths relative to
        the ``source`` vertex.

    See Also:
        * :class:`DiEdge <vertizee.classes.edge.DiEdge>`
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :func:`reconstruct_path
          <vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path>`
        * :class:`ShortestPath
          <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath>`
        * :func:`shortest_paths_bellman_ford`
        * :func:`shortest_paths_dijkstra_fibonacci`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`

    Example:
        >>> g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])
        >>> paths: VertexDict[ShortestPath] = \
            shortest_paths_dijkstra(g, 's', save_paths=True)
        >>> len(paths)
        5
        >>> paths['s'].length
        0
        >>> paths['y'].path
        [s, y]
        >>> paths['y'].length
        5
        >>> paths['x'].path
        [s, y, t, x
        >>> paths['x'].length
        9

    References:
     .. [CLRS2009_4] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                     Introduction to Algorithms: Third Edition, page 658. The MIT Press, 2009.
    """
    try:
        s: Vertex = graph[source]
    except KeyError:
        raise VertexNotFound("source vertex was not found in the graph")
    weight_function = get_weight_function(weight)

    vertex_to_path_map: VertexDict[ShortestPath] = VertexDict()
    priority_queue: PriorityQueue[ShortestPath] = PriorityQueue(lambda path: path.length)

    for v in graph:
        vertex_path = ShortestPath(s, v, initial_length=INFINITY, save_paths=save_paths)
        vertex_to_path_map[v] = vertex_path
        priority_queue.add_or_update(vertex_path)
    vertex_to_path_map[s].reinitialize(initial_length=0)

    priority_queue.add_or_update(vertex_to_path_map[s])
    set_of_min_path_vertices = set()

    while len(priority_queue) > 0:
        u_path = priority_queue.pop()
        u: Vertex = u_path.destination
        set_of_min_path_vertices.add(u)
        u_adj_list = u.get_adj_for_search(parent=u_path.predecessor, reverse_graph=reverse_graph)
        while u_adj_list:
            w = u_adj_list.pop()
            if w in set_of_min_path_vertices:
                continue
            w_path = vertex_to_path_map[w]
            relaxed = w_path.relax_edge(u_path, weight_function, reverse_graph)
            if relaxed:
                priority_queue.add_or_update(w_path)

    return vertex_to_path_map


def shortest_paths_dijkstra_fibonacci(
    graph: "GraphBase",
    source: "VertexType",
    weight: Union[Callable, str] = "Edge__weight",
    reverse_graph: bool = False,
    save_paths: bool = False,
) -> "VertexDict[ShortestPath]":
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices of a graph with positive edge weights using Dijkstra's algorithm.

    Running time: :math:`O(n(log(n)) + m)` where :math:`m = |E|` and :math:`n = |V|`. Running time
    is due to implementation using a minimum priority queue based on a Fibonacci heap. For an
    implementation using a binary heap and corresponding running time of math:`O((m + n)log(n))`,
    see :func:`shortest_paths_dijkstra`.

    This algorithm is *not* guaranteed to work if edge weights are negative or are floating point
    numbers (overflows and roundoff errors can cause problems). To handle negative edge weights,
    see :func:`shortest_paths_bellman_ford`.

    Unreachable vertices will have a path length of infinity. In additional,
    :func:`ShortestPath.is_destination_reachable()
    <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath.is_destination_reachable>`
    will return False.

    The Edge class has a built-in ``weight`` property, which is used by default to determine edge
    weights (i.e. edge lengths). Alternatively, a weight function may be specified that accepts
    two vertices and returns the weight of the connecting edge. See :func:`get_weight_function`.

    Note:
        This implementation is based on DIJKSTRA [CLRS2009_4]_.

    Args:
        graph: The graph to search.
        source: The source vertex from which to find shortest paths to all other
            reachable vertices.
        weight: Optional; If callable, then `weight` must be a function
            accepting two Vertex objects (edge endpoints) that returns an edge weight (or length).
            If a string is specified, it is the key to use to retrieve the weight from the
            ``Edge.attr`` dictionary. The default value (``Edge__weight``) uses the property
            ``Edge.weight``.
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path`.
            Defaults to False.

    Returns:
        VertexDict[ShortestPath]: A dictionary mapping vertices to their shortest paths relative to
        the ``source`` vertex.

    See Also:
        * :class:`DiEdge <vertizee.classes.edge.DiEdge>`
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :func:`reconstruct_path
          <vertizee.algorithms.algo_utils.shortest_path_utils.reconstruct_path>`
        * :class:`ShortestPath
          <vertizee.algorithms.algo_utils.shortest_path_utils.ShortestPath>`
        * :func:`shortest_paths_bellman_ford`
        * :func:`shortest_paths_dijkstra`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`
    """
    #
    # TODO(cpeisert): run benchmarks.
    #
    try:
        s: Vertex = graph[source]
    except KeyError:
        raise VertexNotFound("source vertex was not found in the graph")
    weight_function = get_weight_function(weight)

    vertex_to_path_map: VertexDict[ShortestPath] = VertexDict()
    fib_heap: FibonacciHeap[ShortestPath] = FibonacciHeap(lambda path: path.length)

    for v in graph:
        vertex_path = ShortestPath(s, v, initial_length=INFINITY, save_paths=save_paths)
        vertex_to_path_map[v] = vertex_path
        fib_heap.insert(vertex_path)
    vertex_to_path_map[s].reinitialize(initial_length=0)

    fib_heap.update_item_with_decreased_priority(vertex_to_path_map[s])
    set_of_min_path_vertices = set()

    while len(fib_heap) > 0:
        u_path = fib_heap.extract_min()
        assert u_path is not None  # For mypy static type checker.
        u: Vertex = u_path.destination
        set_of_min_path_vertices.add(u)
        u_adj_list = u.get_adj_for_search(parent=u_path.predecessor, reverse_graph=reverse_graph)
        while u_adj_list:
            w = u_adj_list.pop()
            if w in set_of_min_path_vertices:
                continue
            w_path = vertex_to_path_map[w]
            relaxed = w_path.relax_edge(u_path, weight_function, reverse_graph)
            if relaxed:
                fib_heap.update_item_with_decreased_priority(w_path)

    return vertex_to_path_map
