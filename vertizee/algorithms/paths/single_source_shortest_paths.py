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

"""Algorithms for the single-source-shortest-paths problem.

Note:
    For asymptotic runtime analysis, :math:`m = |E|` (the number of edges) and :math:`n = |V|`
    (the number of vertices).

* :func:`shortest_paths` - Finds the shortest paths and associated lengths from the source vertex
  to all reachable vertices. This function chooses the fastest available single-source-shortest-path
  algorithm depending on the properties of the graph.
* :func:`bellman_ford` - Finds the shortest paths in a weighted graph using the Bellman-Ford
  algorithm. Running time: :math:`O(mn)`
* :func:`breadth_first_search_shortest_paths` - Finds the shortest paths in an unweighted graph
  using a breadth-first search. Running time: :math:`O(m + n)`
* :func:`dijkstra` - Finds the shortest paths in a graph with positive edge weights using Dijkstra's
  algorithm. Running time: :math:`O((m + n)log(n))`
* :func:`dijkstra_fibonacci` - Finds the shortest paths in a graph with positive edge weights using
  Dijkstra's algorithm implemented using a Fibonacci-heap-based priority queue. Running time:
  :math:`O(n(log(n)) + m)`
"""

from __future__ import annotations
from typing import Callable, Final, Iterator, Optional, TYPE_CHECKING, Union

from vertizee import exception
from vertizee.algorithms.algo_utils.path_utils import ShortestPath
from vertizee.algorithms.algo_utils import search_utils
from vertizee.algorithms.search import breadth_first_search
from vertizee.classes.data_structures.fibonacci_heap import FibonacciHeap
from vertizee.classes.data_structures.priority_queue import PriorityQueue
from vertizee.classes.data_structures.vertex_dict import VertexDict

if TYPE_CHECKING:
    from vertizee.classes.edge import E
    from vertizee.classes.graph import G
    from vertizee.classes.vertex import V, VertexType

INFINITY: Final = float("inf")


def get_weight_function(
    weight: Union[Callable, str] = "Edge__weight"
) -> Callable[[V, V, bool], float]:
    """Returns a function that accepts two vertices and a boolean indicating if the graph should be
    treated as if it were reversed (i.e. edges of directed graphs in the opposite direction) and
    returns the corresponding edge weight.

    If there is no edge weight, then the edge weight is assumed to be one.

    Note:
        For multigraphs, the minimum edge weight among the parallel edge connections is returned.

    Note:
        To support reversed graphs, custom weight functions should implement the following pattern:

        .. code-block:: python

            def get_weight(v1: V, v2: V, reverse_graph: bool) -> float:
                graph: G[V, E] = v1._parent_graph
                if reverse_graph:
                    if not graph.has_edge(v2, v1):
                        raise AlgorithmError(f"edge ({v2.label}, {v1.label}) not in graph")
                    edge = graph[v2, v1]
                else:
                    if not graph.has_edge(v1, v2):
                        raise AlgorithmError(f"edge ({v1.label}, {v2.label}) not in graph")
                    edge = graph[v1, v2]

                <YOUR CODE HERE>

                return min_weight

        The weight function may also serve as a filter by returning ``None`` for any edge that
        should be excluded from the shortest path search.  For example, adding the following would
        exclude blue edges:

        .. code-block:: python

            if edge.attr.get('color', 'no color attribute') == 'blue':
                return None

    Args:
        weight: Optional; If callable, then ``weight`` itself is returned. If
            a string is specified, it is the key to use to retrieve the weight from an ``Edge.attr``
            dictionary. The default value (``Edge__weight``) returns a function that accesses the
            ``Edge.weight`` property.

    Returns:
        Callable[[V, V, bool], float]: A function that accepts two vertices
        and a boolean indicating if the graph is reversed (i.e. edges of directed graphs in the
        opposite direction) and returns the corresponding edge weight.
    """
    if callable(weight):
        return weight

    if not isinstance(weight, str):
        raise ValueError("'weight' must be a callable function or a string")

    def get_weight(v1: V, v2: V, reverse_graph: bool) -> float:
        graph: G[V, E] = v1._parent_graph
        if reverse_graph:
            if not graph.has_edge(v2, v1):
                raise exception.AlgorithmError(f"edge ({v2.label}, {v1.label}) not in graph")
            edge = graph[v2, v1]
        else:
            if not graph.has_edge(v1, v2):
                raise exception.AlgorithmError(f"edge ({v1.label}, {v2.label}) not in graph")
            edge = graph[v1, v2]

        if graph.is_multigraph():
            if weight == "Edge__weight":
                min_weight = min(c.weight for c in edge.connections())
            else:
                min_weight = min(c.attr.get(weight, 1.0) for c in edge.connections())
        else:
            if weight == "Edge__weight":
                min_weight = edge.weight
            else:
                min_weight = edge.attr.get(weight, 1.0)

        return min_weight

    return get_weight


def shortest_paths(
    graph: G[V, E],
    source: VertexType,
    save_paths: bool = False,
    reverse_graph: bool = False,
    weight: Union[Callable, str] = "Edge__weight"
) -> VertexDict[ShortestPath[V]]:
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices.

    Note:
        For weighted graphs that use custom weight attributes (instead of the built-in ``weight``
        attribute of the edge classes), this function may select the wrong algorithm. If there is
        doubt about which algorithm to use, choose the :func:`Bellman-Ford algorithm
        <bellman_ford>`, which will provide the correct shortest paths for both unweighted and
        weighted graphs (including negative edge weights).

    This function chooses the fastest available single-source-shortest-path algorithm depending on
    the properties of the graph. Note that :math:`m = |E|` (the number of edges) and :math:`n = |V|`
    (the number of vertices):

        * unweighted - :func:`Breadth-first search <breadth_first_search_shortest_paths>` is used
          for unweighted graphs. Running time: :math:`O(m + n)`
        * weighted (positive weights only) - :func:`Dijkstra's algorithm <dijkstra>`
          is used for weighted graphs that only contain positive edge weights. Running time:
          :math:`O((m + n)log(n))`
        * weighted (contains negative edge weights) - The :func:`Bellman-Ford algorithm
          <bellman_ford>` is used for weighted graphs that contain at least one negative edge
          weight. Running time: :math:`O(mn)`

    Unreachable vertices will have a path length of infinity. In additional,
    :func:`ShortestPath.is_destination_reachable
    <vertizee.algorithms.algo_utils.path_utils.ShortestPath.is_destination_reachable>`
    will return False.

    The edge classes have a built-in ``weight`` property, which is used by default to determine
    edge weights (i.e. edge lengths). Alternatively, a weight function may be specified that
    accepts two vertices and returns the weight of the edge. See :func:`get_weight_function`.

    Args:
        graph: The graph to search.
        source: The source vertex from which to find shortest paths to all other reachable vertices.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each path. To
            reconstruct specific shortest paths, see :func:`reconstruct_path
            <vertizee.algorithms.algo_utils.path_utils.reconstruct_path>`. Defaults to False.
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.
        weight: Optional; If callable, then ``weight`` must be a function accepting two vertex
            objects (edge endpoints) that returns an edge weight (or length). If a string is
            specified, it is the key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge__weight``) uses the property ``Edge.weight``.

    Returns:
        VertexDict[ShortestPath[V]]: A dictionary mapping vertices to their shortest paths relative
        to the ``source`` vertex.

    Raises:
        NegativeWeightCycle: If the graph contains a negative weight cycle. **Note that for
            undirected graphs, any negative weight edge is a negative weight cycle.**

    See Also:
        * :func:`get_weight_function`
        * :func:`reconstruct_path
          <vertizee.algorithms.algo_utils.path_utils.reconstruct_path>`
        * :func:`bellman_ford`
        * :func:`dijkstra`
        * :func:`breadth_first_search_shortest_paths`
        * :class:`ShortestPath
          <vertizee.algorithms.algo_utils.path_utils.ShortestPath>`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`

    Example:
        >>> import vertizee as vz
        >>> from vertizee.algorithms.paths import shortest_paths, ShortestPath
        >>> g = vz.DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])
        >>> path_dict: vz.VertexDict[ShortestPath] = shortest_paths(g, 's', save_paths=True)
        >>> len(path_dict)
        5
        >>> path_dict['s'].length
        0
        >>> path_dict['y'].path()
        [s, y]
        >>> path_dict['y'].length
        5
        >>> path_dict['x'].path()
        [s, y, t, x]
        >>> path_dict['x'].length
        9
        >>> path_dict['x'].edge_count
        3
    """
    if graph.is_weighted():
        if graph.has_negative_edge_weights():
            return bellman_ford(graph, source, save_paths, reverse_graph, weight)
        # Positive edge weights.
        return dijkstra(graph, source, save_paths, reverse_graph, weight)
    # Unweighted graph.
    return breadth_first_search_shortest_paths(graph, source, save_paths, reverse_graph)


def bellman_ford(
    graph: G[V, E],
    source: VertexType,
    save_paths: bool = False,
    reverse_graph: bool = False,
    weight: Union[Callable, str] = "Edge__weight"
) -> VertexDict[ShortestPath[V]]:
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices in a weighted graph using the Bellman-Ford algorithm.

    Running time: :math:`O(mn)` where :math:`m = |E|` and :math:`n = |V|`

    The Bellman-Ford algorithm is not as fast as Dijkstra, but it can handle negative edge weights.

    Unreachable vertices will have a path length of infinity. In additional,
    :func:`ShortestPath.is_destination_reachable()
    <vertizee.algorithms.algo_utils.path_utils.ShortestPath.is_destination_reachable>`
    will return False.

    The edge classes have a built-in ``weight`` property, which is used by default to determine
    edge weights (i.e. edge lengths). Alternatively, a weight function may be specified that
    accepts two vertices and returns the weight of the connecting edge. See
    :func:`get_weight_function`.

    Note:
        This implementation is based on BELLMAN-FORD [CLRS2009_3]_.

    Args:
        graph: The graph to search.
        source: The source vertex from which to find shortest paths to all other reachable vertices.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each path. To
            reconstruct specific shortest paths, see :func:`reconstruct_path
            <vertizee.algorithms.algo_utils.path_utils.reconstruct_path>`. Defaults to False.
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.
        weight: Optional; If callable, then ``weight`` must be a function accepting two vertex
            objects (edge endpoints) that returns an edge weight (or length). If a string is
            specified, it is the key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge__weight``) uses the property ``Edge.weight``.

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
          <vertizee.algorithms.algo_utils.path_utils.reconstruct_path>`
        * :class:`ShortestPath
          <vertizee.algorithms.algo_utils.path_utils.ShortestPath>`
        * :func:`dijkstra`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`

    References:
     .. [CLRS2009_3] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                     Introduction to Algorithms: Third Edition, page 651. The MIT Press, 2009.
    """
    try:
        s: V = graph[source]
    except KeyError as error:
        raise exception.VertexNotFound("source vertex was not found in the graph") from error

    if not graph.is_directed() and graph.has_negative_edge_weights():
        raise exception.NegativeWeightCycle("found a negative weight cycle")

    weight_function = get_weight_function(weight)
    vertex_to_path_map: VertexDict[ShortestPath] = VertexDict()

    for v in graph:
        vertex_to_path_map[v] = ShortestPath(s, v, initial_length=INFINITY, save_path=save_paths)
    vertex_to_path_map[s].reinitialize(initial_length=0)

    u_path: ShortestPath
    w_path: ShortestPath
    for _ in range(graph.vertex_count):
        for e in graph.edges():
            u_path = vertex_to_path_map[e.vertex1]
            w_path = vertex_to_path_map[e.vertex2]

            if reverse_graph:
                u_path, w_path = w_path, u_path
            w_path.relax_edge(u_path, weight_function=weight_function, reverse_graph=reverse_graph)
            if not graph.is_directed():
                u_path.relax_edge(
                    w_path, weight_function=weight_function, reverse_graph=reverse_graph
                )

    for e in graph.edges():
        u = e.vertex1
        w = e.vertex2
        u_path = vertex_to_path_map[u]
        w_path = vertex_to_path_map[w]
        if reverse_graph:
            u_path, w_path = w_path, u_path
            u, w = w, u
        weight_u_w = weight_function(u, w, reverse_graph)
        if w_path.length > u_path.length + weight_u_w:
            raise exception.NegativeWeightCycle("found a negative weight cycle")

    return vertex_to_path_map


def breadth_first_search_shortest_paths(
    graph: G[V, E], source: VertexType, save_paths: bool = False, reverse_graph: bool = False
) -> VertexDict[ShortestPath[V]]:
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices in an unweighted graph using a breadth-first search.

    Running time: :math:`O(m + n)` where :math:`m = |E|` and :math:`n = |V|`

    Unreachable vertices will have an empty list of vertices for their path and a length of
    infinity (``float("inf")``). In additional, ``ShortestPath.is_unreachable()`` will return True.

    Args:
        graph: The graph to search.
        source: The source vertex from which to find shortest paths to all other reachable vertices.
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.path_utils.reconstruct_path`.
            Defaults to False.
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Returns:
        VertexDict[ShortestPath]: A dictionary mapping vertices to their shortest paths relative to
        the ``source`` vertex.

    See Also:
        * :class:`ShortestPath
          <vertizee.algorithms.algo_utils.path_utils.ShortestPath>`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`

    Example:
        >>> import vertizee as vz
        >>> from vertizee.algorithms.paths import breadth_first_search_shortest_paths, ShortestPath
        >>> g = vz.Graph()
        >>> g.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (6, 7)])
        >>> path_dict: vz.VertexDict[ShortestPath] = breadth_first_search_shortest_paths(g, 0)
        >>> path_dict[4].path()
        [0, 1, 3, 4]
        >>> path_dict[4].length
        3
        >>> path_dict[6].path()
        []
        >>> path_dict[6].is_unreachable()
        True
    """
    try:
        s: V = graph[source]
    except KeyError as error:
        raise exception.VertexNotFound(f"source vertex '{source}' not found in graph") from error
    vertex_to_path_map: VertexDict[ShortestPath[V]] = VertexDict()

    for v in graph:
        vertex_to_path_map[v] = ShortestPath(s, v, initial_length=INFINITY, save_path=save_paths)
    vertex_to_path_map[s].reinitialize(initial_length=0)

    tuple_generator = breadth_first_search.bfs_labeled_edge_traversal(
        graph, source, reverse_graph=reverse_graph)
    bfs_tree = ((parent, child) for parent, child, label, direction, depth
        in tuple_generator if direction == search_utils.Direction.PREORDER)

    for parent, child in bfs_tree:
        vertex_to_path_map[child].relax_edge(vertex_to_path_map[parent], lambda j, k, rev: 1)

    return vertex_to_path_map


def dijkstra(
    graph: G[V, E],
    source: VertexType,
    save_paths: bool = False,
    reverse_graph: bool = False,
    weight: Union[Callable, str] = "Edge__weight"
) -> VertexDict[ShortestPath[V]]:
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices in a graph with positive edge weights using Dijkstra's algorithm.

    Running time: :math:`O((m + n)log(n))` where :math:`m = |E|` and :math:`n = |V|`. Running time
    is due to implementation using a minimum priority queue based on a binary heap. For an
    implementation built using a Fibonacci heap and corresponding running time of
    :math:`O(n(log(n)) + m)`, see :func:`dijkstra_fibonacci`.

    This algorithm is not guaranteed to work if edge weights are negative or are floating point
    numbers (overflows and roundoff errors can cause problems). To handle negative edge weights,
    see :func:`bellman_ford`.

    Unreachable vertices will have a path length of infinity. In additional,
    :func:`ShortestPath.is_destination_reachable()
    <vertizee.algorithms.algo_utils.path_utils.ShortestPath.is_destination_reachable>`
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
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.path_utils.reconstruct_path`.
            Defaults to False.
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.
        weight: Optional; If callable, then `weight` must be a function
            accepting two Vertex objects (edge endpoints) that returns an edge weight (or length).
            If a string is specified, it is the key to use to retrieve the weight from the
            ``Edge.attr`` dictionary. The default value (``Edge__weight``) uses the property
            ``Edge.weight``.

    Returns:
        VertexDict[ShortestPath]: A dictionary mapping vertices to their shortest paths relative to
        the ``source`` vertex.

    See Also:
        * :class:`DiEdge <vertizee.classes.edge.DiEdge>`
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :func:`reconstruct_path
          <vertizee.algorithms.algo_utils.path_utils.reconstruct_path>`
        * :class:`ShortestPath
          <vertizee.algorithms.algo_utils.path_utils.ShortestPath>`
        * :func:`bellman_ford`
        * :func:`dijkstra_fibonacci`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`

    References:
     .. [CLRS2009_4] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
                     Introduction to Algorithms: Third Edition, page 658. The MIT Press, 2009.
    """
    try:
        s: V = graph[source]
    except KeyError as error:
        raise exception.VertexNotFound("source vertex was not found in the graph") from error
    weight_function = get_weight_function(weight)

    vertex_to_path_map: VertexDict[ShortestPath] = VertexDict()
    priority_queue: PriorityQueue[ShortestPath] = PriorityQueue(lambda path: path.length)

    for v in graph:
        vertex_path = ShortestPath(s, v, initial_length=INFINITY, save_path=save_paths)
        vertex_to_path_map[v] = vertex_path
        priority_queue.add_or_update(vertex_path)
    vertex_to_path_map[s].reinitialize(initial_length=0)

    priority_queue.add_or_update(vertex_to_path_map[s])
    set_of_min_path_vertices = set()

    while priority_queue:
        u_path = priority_queue.pop()
        u: V = u_path.destination
        set_of_min_path_vertices.add(u)
        u_adj_iter = _get_adjacent_to_child(
            child=u, parent=u_path.predecessor, reverse_graph=reverse_graph)
        for w in u_adj_iter:
            if w in set_of_min_path_vertices:
                continue
            w_path = vertex_to_path_map[w]
            relaxed = w_path.relax_edge(u_path, weight_function, reverse_graph)
            if relaxed:
                priority_queue.add_or_update(w_path)

    return vertex_to_path_map


def dijkstra_fibonacci(
    graph: G[V, E],
    source: VertexType,
    save_paths: bool = False,
    reverse_graph: bool = False,
    weight: Union[Callable, str] = "Edge__weight"
) -> VertexDict[ShortestPath[V]]:
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices in a graph with positive edge weights using Dijkstra's algorithm.

    Running time: :math:`O(n(log(n)) + m)` where :math:`m = |E|` and :math:`n = |V|`. Running time
    is due to implementation using a minimum priority queue based on a Fibonacci heap. For an
    implementation using a binary heap and corresponding running time of math:`O((m + n)log(n))`,
    see :func:`dijkstra`.

    This algorithm is *not* guaranteed to work if edge weights are negative or are floating point
    numbers (overflows and roundoff errors can cause problems). To handle negative edge weights,
    see :func:`bellman_ford`.

    Unreachable vertices will have a path length of infinity. In additional,
    :func:`ShortestPath.is_destination_reachable()
    <vertizee.algorithms.algo_utils.path_utils.ShortestPath.is_destination_reachable>`
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
        save_paths: Optional; If True, saves the actual vertex sequences comprising each
            path. To reconstruct specific shortest paths, see
            :func:`vertizee.algorithms.algo_utils.path_utils.reconstruct_path`.
            Defaults to False.
        reverse_graph: Optional; For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.
        weight: Optional; If callable, then `weight` must be a function
            accepting two Vertex objects (edge endpoints) that returns an edge weight (or length).
            If a string is specified, it is the key to use to retrieve the weight from the
            ``Edge.attr`` dictionary. The default value (``Edge__weight``) uses the property
            ``Edge.weight``.

    Returns:
        VertexDict[ShortestPath]: A dictionary mapping vertices to their shortest paths relative to
        the ``source`` vertex.

    See Also:
        * :class:`DiEdge <vertizee.classes.edge.DiEdge>`
        * :class:`Edge <vertizee.classes.edge.Edge>`
        * :func:`reconstruct_path
          <vertizee.algorithms.algo_utils.path_utils.reconstruct_path>`
        * :class:`ShortestPath
          <vertizee.algorithms.algo_utils.path_utils.ShortestPath>`
        * :func:`bellman_ford`
        * :func:`dijkstra`
        * :class:`VertexDict <vertizee.classes.data_structures.vertex_dict.VertexDict>`
    """
    #
    # TODO(cpeisert): run benchmarks.
    #
    try:
        s: V = graph[source]
    except KeyError as error:
        raise exception.VertexNotFound("source vertex was not found in the graph") from error
    weight_function = get_weight_function(weight)

    vertex_to_path_map: VertexDict[ShortestPath] = VertexDict()
    fib_heap: FibonacciHeap[ShortestPath] = FibonacciHeap(lambda path: path.length)

    for v in graph:
        vertex_path = ShortestPath(s, v, initial_length=INFINITY, save_path=save_paths)
        vertex_to_path_map[v] = vertex_path
        fib_heap.insert(vertex_path)
    vertex_to_path_map[s].reinitialize(initial_length=0)

    fib_heap.update_item_with_decreased_priority(vertex_to_path_map[s])
    set_of_min_path_vertices = set()

    while len(fib_heap) > 0:
        u_path = fib_heap.extract_min()
        assert u_path is not None  # For mypy static type checker.
        u: V = u_path.destination
        set_of_min_path_vertices.add(u)
        u_adj_iter = _get_adjacent_to_child(
            child=u, parent=u_path.predecessor, reverse_graph=reverse_graph)
        for w in u_adj_iter:
            if w in set_of_min_path_vertices:
                continue
            w_path = vertex_to_path_map[w]
            relaxed = w_path.relax_edge(u_path, weight_function, reverse_graph)
            if relaxed:
                fib_heap.update_item_with_decreased_priority(w_path)

    return vertex_to_path_map


def _get_adjacent_to_child(
    child: V, parent: Optional[V], reverse_graph: bool
) -> Iterator[V]:
    if child._parent_graph.is_directed():
        if reverse_graph:
            return iter(child.adj_vertices_incoming())
        return iter(child.adj_vertices_outgoing())

    # undirected graph
    adj_vertices = child.adj_vertices()
    if parent:
        adj_vertices = adj_vertices - {parent}
    return iter(adj_vertices)
