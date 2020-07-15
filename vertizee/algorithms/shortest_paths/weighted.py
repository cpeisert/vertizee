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

from typing import Callable, Optional, Union

from vertizee.classes.collections.fibonacci_heap import FibonacciHeap
from vertizee.classes.collections.priority_queue import PriorityQueue
from vertizee.classes.collections.vertex_dict import VertexDict
from vertizee.classes.edge import EdgeType
from vertizee.classes.graph_base import GraphBase
from vertizee.classes.shortest_path import ShortestPath
from vertizee.classes.vertex import Vertex, VertexKeyType
from vertizee.exception import NegativeWeightCycle

INFINITY = float('inf')


def get_weight_function(
        weight: Union[Callable, str] = 'Edge__weight') -> Callable[[Vertex, Vertex, bool], float]:
    """Returns a function that accepts two vertices and a  boolean indicating if the graph is
    reversed (i.e. edges of directed graphs in the opposite direction) and returns the
    corresponding edge weight.

    If there is no edge weight, then the edge weight is assumed to be one.  If `graph` is a
    multigraph, the minimum edge weight over all parallel edges is returned.

    Notes:
        To support reversed graphs, custom weight functions should implement the following pattern:

        ```
        def get_min_weight(v1: Vertex, v2: Vertex, reverse_graph: bool) -> float:
            graph = v1._parent_graph
            if reverse_graph:
                edge: EdgeType = graph[v2][v1]
                edge_str = f'({v2.key}, {v1.key})'
            else:
                edge: EdgeType = graph[v1][v2]
                edge_str = f'({v1.key}, {v2.key})'
            if edge is None:
                raise ValueError(f'graph does not have edge {edge_str}')

            <YOUR CODE HERE>

            return min_weight
        ```

        The weight function may also serve as a filter by returning None for any edge that should
        be excluded from the shortest path search.  For example, adding the following would
        exclude blue edges:

        ```
        if edge.attr.get('color', 'red') == 'blue':
            return None
        ```

    Args:
        weight (Union[Callable, str]): If callable, then `weight` itself is returned. If a string
            is specified, it is the key to use to retrieve the weight from a `Edge.attr`
            dictionary. The default value ('Edge__weight') returns a function that accesses the
            `Edge.weight` property.
        reverse_graph (bool, optional): For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Returns:
        Callable[[Vertex, Vertex, bool], float]: A function that accepts two vertices and a
        boolean indicating if the graph is reversed (i.e. edges of directed graphs in the opposite
        direction) and returns the corresponding edge weight.
    """
    if callable(weight):
        return weight

    if not isinstance(weight, str):
        raise ValueError('`weight` must be a callable function or a string')

    def get_min_weight(v1: Vertex, v2: Vertex, reverse_graph: bool) -> float:
        graph = v1._parent_graph
        if reverse_graph:
            edge: EdgeType = graph[v2][v1]
            edge_str = f'({v2.key}, {v1.key})'
        else:
            edge: EdgeType = graph[v1][v2]
            edge_str = f'({v1.key}, {v2.key})'
        if edge is None:
            raise ValueError(f'graph does not have edge {edge_str}')
        if weight == 'Edge__weight':
            min_weight = edge.weight
        else:
            min_weight = edge.attr.get(weight, 1)

        if len(edge.parallel_edge_weights) > 0:
            min_parallel = min(edge.parallel_edge_weights)
            min_weight = min(min_weight, min_parallel)
        return min_weight

    return get_min_weight


def shortest_paths_bellman_ford(
        graph: GraphBase, source: VertexKeyType, weight: Union[Callable, str] = 'Edge__weight',
        reverse_graph: bool = False) -> VertexDict[ShortestPath]:
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices of a weighted, directed graph using the Bellman-Ford algorithm.

    Running time: O(mn) where m = |E| and n = |V|

    The Bellman-Ford algorithm is not as fast as Dijkstra, but it can handle negative edge weights.
    This implementation is based on "Introduction to Algorithms: Third Edition" [1].

    Unreachable vertices will have a path length of infinity. In additional,
    `ShortestPath.is_destination_unreachable` will return True.

    The Edge class has a built-in `weight` property, which is used by default to determine edge
    weights (i.e. edge lengths). Alternatively, a weight function may be specified that accepts
    two vertices and returns the weight of the connecting edge. See
    `~weighted.get_weight_function`.

    Args:
        graph (GraphBase): The graph to search.
        source (VertexKeyType): The source vertex from which to find shortest paths to all other
            reachable vertices.
        weight (Union[Callable, str]): If callable, then `weight` must be a function accepting two
            Vertex objects (edge endpoints) that returns an edge weight (or length). If a string
            is specified, it is the key to use to retrieve the weight from the `Edge.attr`
            dictionary. The default value ('Edge__weight') uses the property `Edge.weight`.
        reverse_graph (bool, optional): For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Returns:
        VertexDict[ShortestPath]: A dictionary mapping vertices to their shortest
            paths and associated path lengths.

    Raises:
        NegativeWeightCycle: If the graph contains a negative weight cycle. Note that for
            undirected graphs, any negative weight edge is a negative weight cycle.

    See Also:
        `~edge.Edge`
        `~shortest_path.ShortestPath`
        `~vertex_dict.VertexDict`
        `~weighted.get_weight_function`.
        `~weighted.shortest_paths_dijkstra`

    Example:
        >>> g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])
        >>> paths: VertexDict[ShortestPath] = shortest_paths_bellman_ford(g, 's')
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
        [1] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
            Introduction to Algorithms: Third Edition, page 651. The MIT Press, 2009.
    """
    s: Vertex = graph[source]
    if s is None:
        raise ValueError('source vertex not found in the graph')
    weight_function = get_weight_function(weight)
    vertex_to_path_map: VertexDict[ShortestPath] = VertexDict()

    for vertex in graph:
        path = ShortestPath(source=s, destination=vertex, initial_length=INFINITY)
        vertex_to_path_map[vertex] = path
    vertex_to_path_map[s].reinitialize(initial_length=0)

    for _ in range(graph.vertex_count):
        for e in graph.edges:
            u_path: ShortestPath = vertex_to_path_map[e.vertex1]
            w_path: ShortestPath = vertex_to_path_map[e.vertex2]

            if reverse_graph:
                u_path, w_path = w_path, u_path
            w_path.relax(u_path, weight_function=weight_function, reverse_graph=reverse_graph)
            if not graph.is_directed_graph():
                u_path.relax(w_path, weight_function=weight_function, reverse_graph=reverse_graph)

    for e in graph.edges:
        u = e.vertex1
        w = e.vertex2
        u_path: ShortestPath = vertex_to_path_map[u]
        w_path: ShortestPath = vertex_to_path_map[w]
        if reverse_graph:
            u_path, w_path = w_path, u_path
            u, w = w, u
        weight_u_w = weight_function(u, w, reverse_graph)
        if w_path.length > u_path.length + weight_u_w:
            raise NegativeWeightCycle('Bellman-Ford algorithm found a negative weight cycle')

    return vertex_to_path_map


def shortest_paths_dijkstra(
        graph: GraphBase, source: VertexKeyType, weight: Union[Callable, str] = 'Edge__weight',
        reverse_graph: bool = False) -> VertexDict[ShortestPath]:
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices of a graph with positive edge weights using Dijkstra's algorithm.

    Running time: O((m + n)log(n)) where m = |E| and n = |V|. Running time is due to implementation
    using a minimum priority queue based on a binary heap. For an implementation built using a
    Fibonacci heap and corresponding running time of O(n * log(n) + m), see
    `~weighted.shortest_paths_dijkstra_fibonacci`.

    This algorithm is not guaranteed to work if edge weights are negative or are floating point
    numbers (overflows and roundoff errors can cause problems). To handle negative edge weights,
    see `~weighted.shortest_paths_bellman_ford`.

    Unreachable vertices will have a path length of infinity. In additional,
    `ShortestPath.is_destination_unreachable` will return True.

    The Edge class has a built-in `weight` property, which is used by default to determine edge
    weights (i.e. edge lengths). Alternatively, a weight function may be specified that accepts
    two vertices and returns the weight of the connecting edge. See
    `~weighted.get_weight_function`.

    Args:
        graph (GraphBase): The graph to search.
        source (VertexKeyType): The source vertex from which to find shortest paths to all other
            reachable vertices.
        weight (Union[Callable, str], optional): If callable, then `weight` must be a function
            accepting two Vertex objects (edge endpoints) that returns an edge weight (or length).
            If a string is specified, it is the key to use to retrieve the weight from the
            `Edge.attr` dictionary. The default value ('Edge__weight') uses the property
            `Edge.weight`.
        reverse_graph (bool, optional): For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Returns:
        VertexDict[ShortestPath]: A dictionary mapping vertices to their shortest
            paths and associated path lengths.

    See Also:
        `~edge.Edge`
        `~edge.DiEdge`
        `~shortest_path.ShortestPath`
        `~vertex_dict.VertexDict`
        `~weighted.shortest_paths_bellman_ford`
        `~weighted.shortest_paths_dijkstra_fibonacci`

    Example:
        >>> g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])
        >>> paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(g, 's')
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
        [1] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
            Introduction to Algorithms: Third Edition, page 658. The MIT Press, 2009.
    """
    s: Vertex = graph[source]
    if s is None:
        raise ValueError('source vertex not found in the graph')
    weight_function = get_weight_function(weight)

    vertex_to_path_map: VertexDict[ShortestPath] = VertexDict()
    priority_queue: PriorityQueue[ShortestPath] = PriorityQueue(lambda path: path.length)

    for vertex in graph:
        vertex_path = ShortestPath(source=s, destination=vertex, initial_length=INFINITY)
        vertex_to_path_map[vertex] = vertex_path
        priority_queue.add_or_update(vertex_path)

    vertex_to_path_map[s].reinitialize(initial_length=0)
    priority_queue.add_or_update(vertex_to_path_map[s])

    set_of_min_path_vertices = set()

    while len(priority_queue) > 0:
        u_path = priority_queue.pop()
        u: Vertex = u_path.destination
        set_of_min_path_vertices.add(u)
        u_parent = _get_parent_vertex_of_path_destination(u_path)
        u_adj_list = u.get_adj_for_search(parent=u_parent, reverse_graph=reverse_graph)
        while u_adj_list:
            w = u_adj_list.pop()
            if w in set_of_min_path_vertices:
                continue
            w_path = vertex_to_path_map[w]
            relaxed = w_path.relax(u_path, weight_function, reverse_graph)
            if relaxed:
                priority_queue.add_or_update(w_path)

    return vertex_to_path_map


def shortest_paths_dijkstra_fibonacci(
        graph: GraphBase, source: VertexKeyType, weight: Union[Callable, str] = 'Edge__weight',
        reverse_graph: bool = False) -> VertexDict[ShortestPath]:
    """Finds the shortest paths and associated lengths from the source vertex to all reachable
    vertices of a graph with positive edge weights using Dijkstra's algorithm.

    Running time:  O(n * log(n) + m) where m = |E| and n = |V|. Running time is due to
    implementation using a minimum priority queue based on a Fibonacci heap. For an implementation
    using a binary heap and corresponding running time of O((m + n)log(n)), see
    `~weighted.shortest_paths_dijkstra`.

    This algorithm is not guaranteed to work if edge weights are negative or are floating point
    numbers (overflows and roundoff errors can cause problems). To handle negative edge weights,
    see `~weighted.shortest_paths_bellman_ford`.

    Unreachable vertices will have a path length of infinity. In additional,
    `ShortestPath.is_destination_unreachable` will return True.

    The Edge class has a built-in `weight` property, which is used by default to determine edge
    weights (i.e. edge lengths). Alternatively, a weight function may be specified that accepts
    two vertices and returns the weight of the connecting edge. See
    `~weighted.get_weight_function`.

    Args:
        graph (GraphBase): The graph to search.
        source (VertexKeyType): The source vertex from which to find shortest paths to all other
            reachable vertices.
        weight (Union[Callable, str], optional): If callable, then `weight` must be a function
            accepting two Vertex objects (edge endpoints) that returns an edge weight (or length).
            If a string is specified, it is the key to use to retrieve the weight from the
            `Edge.attr` dictionary. The default value ('Edge__weight') uses the property
            `Edge.weight`.
        reverse_graph (bool, optional): For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.

    Returns:
        VertexDict[ShortestPath]: A dictionary mapping vertices to their shortest
            paths and associated path lengths.

    See Also:
        `~edge.Edge`
        `~edge.DiEdge`
        `~shortest_path.ShortestPath`
        `~vertex_dict.VertexDict`
        `~weighted.shortest_paths_bellman_ford`
        `~weighted.shortest_paths_dijkstra`

    References:
        [1] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
            Introduction to Algorithms: Third Edition, page 658. The MIT Press, 2009.
    """
    #
    # TODO(cpeisert): run benchmarks.
    #
    s: Vertex = graph[source]
    if s is None:
        raise ValueError('source vertex not found in the graph')
    weight_function = get_weight_function(weight)

    vertex_to_path_map: VertexDict[ShortestPath] = VertexDict()
    fib_heap: FibonacciHeap[ShortestPath] = FibonacciHeap(lambda path: path.length)

    for vertex in graph:
        vertex_path = ShortestPath(source=s, destination=vertex, initial_length=INFINITY)
        vertex_to_path_map[vertex] = vertex_path
        fib_heap.insert(vertex_path)

    vertex_to_path_map[s].reinitialize(initial_length=0)
    fib_heap.update_item_with_decreased_priority(vertex_to_path_map[s])

    set_of_min_path_vertices = set()

    while len(fib_heap) > 0:
        u_path = fib_heap.extract_min()
        u: Vertex = u_path.destination
        set_of_min_path_vertices.add(u)
        u_parent = _get_parent_vertex_of_path_destination(u_path)
        u_adj_list = u.get_adj_for_search(parent=u_parent, reverse_graph=reverse_graph)
        while u_adj_list:
            w = u_adj_list.pop()
            if w in set_of_min_path_vertices:
                continue
            w_path = vertex_to_path_map[w]
            relaxed = w_path.relax(u_path, weight_function, reverse_graph)
            if relaxed:
                fib_heap.update_item_with_decreased_priority(w_path)

    return vertex_to_path_map


def _get_parent_vertex_of_path_destination(shortest_path: ShortestPath) -> Optional[Vertex]:
    if shortest_path.is_destination_unreachable() or len(shortest_path._path) < 2:
        return None
    return shortest_path.path[-2]
