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

"""Algorithms for finding minimum/maximum spanning trees/forests graphs."""

from typing import Iterator

from vertizee.classes.collections.fibonacci_heap import FibonacciHeap
from vertizee.classes.collections.priority_queue import PriorityQueue
from vertizee.classes.collections.union_find import UnionFind
from vertizee.classes.edge import EdgeType
from vertizee.classes.graph_base import GraphBase
from vertizee.classes.vertex import Vertex

INFINITY = float("inf")


def _weight_function(edge: EdgeType, weight: str = 'Edge__weight', minimum: bool = True) -> float:
    """Returns the weight of a given edge.

    If there is no edge weight, then the edge weight is assumed to be one.  If `graph` is a
    multigraph, the minimum (or maximum) edge weight over all parallel edges is returned.

    Args:
        edge (EdgeType): The edge whose weight is returned.
        weight (str, optional): The key to use to retrieve the weight from the `Edge.attr`
            dictionary. The default value ('Edge__weight') uses the `Edge.weight` property.
        minimum (bool, optional): True to return the minimum edge weight or False to return the
            maximum edge weight.

    Returns:
        float: The edge weight.
    """
    if weight == 'Edge__weight':
        edge_weight = edge.weight
    else:
        edge_weight = edge.attr.get(weight, 1)

    if len(edge.parallel_edge_weights) > 0:
        if minimum:
            min_parallel = min(edge.parallel_edge_weights)
            edge_weight = min(edge_weight, min_parallel)
        else:
            max_parallel = max(edge.parallel_edge_weights)
            edge_weight = max(edge_weight, max_parallel)
    if minimum:
        return edge_weight
    else:
        return -1 * edge_weight


def spanning_tree_kruskal(
        graph: GraphBase, weight: str = 'Edge__weight', minimum: bool = True) -> Iterator[EdgeType]:
    """Iterates over a minimum (or maximum) spanning tree of a weighted graph using Kruskal's
    algorithm.

    This algorithm is only defined for undirected graphs. To find the spanning tree of a directed
    graph, see `~spanning.spanning_arborescence_ggst`.

    The Edge class has a built-in `weight` property, which is used by default to determine edge
    weights (a.k.a. edge lengths). Alternatively, a key name may be provided to lookup the weight
    in the `Edge.attr` dictionary. If `graph` is a multigraph, the minimum (or maximum) edge
    weight over all parallel edges is returned.

    Args:
        graph (GraphBase): The graph to search.
        weight (str, optional): The key to use to retrieve the weight from the `Edge.attr`
            dictionary. The default value ('Edge__weight') uses the `Edge.weight` property.
        minimum (bool, optional): True to return the minimum spanning tree, or False to return
            the maximum spanning tree. Defaults to True.

    Returns:
        Iterator[EdgeType]: An iterator over the edges of the minimum (or maximum) spanning tree
            discovered using Kruskal's algorithm.

    See Also:
        `~edge.Edge`
        `~edge.DiEdge`
        `~spanning.spanning_arborescence_ggst`
        `~union_find.UnionFind`

    References:
        [1] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
            Introduction to Algorithms: Third Edition, page 631. The MIT Press, 2009.
    """
    if graph.is_directed_graph():
        raise ValueError('graph must be undirected; see spanning_arborescence_ggst')

    edge_weight_pairs = [(e, _weight_function(e, weight, minimum)) for e in graph.edges]
    sorted_edges = \
        [p[0] for p in sorted(edge_weight_pairs, key=lambda pair: pair[1])]
    union_find = UnionFind(*graph.vertices)

    for edge in sorted_edges:
        if not union_find.in_same_set(edge.vertex1, edge.vertex2):
            union_find.union(edge.vertex1, edge.vertex2)
            yield edge


def spanning_tree_prim(
        graph: GraphBase, root: Vertex = None, weight: str = 'Edge__weight',
        minimum: bool = True) -> Iterator[EdgeType]:
    """Iterates over a minimum (or maximum) spanning tree of a weighted graph using Prim's
    algorithm.

    This algorithm is only defined for undirected graphs. To find the spanning tree of a directed
    graph, see `~spanning.spanning_arborescence_ggst`.

    The Edge class has a built-in `weight` property, which is used by default to determine edge
    weights (a.k.a. edge lengths). Alternatively, a key name may be provided to lookup the weight
    in the `Edge.attr` dictionary. If `graph` is a multigraph, the minimum (or maximum) edge
    weight over all parallel edges is returned.

    Args:
        graph (GraphBase): The graph to search.
        root (Vertex, optional): The root vertex of the minimum spanning tree to be grown. If not
            specified, an arbitrary root vertex is chosen. Defaults to None.
        weight (str, optional): The key to use to retrieve the weight from the `Edge.attr`
            dictionary. The default value ('Edge__weight') uses the `Edge.weight` property.
        minimum (bool, optional): True to return the minimum spanning tree, or False to return
            the maximum spanning tree. Defaults to True.

    Returns:
        Iterator[EdgeType]: An iterator over the edges of the minimum (or maximum) spanning tree
            discovered using Prim's algorithm.

    See Also:
        `~edge.Edge`
        `~edge.DiEdge`
        `~spanning.spanning_arborescence_ggst`
        `~union_find.UnionFind`

    References:
        [1] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
            Introduction to Algorithms: Third Edition, page 634. The MIT Press, 2009.
    """
    PRIM_PARENT_KEY = '__prim_parent'
    PRIM_PRIORITY_KEY = '__prim_priority'

    if graph.is_directed_graph():
        raise ValueError('graph must be undirected; see spanning_arborescence_ggst')
    if root is not None:
        r: Vertex = graph[root]
        if r is None:
            raise ValueError('root vertex not found in the graph')
    else:
        if len(graph.vertices) > 0:
            r = next(iter(graph.vertices))
        else:
            return iter([])

    def prim_priority_function(v: Vertex):
        return v.attr[PRIM_PRIORITY_KEY]

    priority_queue: PriorityQueue[Vertex] = PriorityQueue(prim_priority_function)
    for v in graph:
        v.attr[PRIM_PARENT_KEY] = None
        v.attr[PRIM_PRIORITY_KEY] = INFINITY
        priority_queue.add_or_update(v)
    r.attr[PRIM_PRIORITY_KEY] = 0
    priority_queue.add_or_update(r)

    vertices_in_tree = set()
    tree_edge: EdgeType = None

    while len(priority_queue) > 0:
        u = priority_queue.pop()
        vertices_in_tree.add(u)
        if u.attr[PRIM_PARENT_KEY] is not None:
            parent = u.attr[PRIM_PARENT_KEY]
            adj_vertices = u.adjacent_vertices - {parent}
            tree_edge = graph[parent][u]
        else:
            adj_vertices = u.adjacent_vertices

        for v in adj_vertices:
            u_v_weight = _weight_function(graph[u][v], weight, minimum)
            if v not in vertices_in_tree and u_v_weight < v.attr[PRIM_PRIORITY_KEY]:
                v.attr[PRIM_PARENT_KEY] = u
                v.attr[PRIM_PRIORITY_KEY] = u_v_weight
                priority_queue.add_or_update(v)
        if tree_edge:
            yield tree_edge


def spanning_tree_prim_fibonacci(
        graph: GraphBase, root: Vertex = None, weight: str = 'Edge__weight',
        minimum: bool = True) -> Iterator[EdgeType]:
    """Iterates over a minimum (or maximum) spanning tree of a weighted graph using Prim's
    algorithm implemented using a Fibonacci heap.

    This algorithm is only defined for undirected graphs. To find the spanning tree of a directed
    graph, see `~spanning.spanning_arborescence_ggst`.

    The Edge class has a built-in `weight` property, which is used by default to determine edge
    weights (a.k.a. edge lengths). Alternatively, a key name may be provided to lookup the weight
    in the `Edge.attr` dictionary. If `graph` is a multigraph, the minimum (or maximum) edge
    weight over all parallel edges is returned.

    Args:
        graph (GraphBase): The graph to search.
        root (Vertex, optional): The root vertex of the minimum spanning tree to be grown. If not
            specified, an arbitrary root vertex is chosen. Defaults to None.
        weight (str, optional): The key to use to retrieve the weight from the `Edge.attr`
            dictionary. The default value ('Edge__weight') uses the `Edge.weight` property.
        minimum (bool, optional): True to return the minimum spanning tree, or False to return
            the maximum spanning tree. Defaults to True.

    Returns:
        Iterator[EdgeType]: An iterator over the edges of the minimum (or maximum) spanning tree
            discovered using Prim's algorithm.

    See Also:
        `~edge.Edge`
        `~edge.DiEdge`
        `~spanning.spanning_arborescence_ggst`
        `~union_find.UnionFind`

    References:
        [1] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
            Introduction to Algorithms: Third Edition, page 634. The MIT Press, 2009.
    """
    PRIM_PARENT_KEY = '__prim_parent'
    PRIM_PRIORITY_KEY = '__prim_priority'

    if graph.is_directed_graph():
        raise ValueError('graph must be undirected; see spanning_arborescence_ggst')
    if root is not None:
        r: Vertex = graph[root]
        if r is None:
            raise ValueError('root vertex not found in the graph')
    else:
        if len(graph.vertices) > 0:
            r = next(iter(graph.vertices))
        else:
            return iter([])

    def prim_priority_function(v: Vertex):
        return v.attr[PRIM_PRIORITY_KEY]

    fib_heap: FibonacciHeap[Vertex] = FibonacciHeap(prim_priority_function)
    for v in graph:
        v.attr[PRIM_PARENT_KEY] = None
        v.attr[PRIM_PRIORITY_KEY] = INFINITY
        fib_heap.insert(v)
    r.attr[PRIM_PRIORITY_KEY] = 0
    fib_heap.update_item_with_decreased_priority(r)

    vertices_in_tree = set()
    tree_edge: EdgeType = None

    while len(fib_heap) > 0:
        u = fib_heap.extract_min()
        vertices_in_tree.add(u)
        if u.attr[PRIM_PARENT_KEY] is not None:
            parent = u.attr[PRIM_PARENT_KEY]
            adj_vertices = u.adjacent_vertices - {parent}
            tree_edge = graph[parent][u]
        else:
            adj_vertices = u.adjacent_vertices

        for v in adj_vertices:
            u_v_weight = _weight_function(graph[u][v], weight, minimum)
            if v not in vertices_in_tree and u_v_weight < v.attr[PRIM_PRIORITY_KEY]:
                v.attr[PRIM_PARENT_KEY] = u
                v.attr[PRIM_PRIORITY_KEY] = u_v_weight
                fib_heap.update_item_with_decreased_priority(v)
        if tree_edge:
            yield tree_edge
