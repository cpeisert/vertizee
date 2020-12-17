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

"""Algorithms for finding optimum :term:`arborescences <arborescence>` and
:term:`branchings <branching>` of :term:`digraphs <digraph>`.

Functions:

* :func:`optimum_directed_forest` - Iterates over the minimum (or maximum) :term:`arborescences
  <arborescence>` comprising a directed :term:`spanning forest` of a weighted,
  :term:`directed graph <digraph>`.
* :func:`edmonds` - Iterates over the maximum (or minimum) :term:`arborescences <arborescence>` of
  a :term:`digraph` comprising an :term:`optimum spanning branching` using Edmonds' algorithm.
"""

from __future__ import annotations
from typing import Callable, Dict, Final, Iterator, Union

from vertizee import exception

from vertizee.algorithms.algo_utils.spanning_utils import (
    Cycle, PseudoEdge, PseudoGraph, PseudoVertex
)
from vertizee.classes.data_structures.union_find import UnionFind
from vertizee.classes.data_structures.tree import Tree
from vertizee.classes.graph import DiGraph, MultiDiGraph
from vertizee.classes.edge import E
from vertizee.classes.vertex import V


INFINITY: Final = float("inf")


class ReverseSearchState:
    """A class to save the state of which adjacent vertices (``parents``) of a vertex (``child``)
    still have not been visited in a reverse depth-first search.

    Args:
        child: The child vertex relative to ``parents`` in a reverse depth-first search tree.
        parents: An iterator over the unvisited parents of ``child`` in a search tree.
    """
    def __init__(self, child: V, parents: Iterator[V]) -> None:
        self.child = child
        self.parents = parents


def get_weight_function(weight: str = "Edge__weight", minimum: bool = True) -> Callable[[E], float]:
    """Returns a function that accepts an edge and returns the corresponding edge weight.

    If there is no edge weight, then the edge weight is assumed to be one.

    Note:
        For multigraphs, the minimum (or maximum) edge weight among the parallel edge connections
        is returned.

    Args:
        weight: Optional; The key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge_weight``) uses the ``Edge.weight`` property.
        minimum: Optional; For multigraphs, if True, then the minimum weight from the parallel edge
            connections is returned, otherwise the maximum weight. Defaults to True.

    Returns:
        Callable[[E], float]: A function that accepts an edge and returns the
        corresponding edge weight.
    """

    def default_weight_function(edge: E) -> float:
        if edge._parent_graph.is_multigraph():
            if minimum:
                return min(c.weight for c in edge.connections())
            return max(c.weight for c in edge.connections())
        return edge.weight

    def attr_weight_function(edge: E) -> float:
        if edge._parent_graph.is_multigraph():
            if minimum:
                return min(c.attr.get(weight, 1.0) for c in edge.connections())
            return max(c.attr.get(weight, 1.0) for c in edge.connections())
        return edge.attr.get(weight, 1.0)

    if weight == "Edge__weight":
        return default_weight_function
    return attr_weight_function


def get_total_weight_function(weight: str = "Edge__weight") -> Callable[[E], float]:
    """Returns a function that accepts an edge and returns the total weight of the edge, which
    in the case of multiedges, includes the weights of all parallel edge connections.

    If there is no edge weight, then the edge weight is assumed to be one.

    Args:
        weight: Optional; The key to use to retrieve the weight from the ``Edge.attr``
            dictionary. The default value (``Edge_weight``) uses the ``Edge.weight`` property.

    Returns:
        Callable[[E], float]: A function that accepts an edge and returns the total weight of the
        edge, including parallel connections.
    """

    def default_total_weight_function(edge: E) -> float:
        if edge._parent_graph.is_multigraph():
            return sum(c.weight for c in edge.connections())
        return edge.weight

    def attr_total_weight_function(edge: E) -> float:
        if edge._parent_graph.is_multigraph():
            return sum(c.attr.get(weight, 1.0) for c in edge.connections())
        return edge.attr.get(weight, 1.0)

    if weight == "Edge__weight":
        return default_total_weight_function
    return attr_total_weight_function


def edmonds(
    digraph: Union[DiGraph, MultiDiGraph], minimum: bool = True,
    find_spanning_arborescence: bool = False, weight: str = "Edge__weight"
) -> Iterator[Tree[V, E]]:
    """Iterates over the maximum (or minimum) :term:`arborescences <arborescence>` of a
    :term:`digraph` comprising an :term:`optimum spanning branching` using Edmonds' algorithm.
    :cite:`1967:edmonds`

    Running time: :math:`O(mn)` where :math:`m = |E|` and :math:`n = |V|`

    Note:
        In multigraphs, finding optimum branchings does not require knowing all of the parallel
        edge weights, but only the maximum or minimum weight connection of each multiedge
        (depending on the value of ``minimum``). Hence, for the purposes of analysis, this
        algorithm creates a new digraph (or pseudodigraph) by replace parallel edges with either
        the max or min weight connection. The ``digraph`` object is not modified.

    Args:
        graph: The directed graph to iterate.
        minimum: Optional;  True to return the minimum arborescences, or False to return
            the maximum arborescences. Defaults to True.
        find_spanning_arborescence: If True, then attempts to find an
            :term:`optimum spanning arborescence` as opposed to an
            :term:`optimum spanning branching`.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Yields:
        Iterator[Tree[V, E]]: An iterator over the minimum (or maximum) arborescences. If only one
        arborescence is yielded prior to ``StopIteration``, then it is a
        :term:`spanning arborescence`.

    See Also:
        * :func:`spanning_tree`
        * :func:`kruskal`
        * :func:`prim`
        * :func:`prim_fibonacci`
    """
    if len(digraph) == 0:
        raise exception.Unfeasible("directed forests are undefined for empty graphs")
    if not digraph.is_directed():
        raise exception.GraphTypeNotSupported(
            "graph must be directed; for undirected graphs see spanning_tree and optimum_forest")

    weight_function = get_weight_function(weight, minimum=minimum)
    sign = -1 if minimum else 1

    arborescence_vertex_sets: UnionFind[PseudoVertex] = UnionFind()
    """A collection of disjoint sets of pseudovertices, where each set comprises an arborescence."""

    representative_vertex_to_arborescence: Dict[PseudoVertex, Tree] = dict()
    """A mapping from the representative vertex of each set of vertices in
    ``arborescence_vertex_sets`` to a tree object representation of the arborescence."""

    pseudograph = PseudoGraph()
    for v in digraph.vertices():
        pv = PseudoVertex(
            label=v.label,
            parent_graph=pseudograph,
            incident_edge_labels=v._incident_edges._incident_edge_labels)
        pseudograph.add_vertex_object(pv)
        arborescence_vertex_sets.make_set(pv)
        representative_vertex_to_arborescence[pv] = Tree(pv)

    for e in digraph.edges():
        v1 = pseudograph[e.vertex1]
        v2 = pseudograph[e.vertex2]
        pe = PseudoEdge(vertex1=v1, vertex2=v2, weight=(sign * weight_function(e)))
        pseudograph.add_edge_object(pe)

    #
    # The following are steps taken from the paper "Optimum Branchings", section 4.
    # :cite:`1967:edmonds`
    #

    # Step (1): Choose a node v in G_i and not in D_i. Put v into bucket D_i. If there is in G_i
    # a _positively weighted_ edge directed toward v, put one of them having max weight into bucket
    # E_i. Repeat this step until (a) E_i no longer comprises a branching in G_i, or until (b) every
    # node of G_i is in D_i, and E_i comprises the edges of a branching. When case (a) occurs, apply
    # step (2).
    vertices = set(pseudograph.vertices())
    while vertices:
        vertex = vertices.pop()
        if not vertex.incident_edges_incoming():
            continue

        optimum_edge = None
        for edge in vertex.incident_edges_incoming():
            if edge.is_loop():
                continue
            if optimum_edge is None:
                optimum_edge = edge
            elif edge.weight > optimum_edge.weight:
                optimum_edge = edge

        if not optimum_edge:
            continue

        if not find_spanning_arborescence:
            # Note: a minimum spanning branching does not contain any positive edges and a
            # maximum spanning branching does not contain any negative edges. This is because a
            # trivial spanning branching may always be formed from the digraph vertex set with no
            # edges, since each vertex comprises a trivial arborescence.
            if optimum_edge.weight < 0:
                continue

        parent_candidate: PseudoVertex = optimum_edge.tail
        if arborescence_vertex_sets.in_same_set(vertex, parent_candidate):
            # Step (2): Store cycle Q_i and the edge of the cycle with minimum weight. Obtain
            # a new graph by "shrinking" to a single new vertex v_1^{i + 1} the cycle Q_i and
            # every edge of Q_i.
            new_vertex = _contract_cycle(
                pseudograph, cycle_edge=optimum_edge, union_find=arborescence_vertex_sets,
                vertex_to_arborescence=representative_vertex_to_arborescence)
            vertices.discard(new_vertex.cycle.vertices)
            vertices.add(new_vertex)
            continue

        if vertex.parent_vertex:
            raise exception.AlgorithmError(f"vertex {vertex} already has a parent vertex "
                f"{vertex.parent_vertex} (i.e., a vertex connected by an incoming edge) in "
                "addition to a second incoming edge, which violates the definition of arborescence")
        vertex.parent_vertex = parent_candidate

        # parent_representative = arborescence_vertex_sets[
        #     vertex.parent_vertex.get_an_original_vertex()]
        # child_representative = arborescence_vertex_sets[vertex.get_an_original_vertex()]
        parent_representative = arborescence_vertex_sets[vertex.parent_vertex]
        child_representative = arborescence_vertex_sets[vertex]

        arborescence_vertex_sets.union(child_representative, parent_representative)
        new_representative = arborescence_vertex_sets[parent_representative]

        parent_arborescence = representative_vertex_to_arborescence[parent_representative]
        child_arborescence = representative_vertex_to_arborescence[child_representative]
        parent_arborescence.merge(child_arborescence)
        parent_arborescence._edges[optimum_edge.label] = optimum_edge

        representative_vertex_to_arborescence.pop(parent_representative)
        representative_vertex_to_arborescence.pop(child_representative)
        representative_vertex_to_arborescence[new_representative] = parent_arborescence

    # Step (3): The most recently discovered cycle (that is, the cycle on top of the pseudograph
    # ``cycle_stack``) corresponds to a vertex with the same label as the cycle. This vertex was
    # created by contracting the cycle in the original graph. In the Edmonds paper, this new vertex
    # is designated v_1^{i + 1}. We shall refer to this vertex as 'vertex_cycle_i+1' (i.e. the
    # vertex created from a cycle on the ``cycle_stack`` numbered (i + 1)). Hence, the preceding
    # cycle on the stack corresponds to vertex_cycle_i.
    #
    # In the case where vertex_cycle_i+1 is not a root of an arborescence, there is a unique edge
    # in an arborescence containing vertex_cycle_i+1, say edge_i+1, that is directed toward
    # vertex_cycle_i+1. In the version of the graph prior to contracting cycle i+1, the previous
    # version of edge_i+1, designated as edge_i, has a head vertex, head_i that is part of
    # cycle_i+1. Within cycle_i+1, there is a cycle edge, edge_i_2, directed toward head_i. The
    # edges, edge_i and edge_i_2, are the only two edges directed toward head_i. Deleting edge
    # edge_i_2 yields a branching in the version of the graph prior to contracting cycle_i+1.
    #
    # In the case where vertex_cycle_i+1 is the root of an arborescence, there are no two edges
    # in the branching directed toward the same node. Therefore, deleting any edge of cycle_i+1
    # yields an arborescence in the version of the graph prior to contracting cycle_i+1. To obtain
    # an optimum branching, delete from cycle_i+1 the cycle edge with minimum weight.
    while pseudograph.cycle_stack:
        cycle: Cycle = pseudograph.cycle_stack.pop()
        vertex_cycle: PseudoVertex = pseudograph[cycle.label]
        edge_to_delete = None

        if (vertex_cycle.parent_vertex and
            pseudograph.has_edge(vertex_cycle.parent_vertex, vertex_cycle)):
            # vertex_cycle_i+1 is _not_ the root of an arborescence
            incoming_edge: PseudoEdge = pseudograph[vertex_cycle.parent_vertex, vertex_cycle]
            if not incoming_edge.previous_version:
                raise exception.AlgorithmError(f"edge {incoming_edge} directed toward cycle "
                    f"{cycle.label} does not have a previous version; this should not be possible")
            head_i = incoming_edge.previous_version.head

            edge_i_2 = None
            for cycle_edge in cycle.edges:
                if cycle_edge.head == head_i:
                    edge_i_2 = cycle_edge
                    break
            if not edge_i_2:
                raise exception.AlgorithmError(
                    f"no incoming cycle edge found for cycle vertex {head_i}")
            edge_to_delete = edge_i_2

        if not edge_to_delete:
            # vertex_cycle_i+1 is the root of an arborescence
            edge_to_delete = cycle.min_weight_edge

        representative_vertex = arborescence_vertex_sets[vertex_cycle]
        arborescence = representative_vertex_to_arborescence[representative_vertex]
        _undo_cycle_contraction(pseudograph, cycle, edge_to_delete=edge_to_delete,
            arborescence=arborescence)

    for pseudo_arborescence in representative_vertex_to_arborescence.values():
        arborescence = Tree(digraph._vertices[pseudo_arborescence.root.label])
        for vertex in pseudo_arborescence.vertices():
            if vertex.contains_cycle():
                continue
            arborescence._vertex_set.add(digraph._vertices[vertex.label])
        for edge in pseudo_arborescence.edges():
            arborescence._edges[edge.label] = digraph._edges[edge.label]
        yield arborescence


def optimum_directed_forest(
    digraph: Union[DiGraph, MultiDiGraph], minimum: bool = True, weight: str = "Edge__weight"
) -> Iterator[Tree[V, E]]:
    """Iterates over the minimum (or maximum) :term:`arborescences <arborescence>` comprising a
    directed :term:`optimum spanning forest` (also called an :term:`optimum branching <branching>`)
    of a weighted, :term:`directed graph`.

    Running time: :math:`O(m + n(\\log{n}))` where :math:`m = |E|` and :math:`n = |V|`

    Args:
        graph: The directed graph to iterate.
        minimum: Optional;  True to return the minimum arborescences, or False to return
            the maximum arborescences. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the ``E.attr`` dictionary. The
            default value (``Edge__weight``) uses the property ``E.weight``.

    Yields:
        Iterator[Tree[V, E]]: An iterator over the minimum (or maximum) arborescences. If only one
        arborescence is yielded prior to ``StopIteration``, then it is a
        :term:`spanning arborescence`.

    See Also:
        * :func:`spanning_tree`
        * :func:`kruskal`
        * :func:`prim`
        * :func:`prim_fibonacci`
        * :class:`UnionFind <vertizee.classes.data_structures.union_find.UnionFind>`

    Note:
        This implementation is based on the treatment by Gabow, Galil, Spencer, and Tarjan in their
        paper :download:`"Efficient algorithms for finding minimum spanning trees in undirected and
        directed graphs."
        </references/Efficient_algorithms_for_finding_min_spanning_trees_GGST.pdf>` [GGST1986]_ The
        work of Gabow et al. builds upon the Chu–Liu/Edmonds' algorithm presented in the paper
        :download:`"Optimum Branchings." </references/Optimum_Branchings_Edmonds.pdf>`. [E1986]_

    References:
     .. [E1986] Jack Edmonds. :download:`"Optimum Branchings."
            </references/Optimum_Branchings_Edmonds.pdf>` Journal of Research of the National
            Bureau of Standards Section B, 71B (4):233–240, 1967.

     .. [GGST1986] Harold N. Gabow, Zvi Galil, Thomas Spencer, and Robert E. Tarjan.
            :download:`"Efficient algorithms for finding minimum spanning trees in undirected and
            directed graphs."
            </references/Efficient_algorithms_for_finding_min_spanning_trees_GGST.pdf>`
            Combinatorica 6:109-122. Springer, 1986.
    """
    if len(digraph) == 0:
        raise exception.Unfeasible("directed forests are undefined for empty graphs")
    if not digraph.is_directed():
        raise exception.GraphTypeNotSupported(
            "graph must be directed; for undirected graphs see spanning_tree")

    # weight_function = get_weight_function(weight, minimum=minimum)
    # total_weight_function = get_total_weight_function(weight)
    # sign = 1 if minimum else -1


def _contract_cycle(
    graph: PseudoGraph, cycle_edge: PseudoEdge, union_find: UnionFind,
    vertex_to_arborescence: Dict[PseudoVertex, Tree]
) -> PseudoVertex:
    """Helper method to contract a cycle in a graph to a new vertex. The ``cycle_edge`` is the last
    edge that was discovered forming the cycle. The cycle may be followed by using the
    ``parent_vertex`` properties of the cycle vertices starting with ``cycle_edge.tail`` until
    cycling around to ``cycle_edge.head``. (The parent vertex refers to the parent within an
    :term:`arborescence`.)

    Each cycle edge is stored within the new vertex's ``cycle.edges`` attribute and each cycle
    vertex is stored in the new vertex's ``cycle.vertices`` attribute. These cycle edges and
    vertices are removed from the graph.

    Edges in the graph coming into the cycle (i.e. edges whose tail vertices are outside the cycle
    and whose head vertices are on the cycle) as well as edges leaving the cycle (i.e. edges whose
    tail vertices are on the cycle and whose head vertices are outside the cycle) are removed from
    the graph and replaced by new edges. Each new edge is formed by using the previous edge's
    vertex that was not on the cycle and replacing the vertex on the cycle with the new vertex
    formed by contracting the cycle. The original edges are stored within the new pseudoedge
    objects in the property ``previous_version``, that is, each new edge stores a reference to the
    version of itself prior to substituting either its tail or head with the new vertex.

    In addition, edges coming into the cycle are reweighted as follows:

        new_weight = old_weight + weight_of_min_weight_cycle_edge - weight_of_incoming_cycle_edge

    weight_of_incoming_cycle_edge - The weight of the cycle edge that is directed toward the
    head of the incoming edge.

    Args:
        graph: The graph containing the cycle.
        cycle_edge: The final edge discovered that forms a cycle in the graph.
        union_find: A collection of disjoint sets of vertices, where each set is an arborescence.
        vertex_to_arborescence: A mapping from representative vertices from ``union_find`` to
            :class:`Tree` objects, where each tree is the arborescence containing the vertex.

    Returns:
        PseudoVertex: A new pseudovertex comprising the contracted vertices and edges of the cycle.
    """
    cycle_start = cycle_edge.tail
    cycle_end = cycle_edge.head
    representative_vertex = union_find[cycle_start]
    arborescence: Tree = vertex_to_arborescence[representative_vertex]

    label = graph.create_cycle_label()
    cycle = Cycle(label)
    cycle.add_cycle_edge(cycle_edge)
    graph.cycle_stack.append(cycle)

    new_vertex = PseudoVertex(label, parent_graph=graph)
    new_vertex.cycle = cycle
    arborescence._vertex_set.add(new_vertex)
    graph.add_vertex_object(new_vertex)

    vertex: PseudoVertex = cycle_start
    cycle.min_weight_edge = cycle_edge
    while vertex != cycle_end:
        next_cycle_edge = graph[vertex.parent_vertex, vertex]
        if next_cycle_edge.weight < cycle.min_weight_edge.weight:
            cycle.min_weight_edge = next_cycle_edge
        cycle.add_cycle_edge(next_cycle_edge)
        vertex = vertex.parent_vertex

    for vertex in cycle.vertices:
        for edge in vertex.incident_edges_incoming():
            if edge in cycle.edges or edge not in graph:
                continue

            if edge in arborescence and edge.tail not in cycle.vertices:
                new_vertex.parent_vertex = edge.tail

            cycle_vertex = edge.head
            incoming_cycle_edge = None
            for e in cycle_vertex.incident_edges_incoming():
                if e in cycle.edges:
                    incoming_cycle_edge = e
                    break
            if not incoming_cycle_edge:
                raise exception.AlgorithmError(
                    f"no incoming cycle edge found for cycle vertex {cycle_vertex}")

            new_weight = edge.weight + cycle.min_weight_edge.weight - incoming_cycle_edge.weight
            if edge.tail in cycle.vertices:
                new_edge = PseudoEdge(new_vertex, new_vertex, weight=new_weight)
            else:
                new_edge = PseudoEdge(edge.tail, new_vertex, weight=new_weight)
            new_edge.previous_version = edge
            cycle.incoming_edges.add(new_edge)
            new_vertex._add_edge(new_edge)

            if edge in arborescence:
                arborescence._edges.pop(edge.label, None)
                arborescence._edges[new_edge.label] = new_edge
            graph.remove_edge(edge.vertex1, edge.vertex2)
            graph.add_edge_object(new_edge)

        for edge in vertex.incident_edges_outgoing():
            if edge in cycle.edges or edge not in graph:
                continue

            head: PseudoVertex = edge.head
            cycle.outgoing_edge_head_previous_parent[edge] = head.parent_vertex

            if head.parent_vertex and head.parent_vertex == edge.tail:
                head.parent_vertex = new_vertex

            if edge.head in cycle.vertices:
                new_edge = PseudoEdge(new_vertex, new_vertex, weight=0)
            else:
                new_edge = PseudoEdge(new_vertex, edge.head, weight=edge.weight)
            new_edge.previous_version = edge
            cycle.outgoing_edges.add(new_edge)
            new_vertex._add_edge(new_edge)

            if edge in arborescence:
                arborescence._edges.pop(edge.label, None)
                arborescence._edges[new_edge.label] = new_edge
            graph.remove_edge(edge.vertex1, edge.vertex2)
            graph.add_edge_object(new_edge)

        vertex = vertex.parent_vertex

    for edge in cycle.edges:
        arborescence._edges.pop(edge.label, None)
        graph.remove_edge(edge.vertex1, edge.vertex2)

    union_find.make_set(new_vertex)
    union_find.union(new_vertex, representative_vertex)
    new_representative = union_find[representative_vertex]
    if new_representative != representative_vertex:
        vertex_to_arborescence.pop(representative_vertex)
        vertex_to_arborescence[new_representative] = arborescence

    if arborescence.root in cycle.vertices:
        arborescence._root = new_vertex
    while arborescence.root.parent_vertex:
        arborescence._root = arborescence.root.parent_vertex

    return new_vertex


def _undo_cycle_contraction(graph: PseudoGraph, cycle: Cycle, edge_to_delete: PseudoEdge,
    arborescence: Tree) -> None:
    """
    This is a helper function that undoes a cycle contraction by add the cycle edges back to the
    graph and removing one of the cycle edges, thus breaking the cycle. In addition, each incoming
    and outgoing cycle edge is rolled back to its previous version prior to the cycle contraction.
    """
    for edge in cycle.edges:
        arborescence._edges[edge.label] = edge
        graph.add_edge_object(edge)

    arborescence._edges.pop(edge_to_delete.label)
    graph.remove_edge(edge_to_delete.vertex1, edge_to_delete.vertex2)

    cycle_vertex = graph[cycle.label]
    if arborescence.root == cycle_vertex:
        arborescence._root = next(iter(cycle.vertices))
    while arborescence.root.parent_vertex:
        arborescence._root = arborescence.root.parent_vertex

    for edge in cycle.incoming_edges:
        if edge in arborescence:
            arborescence._edges.pop(edge.label, None)
            arborescence._edges[edge.previous_version.label] = edge.previous_version
        if edge in graph:
            graph.remove_edge(edge.vertex1, edge.vertex2)
            graph.add_edge_object(edge.previous_version)
    for edge in cycle.outgoing_edges:
        if edge in arborescence:
            arborescence._edges.pop(edge.label, None)
            arborescence._edges[edge.previous_version.label] = edge.previous_version
        if edge in graph:
            graph.remove_edge(edge.vertex1, edge.vertex2)
            graph.add_edge_object(edge.previous_version)
        edge.head.parent_vertex = cycle.outgoing_edge_head_previous_parent[edge.previous_version]

    arborescence._vertex_set.remove(cycle_vertex)
    graph.remove_vertex(cycle_vertex)
