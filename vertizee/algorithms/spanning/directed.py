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

# pylint: disable=line-too-long
"""
========================================================
Spanning: directed graphs (arborescences and branchings)
========================================================

Algorithms for finding optimum :term:`arborescences <arborescence>` and
:term:`directed forests <directed forest>` (also called :term:`branchings <branching>`). The
asymptotic running times use the notation that for some digraph :math:`G(V, E)`, the number of
vertices is :math:`n = |V|` and the number of edges is :math:`m = |E|`.

**Recommended Tutorial**: :doc:`Spanning trees, arborescences, forests, and branchings <../../tutorials/spanning_tree_arborescence>` - |image-colab-spanning|

.. |image-colab-spanning| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/cpeisert/vertizee/blob/master/docs/source/tutorials/spanning_tree_arborescence.ipynb

Function summary
================

* :func:`optimum_directed_forest` - Iterates over a minimum (or maximum) :term:`directed forest` of
  a weighted, :term:`digraph` using Edmonds' algorithm. :cite:`1967:edmonds` A directed
  forest is also called a :term:`branching`. Running time: :math:`O(mn)`
* :func:`spanning_arborescence` - Returns a minimum (or maximum) :term:`spanning arborescence` of
  a weighted, :term:`digraph` using Edmonds' algorithm. :cite:`1967:edmonds`
  Running time: :math:`O(mn)`
* :func:`edmonds` - Iterates over the :term:`optimum spanning arborescence` or the
  :term:`optimum spanning branching` of a :term:`digraph` using Edmonds' algorithm.
  :cite:`1967:edmonds` Running time: :math:`O(mn)`

Detailed documentation
======================
"""

from __future__ import annotations
from typing import cast, Dict, Final, Iterator, Set, TYPE_CHECKING, Union

from vertizee import exception

from vertizee.algorithms.algo_utils.spanning_utils import (
    Cycle,
    get_weight_function,
    PseudoEdge,
    PseudoGraph,
    PseudoVertex,
)
from vertizee.classes.data_structures.union_find import UnionFind
from vertizee.classes.data_structures.tree import Tree
from vertizee.classes.edge import MutableEdgeBase
from vertizee.classes.vertex import DiVertex, MultiDiVertex, VertexType

if TYPE_CHECKING:
    from vertizee.classes.graph import DiGraph, MultiDiGraph


INFINITY: Final[float] = float("inf")


def edmonds(
    digraph: Union[DiGraph, MultiDiGraph],
    minimum: bool = True,
    find_spanning_arborescence: bool = False,
    weight: str = "Edge__weight",
) -> Iterator[Tree[Union[DiVertex, MultiDiVertex]]]:
    """Iterates over the :term:`optimum spanning arborescence` or the
    :term:`optimum spanning branching` of a :term:`digraph` using Edmonds' algorithm.
    :cite:`1967:edmonds`

    Running time: :math:`O(mn)` where :math:`m = |E|` and :math:`n = |V|`

    Note:
        This algorithm is only defined for *directed* graphs. To find the optimum forest of an
        undirected graph, see :func:`optimum_forest
        <vertizee.algorithms.spanning.undirected.optimum_forest>`. To find the spanning tree of an
        undirected graph, see :func:`spanning_tree
        <vertizee.algorithms.spanning.undirected.spanning_tree>`.

    Note:
        In multigraphs, finding optimum branchings does not require knowing all of the parallel
        edge weights, but only the maximum or minimum weight connection of each multiedge
        (depending on the value of ``minimum``). Hence, this algorithm creates a new digraph (or
        pseudodigraph) by replacing parallel edges with either the max or min weight connection.
        The original ``digraph`` object is not modified.

    Args:
        digraph: The directed graph to explore.
        minimum: Optional;  True to return the minimum arborescences, or False to return
            the maximum arborescences. Defaults to True.
        find_spanning_arborescence: If True, then attempts to find an
            :term:`optimum spanning arborescence` as opposed to an
            :term:`optimum spanning branching`.
        weight: Optional; The key to use to retrieve the weight from the edge ``attr``
            dictionary. The default value ("Edge__weight") uses the edge property ``weight``.

    Yields:
        Iterator[Tree[Union[DiVertex, MultiDiVertex]]]: An iterator over the minimum (or maximum)
        arborescences.

    Raises:
        Unfeasible: If ``find_spanning_arborescence`` is set to True and the graph does not contain
            a spanning arborescence, an Unfeasible exception is raised.

    See Also:
        * :func:`optimum_directed_forest`
        * :func:`spanning_arborescence`
        * :class:`Tree <vertizee.classes.data_structures.tree.Tree>`
    """
    if len(digraph) == 0:
        raise exception.Unfeasible("directed forests are undefined for empty graphs")
    if not digraph.is_directed():
        raise exception.GraphTypeNotSupported(
            "graph must be directed; for undirected graphs see spanning_tree and optimum_forest"
        )

    weight_function = get_weight_function(weight, minimum=minimum)
    sign = -1 if minimum else 1

    arborescence_vertex_sets: UnionFind[PseudoVertex] = UnionFind()
    """A collection of disjoint sets of pseudovertices, where each set comprises an arborescence."""

    representative_vertex_to_arborescence: Dict[PseudoVertex, Tree[PseudoVertex]] = dict()
    """A mapping from the representative vertex of each set of vertices in
    ``arborescence_vertex_sets`` to a tree object representation of the arborescence."""

    pseudograph = PseudoGraph()
    for v in digraph.vertices():
        pv = PseudoVertex(
            label=v.label,
            parent_graph=pseudograph,
            incident_edge_labels=v._incident_edges._incident_edge_labels,
        )
        pseudograph.add_vertex_object(pv)
        arborescence_vertex_sets.make_set(pv)
        representative_vertex_to_arborescence[pv] = Tree(pv)

    for e in digraph.edges():
        v1 = cast(PseudoVertex, pseudograph[e.vertex1])
        v2 = cast(PseudoVertex, pseudograph[e.vertex2])
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
    vertices: Set[PseudoVertex] = set(pseudograph.vertices())
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

        assert isinstance(optimum_edge, PseudoEdge)
        parent_candidate = optimum_edge.vertex1
        if arborescence_vertex_sets.in_same_set(vertex, parent_candidate):
            # Step (2): Store cycle Q_i and the edge of the cycle with minimum weight. Obtain
            # a new graph by "shrinking" to a single new vertex v_1^{i + 1} the cycle Q_i and
            # every edge of Q_i.
            new_vertex = _contract_cycle(
                pseudograph,
                cycle_edge=optimum_edge,
                union_find=arborescence_vertex_sets,
                vertex_to_arborescence=representative_vertex_to_arborescence,
            )
            assert new_vertex.cycle is not None
            vertices.difference_update(new_vertex.cycle.vertices)
            vertices.add(new_vertex)
            continue

        if vertex.parent_vertex:
            raise exception.AlgorithmError(
                f"vertex {vertex} already has a parent vertex "
                f"{vertex.parent_vertex} (i.e., a vertex connected by an incoming edge) in "
                "addition to a second incoming edge, which violates the definition of arborescence"
            )
        vertex.parent_vertex = parent_candidate

        parent_representative = arborescence_vertex_sets[vertex.parent_vertex]
        child_representative = arborescence_vertex_sets[vertex]

        arborescence_vertex_sets.union(child_representative, parent_representative)
        new_representative = arborescence_vertex_sets[parent_representative]

        parent_arborescence = representative_vertex_to_arborescence[parent_representative]
        child_arborescence = representative_vertex_to_arborescence[child_representative]
        parent_arborescence.merge(child_arborescence)
        parent_arborescence._edges[optimum_edge.label] = cast(
            MutableEdgeBase[PseudoVertex], optimum_edge
        )

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
        vertex_cycle = cast(PseudoVertex, pseudograph[cycle.label])
        edge_to_delete = None

        if vertex_cycle.parent_vertex and pseudograph.has_edge(
            vertex_cycle.parent_vertex, vertex_cycle
        ):
            # vertex_cycle_i+1 is _not_ the root of an arborescence
            incoming_edge: PseudoEdge = cast(
                PseudoEdge, pseudograph.get_edge(vertex_cycle.parent_vertex, vertex_cycle)
            )
            if not incoming_edge.previous_version:
                raise exception.AlgorithmError(
                    f"edge {incoming_edge} directed toward cycle "
                    f"{cycle.label} does not have a previous version; this should not be possible"
                )
            head_i = incoming_edge.previous_version.vertex2

            edge_i_2 = None
            for cycle_edge in cycle.edges:
                if cycle_edge.vertex2 == head_i:
                    edge_i_2 = cycle_edge
                    break
            if not edge_i_2:
                raise exception.AlgorithmError(
                    f"no incoming cycle edge found for cycle vertex {head_i}"
                )
            edge_to_delete = edge_i_2

        if not edge_to_delete:
            # vertex_cycle_i+1 is the root of an arborescence
            edge_to_delete = cycle.min_weight_edge

        representative_vertex = arborescence_vertex_sets[vertex_cycle]
        arborescence = representative_vertex_to_arborescence[representative_vertex]
        assert isinstance(edge_to_delete, PseudoEdge)
        _undo_cycle_contraction(
            pseudograph, cycle, edge_to_delete=edge_to_delete, arborescence=arborescence
        )

    for pseudo_arborescence in representative_vertex_to_arborescence.values():
        if (
            find_spanning_arborescence
            and len(pseudo_arborescence.edges()) != digraph.vertex_count - 1
        ):
            raise exception.Unfeasible(
                "digraph does not contain a spanning arborescence; see optimum_directed_forest()"
            )

        final_arborescence: Tree[Union[DiVertex, MultiDiVertex]] = Tree(
            digraph._vertices[pseudo_arborescence.root.label]
        )
        for vertex in pseudo_arborescence.vertices():
            if vertex.contains_cycle():
                continue
            final_arborescence._vertices[vertex.label] = digraph._vertices[vertex.label]
        for pseudoedge in pseudo_arborescence.edges():
            final_arborescence._edges[pseudoedge.label] = digraph._edges[pseudoedge.label]
        yield final_arborescence


def optimum_directed_forest(
    digraph: Union[DiGraph, MultiDiGraph], minimum: bool = True, weight: str = "Edge__weight"
) -> Iterator[Tree[Union[DiVertex, MultiDiVertex]]]:
    """Iterates over a minimum (or maximum) :term:`directed forest` of a weighted,
    :term:`directed graph` using Edmonds' algorithm. :cite:`1967:edmonds` A directed forest is also
    called a :term:`branching`.

    Running time: :math:`O(mn)` where :math:`m = |E|` and :math:`n = |V|`

    Note:
        This algorithm is only defined for *directed* graphs. To find the optimum forest of an
        undirected graph, see :func:`optimum_forest
        <vertizee.algorithms.spanning.undirected.optimum_forest>`.

    Args:
        digraph: The directed graph to explore.
        minimum: Optional;  True to return the minimum directed forest, or False to return
            the maximum directed forest. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the edge ``attr``
            dictionary. The default value ("Edge__weight") uses the edge property ``weight``.

    Yields:
        Iterator[Tree[V, E]]: An iterator over the minimum (or maximum) arborescences. If only one
        arborescence is yielded prior to ``StopIteration``, then it is a
        :term:`spanning arborescence`.

    See Also:
        :class:`Tree <vertizee.classes.data_structures.tree.Tree>`
    """
    return edmonds(digraph, minimum, find_spanning_arborescence=False, weight=weight)


def spanning_arborescence(
    digraph: Union[DiGraph, MultiDiGraph], minimum: bool = True, weight: str = "Edge__weight"
) -> Tree[Union[DiVertex, MultiDiVertex]]:
    """Returns a minimum (or maximum) :term:`spanning arborescence` of a weighted,
    :term:`directed graph` using Edmonds' algorithm. :cite:`1967:edmonds`

    Running time: :math:`O(mn)` where :math:`m = |E|` and :math:`n = |V|`

    Note:
        This algorithm is only defined for *directed* graphs. To find the spanning tree of an
        undirected graph, see :func:`spanning_tree
        <vertizee.algorithms.spanning.undirected.spanning_tree>`.

    Args:
        digraph: The directed graph to explore.
        minimum: Optional;  True to return the minimum arborescences, or False to return
            the maximum arborescences. Defaults to True.
        weight: Optional; The key to use to retrieve the weight from the edge ``attr``
            dictionary. The default value ("Edge__weight") uses the edge property ``weight``.

    Returns:
        Tree: The minimum (or maximum) spanning arborescence discovered using Edmonds' algorithm.

    Raises:
        Unfeasible: An Unfeasible exception is raised if the graph does not contain a spanning
            arborescence.

    See Also:
        :class:`Tree <vertizee.classes.data_structures.tree.Tree>`
    """
    return next(edmonds(digraph, minimum, find_spanning_arborescence=True, weight=weight))


def _contract_cycle(
    graph: PseudoGraph,
    cycle_edge: PseudoEdge,
    union_find: UnionFind[PseudoVertex],
    vertex_to_arborescence: Dict[PseudoVertex, Tree[PseudoVertex]],
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
    the graph and replaced by new edges. Each new edge is formed by replacing the vertex that was
    on the cycle with the new vertex formed by contracting the cycle. The original edges are stored
    within the new pseudoedge objects in the property ``previous_version``, that is, each new edge
    stores a reference to the version of itself prior to substituting either its tail or head with
    the new vertex (formed by contracting the cycle).

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
    cycle_start = cycle_edge.vertex1  # the tail vertex
    cycle_end = cycle_edge.vertex2  # the head vertex
    representative_vertex = union_find[cycle_start]
    arborescence: Tree[PseudoVertex] = vertex_to_arborescence[representative_vertex]

    label = graph.create_cycle_label()
    cycle = Cycle(label)
    cycle.add_cycle_edge(cycle_edge)
    graph.cycle_stack.append(cycle)

    new_vertex = PseudoVertex(label, parent_graph=graph)
    new_vertex.cycle = cycle
    arborescence._vertices[new_vertex.label] = new_vertex
    graph.add_vertex_object(new_vertex)

    vertex: PseudoVertex = cycle_start
    cycle.min_weight_edge = cycle_edge
    while vertex != cycle_end:
        next_edge = graph.get_edge(cast(VertexType, vertex.parent_vertex), vertex)
        next_cycle_edge = cast(PseudoEdge, next_edge)

        if next_cycle_edge.weight < cycle.min_weight_edge.weight:
            cycle.min_weight_edge = next_cycle_edge
        cycle.add_cycle_edge(next_cycle_edge)
        # In order to form a cycle, there must be a parent_vertex link for each vertex on the cycle.
        assert vertex.parent_vertex is not None
        vertex = vertex.parent_vertex

    for vertex in cycle.vertices:
        for e in vertex.incident_edges_incoming():
            edge = cast(PseudoEdge, e)
            if edge in cycle.edges or edge not in graph:
                continue

            if edge in arborescence and edge.vertex1 not in cycle.vertices:
                new_vertex.parent_vertex = edge.vertex1

            cycle_vertex = edge.vertex2
            incoming_cycle_edge = None
            for incident_edge in cycle_vertex.incident_edges_incoming():
                if incident_edge in cycle.edges:
                    incoming_cycle_edge = incident_edge
                    break
            if not incoming_cycle_edge:
                raise exception.AlgorithmError(
                    f"no incoming cycle edge found for cycle vertex {cycle_vertex}"
                )

            new_weight = edge.weight + cycle.min_weight_edge.weight - incoming_cycle_edge.weight
            if edge.vertex1 in cycle.vertices:
                new_edge = PseudoEdge(new_vertex, new_vertex, weight=new_weight)
            else:
                new_edge = PseudoEdge(edge.vertex1, new_vertex, weight=new_weight)

            new_edge.previous_version = edge
            cycle.incoming_edges.add(new_edge)
            new_vertex._add_edge(new_edge)

            if edge in arborescence:
                arborescence._edges.pop(edge.label, None)
                arborescence._edges[new_edge.label] = cast(MutableEdgeBase[PseudoVertex], new_edge)
            graph.remove_edge(edge.vertex1, edge.vertex2)
            graph.add_edge_object(new_edge)

        for e in vertex.incident_edges_outgoing():
            edge = cast(PseudoEdge, e)
            if edge in cycle.edges or edge not in graph:
                continue

            head = edge.vertex2
            cycle.outgoing_edge_head_previous_parent[edge] = head.parent_vertex

            if head.parent_vertex and head.parent_vertex == edge.vertex1:
                head.parent_vertex = new_vertex

            if edge.vertex2 in cycle.vertices:
                new_edge = PseudoEdge(new_vertex, new_vertex, weight=0)
            else:
                new_edge = PseudoEdge(new_vertex, edge.vertex2, weight=edge.weight)
            new_edge.previous_version = edge
            cycle.outgoing_edges.add(new_edge)
            new_vertex._add_edge(new_edge)

            if edge in arborescence:
                arborescence._edges.pop(edge.label, None)
                arborescence._edges[new_edge.label] = cast(MutableEdgeBase[PseudoVertex], new_edge)
            graph.remove_edge(edge.vertex1, edge.vertex2)
            graph.add_edge_object(new_edge)

        assert vertex.parent_vertex is not None
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


def _undo_cycle_contraction(
    graph: PseudoGraph, cycle: Cycle, edge_to_delete: PseudoEdge, arborescence: Tree[PseudoVertex]
) -> None:
    """
    This is a helper function that undoes a cycle contraction by adding the cycle edges back to the
    graph and removing one of the cycle edges, thus breaking the cycle. In addition, each incoming
    and outgoing cycle edge is rolled back to its previous version prior to the cycle contraction.
    """
    for edge in cycle.edges:
        arborescence._edges[edge.label] = cast(MutableEdgeBase[PseudoVertex], edge)
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
            assert edge is not None
            assert edge.previous_version is not None
            arborescence._edges.pop(edge.label, None)
            arborescence._edges[edge.previous_version.label] = cast(
                MutableEdgeBase[PseudoVertex], edge.previous_version
            )
        if edge in graph:
            graph.remove_edge(edge.vertex1, edge.vertex2)
            assert edge.previous_version is not None
            graph.add_edge_object(edge.previous_version)
    for edge in cycle.outgoing_edges:
        if edge in arborescence:
            assert edge.previous_version is not None
            arborescence._edges.pop(edge.label, None)
            arborescence._edges[edge.previous_version.label] = cast(
                MutableEdgeBase[PseudoVertex], edge.previous_version
            )
        if edge in graph:
            graph.remove_edge(edge.vertex1, edge.vertex2)
            assert edge.previous_version is not None
            graph.add_edge_object(edge.previous_version)
        assert edge.previous_version is not None
        edge.vertex2.parent_vertex = cycle.outgoing_edge_head_previous_parent[edge.previous_version]

    arborescence._vertices.pop(cycle_vertex.label)
    graph.remove_vertex(cycle_vertex)
