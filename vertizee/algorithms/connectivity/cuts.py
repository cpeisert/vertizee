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

"""Graph algorithms for finding and evaluating cuts in a graph."""

import math
from typing import Dict

from vertizee.classes.edge import EdgeType
from vertizee.classes.graph_base import GraphBase
from vertizee.classes.vertex import Vertex


class KargerResults:
    """Container class to store the results of computing the minimum cut.

    Attributes:
        contracted_graph: The modified graph after edge contractions.
        cut_edge: The final edge comprising the minimum cut.
        karger_contract_run_count: Count of algorithm iterations.
    """
    def __init__(self, contracted_graph: GraphBase = None, cut_edge: EdgeType = None,
                 karger_contract_run_count: int = 0):
        self.contracted_graph: GraphBase = contracted_graph
        self.cut_edge: EdgeType = cut_edge
        self.karger_contract_run_count = karger_contract_run_count


def brute_force_min_cut(graph: GraphBase) -> KargerResults:
    """Uses multiple iterations of the Karger algorithm to find the minimum cut of the graph.

    Note that for the Karger algorithm to be guaranteed of finding a minimum cut on a graph with
    n vertices, it must be run at least n^2 * log(n) iterations.

    See Also:
        `Karger algorithm <https://en.wikipedia.org/wiki/Karger%27s_algorithm>`_

    Args:
        graph: The multigraph in which to find the minimum cut.

    Returns:
        The minimum cut, which is the final EdgeType object containing the last two vertices after
        all other vertices have been merged through the edge contraction process.
    """
    n = graph.vertex_count
    run_iterations = 4
    if n > 2:
        run_iterations = int((n ** 2) * math.log2(n))

    cuts: Dict[int, EdgeType] = {}
    for _ in range(run_iterations):
        results: KargerResults = karger_contract(graph)
        if results.cut_edge is not None:
            cut_size = results.cut_edge.parallel_edge_count + 1
            cuts[cut_size] = results.cut_edge

    min_cut_size = min(cuts.keys())
    return KargerResults(cut_edge=cuts[min_cut_size], karger_contract_run_count=run_iterations)


"""
    Use the Karger-Stein algorithm (https://en.wikipedia.org/wiki/Karger%27s_algorithm) to find
    the minimum cut size of the graph.
    :param graph: The multi-graph to cut.
    :return: The cut, which is the final EdgeType object containing the last two vertices after all
        other vertices have been merged through the edge contraction process.
    """

def fast_min_cut(graph: GraphBase) -> KargerResults:
    """Uses the Karger-Stein algorithm to find the minimum cut of the graph.

    See Also:
        `Karger-Stein algorithm <https://en.wikipedia.org/wiki/Karger%27s_algorithm>`_

    Args:
        graph: The multigraph in which to find the minimum cut.

    Returns:
        The minimum cut, which is the final EdgeType object containing the last two vertices after
        all other vertices have been merged through the edge contraction process.
    """
    if graph.vertex_count <= 6:
        return brute_force_min_cut(graph)

    t = math.ceil(1 + (graph.vertex_count / math.sqrt(2)))
    results1: KargerResults = karger_contract(graph, t)
    results2: KargerResults = karger_contract(graph, t)
    fmc_results1 = fast_min_cut(results1.contracted_graph)
    fmc_results2 = fast_min_cut(results2.contracted_graph)
    total_contract_runs = 2 + fmc_results1.karger_contract_run_count \
        + fmc_results2.karger_contract_run_count
    if fmc_results1.cut_edge.parallel_edge_count <= fmc_results2.cut_edge.parallel_edge_count:
        return KargerResults(
            cut_edge=fmc_results1.cut_edge, karger_contract_run_count=total_contract_runs)
    else:
        return KargerResults(
            cut_edge=fmc_results2.cut_edge, karger_contract_run_count=total_contract_runs)


def karger_contract(graph: GraphBase, minimum_vertices: int = 2) -> KargerResults:
    """Use the Karger algorithm to contract the graph by repeatedly selecting a random edge,
    removing the edge, and merging the vertices of the deleted edge.

    This process is continued until the number of vertices |V| <= minimum_vertices.

    Args:
        graph: The multigraph for which a minimum cut is to be found. A deep copy is made such that
            the caller can be guaranteed that the original graph passed to this function is not
            modified.
        minimum_vertices: Optional; The minimum number of vertices to leave in the graph after
            performing repeated edge removal and vertex merging. Defaults to 2.

    Returns:
        The minimum cut, which is the final EdgeType object containing the last two vertices after
        all other vertices have been merged through the edge contraction process.
    """
    if len(graph.edges) < 1:
        return KargerResults(contracted_graph=graph, cut_edge=None)
    if minimum_vertices < 2:
        minimum_vertices = 2

    contracted_graph = graph.deepcopy()
    while contracted_graph.vertex_count > minimum_vertices:
        random_edge: EdgeType = contracted_graph.get_random_edge()
        merged_vertex: Vertex = random_edge.vertex1
        contracted_graph.remove_edge_from(random_edge)
        contracted_graph.merge_vertices(merged_vertex, random_edge.vertex2)
        merged_vertex.delete_loops()

    cut_edge = None
    if len(contracted_graph.edges) == 1:
        cut_edge = contracted_graph.edges.pop()
    return KargerResults(contracted_graph=contracted_graph, cut_edge=cut_edge)
