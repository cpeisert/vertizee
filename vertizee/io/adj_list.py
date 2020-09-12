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

"""
Subroutines for reading and writing graph adjacency lists.

Adjacency List

The adjacency list format consists of lines with vertex labels. The first label in a line is the
source vertex. Further labels in the line are considered destination vertices and are added to the
graph along with an edge between the source vertex and destination vertex.

In the case of undirected graphs, potential edges are first checked against the graph, and only
added if the edge is not present.

Example::

    UNDIRECTED GRAPH ADJACENCY LIST
    1 2 2 3
    2 1 1
    3 1

    Represents undirected edges:
    (1, 2) - two parallel edges
    (1, 3) - one edge


    DIRECTED GRAPH ADJACENCY LIST
    1 2 2 3
    2 1
    3

    Represents directed edges:
    (1, 2) - two parallel edges
    (2, 1) - one edge [parallel to the two edges from (1, 2)]
    (1, 3) - one edge

    Note that directed edges (1, 2) and (2, 1) would be stored as separate DiEdge
    objects. In the case if DiEdge(1, 2), the `parallel_edge_count` would be 1. In the case of
    DiEdge(2, 1), the `parallel_edge_count` would be 0, since there is only one directed edge with
    tail 2 and head 1.


Weighted Adjacency List

The weighted adjacency list format consists of lines starting with a source vertex label followed
by pairs of destination vertex labels and associated edge weights.

Example::

    UNDIRECTED GRAPH ADJACENCY LIST
    1   2,100   3,50
    2   3,25
    3

    Represents undirected edges:
    (1, 2, 100) - edge between vertices 1 and 2 with weight 100
    (1, 3, 50) - edge between vertices 1 and 3 with weight 50
    (2, 3, 25) - edge between vertices 2 and 3 with weight 25

    Alternatively, the same graph could be represented with the following adjacency list:
    1   2,100   3,50
    2   1,100   3,25
    3   1,50    2,25
"""

from collections import Counter
import re
from typing import List, Set, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from vertizee.classes.edge import EdgeType
    from vertizee.classes.graph_base import GraphBase
    from vertizee.classes.vertex import Vertex


def read_adj_list(path: str, new_graph: "GraphBase", delimiters: str = r",\s*|\s+"):
    """Reads an adjacency list from a text file and populates `new_graph`.

    The `new_graph` is cleared and then vertices and edges are added from the adjacency list.
    The adjacency list is interpreted as either directed or undirected edges to match the type of
    `new_graph` (e.g. Graph, DiGraph, MultiDiGraph).

    Args:
        path (str): The adjacency list file path.
        new_graph (GraphBase): The new graph to create from the adjacency list.
        delimiters (str, optional): File delimiters. Defaults to optional commas and Unicode
            whitespace characters.
    """
    new_graph.clear()
    with open(path, mode="r") as f:
        lines = f.read().splitlines()
    if lines is None or len(lines) == 0:
        return

    for line in lines:
        vertices = re.split(delimiters, line)
        vertices = [v for v in vertices if len(v) > 0]

        if len(vertices) == 0:
            continue
        if len(vertices) == 1:
            new_graph.add_vertex(vertices[0])
            continue

        source = vertices[0]
        destination_vertices = vertices[1:]
        edge_tuples: List[Tuple[str, str]] = [(source, t) for t in destination_vertices]
        if len(edge_tuples) == 0:
            new_graph.add_vertex(source)
            continue
        if not new_graph.is_directed_graph():
            edge_tuples = _remove_duplicate_edges(new_graph, edge_tuples)
        new_graph.add_edges_from(edge_tuples)


def read_weighted_adj_list(path: str, new_graph: "GraphBase"):
    """Reads an adjacency list from a text file and populates `new_graph`.

    The `new_graph` is cleared and then vertices and edges are added from the adjacency list.
    The adjacency list is interpreted as either directed or undirected edges to match the type of
    `new_graph` (e.g. Graph, DiGraph, MultiDiGraph).

    Args:
        path (str): The adjacency list file path.
        new_graph (GraphBase): The new graph to create from the adjacency list.
    """
    new_graph.clear()
    with open(path, mode="r") as f:
        lines = f.read().splitlines()
    if lines is None or len(lines) == 0:
        return

    for line in lines:
        source_match = re.search(r"\w+\b", line)
        if not source_match:
            continue
        source = source_match.group(0)
        line = line[len(source) :]

        edge_tuples: List[Tuple[str, str, float]] = []
        for match in re.finditer(r"\b(\w+)\s*,\s*(\w+)", line):
            edge_tuples.append((source, match.group(1), float(match.group(2))))

        if len(edge_tuples) == 0:
            new_graph.add_vertex(source)
            continue
        if not new_graph.is_directed_graph():
            edge_tuples = _remove_duplicate_edges(new_graph, edge_tuples)
        new_graph.add_edges_from(edge_tuples)


def write_adj_list_to_file(
    path: str,
    graph: "GraphBase",
    delimiter: str = "\t",
    include_weights: bool = False,
    weights_are_integers: bool = False,
):
    """Writes a graph as an adjacency list to a file.

    If `include_weights` is True, then the adjacency list output format is:
    source1     destination1,weight1     destination2,weight2
    source2     destination3,weight3     destination4,weight4
    ...

    Args:
        path (str): Path to the output file.
        graph (GraphBase): The graph to write out.
        delimiter (str, optional): The delimiter to use to separate vertex-weight pairs in the
            output file. Defaults to '\t' (tab).
        include_weights (bool, optional): If True, write out the edge weights with the vertices.
            See
        weights_are_integers (bool, optional): If True, floating point weights are converted to int.
    """
    lines = []

    vertices = graph.vertices
    if all([x.key.isdecimal() for x in vertices]):
        vertices = sorted(vertices, key=lambda v: int(v.key))
    else:
        vertices = sorted(vertices, key=lambda v: v.key)

    for vertex in vertices:
        source_vertex_key = vertex.key
        line = f"{source_vertex_key}"
        if len(vertex.loops) > 0:
            loop_edge: "EdgeType" = next(iter(vertex.loops))
            line = _add_loop_edges_to_line(
                line, loop_edge, delimiter, include_weights, weights_are_integers
            )

        adj_edges = _get_adj_edges_excluding_loops(graph, vertex)
        if adj_edges is None:
            lines.append(line)
            continue

        sorted_edges = sorted(adj_edges, key=lambda e: e.__str__())
        for edge in sorted_edges:
            line = _add_edge_to_line(
                line, edge, vertex, delimiter, include_weights, weights_are_integers
            )

        lines.append(line)

    with open(path, mode="w") as f:
        for line in lines:
            f.write(f"{line}\n")


def _add_edge_to_line(
    line: str,
    edge: "EdgeType",
    source_vertex: "Vertex",
    delimiter: str = "\t",
    include_weights: bool = False,
    weights_are_integers: bool = False,
) -> str:

    if edge.vertex1 == source_vertex:
        destination_key = edge.vertex2.key
    else:
        destination_key = edge.vertex1.key

    if include_weights:
        if weights_are_integers:
            weight = int(edge.weight)
        else:
            weight = edge.weight
        line += f"{delimiter}{destination_key},{weight}"
        while len(edge._parallel_edge_weights) < edge.parallel_edge_count:
            edge._parallel_edge_weights.append(1)
        for i in range(0, edge.parallel_edge_count):
            if weights_are_integers:
                weight = int(edge._parallel_edge_weights[i])
            else:
                weight = edge._parallel_edge_weights[i]
            line += f"{delimiter}{destination_key},{weight}"
    else:  # Exclude edge weights.
        line += f"{delimiter}{destination_key}"
        for i in range(0, edge.parallel_edge_count):
            line += f"{delimiter}{destination_key}"
    return line


def _add_loop_edges_to_line(
    line: str,
    loop_edge: "EdgeType",
    delimiter: str = "\t",
    include_weights: bool = False,
    weights_are_integers: bool = False,
) -> str:

    source_vertex_key = loop_edge.vertex1.key

    if include_weights:
        if weights_are_integers:
            weight = int(loop_edge.weight)
        else:
            weight = loop_edge.weight
        line += f"{delimiter}{source_vertex_key},{weight}"

        # If parallel self-loops are missing weights, set to default weight 1.
        while len(loop_edge._parallel_edge_weights) < loop_edge.parallel_edge_count:
            loop_edge._parallel_edge_weights.append(1)
        for i in range(0, loop_edge.parallel_edge_count):
            if weights_are_integers:
                weight = int(loop_edge._parallel_edge_weights[i])
            else:
                weight = loop_edge._parallel_edge_weights[i]
            line += f"{delimiter}{source_vertex_key},{weight}"
    else:
        line += f"{delimiter}{source_vertex_key}"
        for i in range(0, loop_edge.parallel_edge_count):
            line += f"{delimiter}{source_vertex_key}"
    return line


def _get_adj_edges_excluding_loops(
    graph: "GraphBase", vertex: "Vertex", reverse_graph: bool = False
) -> Set["EdgeType"]:
    """Helper function to retrieve the adjacent edges of a vertex, excluding self loops.

    If `reverse_graph` is True and it is a directed graph, then the child's incoming adjacency
    edges are returned rather than the outgoing edges. This is equivalent to reversing the
    direction of all edges in the digraph.

    Args:
        graph (GraphBase): The graph to search.
        vertex (Vertex): The vertex whose adjacent edges are to be retrieved.
        reverse_graph (bool, optional): For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.
    """
    if graph.is_directed_graph():
        if reverse_graph:
            return vertex.edges_incoming
        else:
            return vertex.edges_outgoing
    else:  # undirected graph
        if len(vertex.loops) > 0:
            loop_edges = next(iter(vertex.loops))
            edges = vertex.edges
            edges.remove(loop_edges)
            return edges
        else:
            return vertex.edges


def _remove_duplicate_edges(graph: "GraphBase", edge_tuples: List[Tuple]) -> List[Tuple]:
    """For undirected graphs, adjacency lists generally repeat edge entries for each endpoint.
    For example, edges (1, 2), (1, 3) would appear as:

    1   2   3
    2   1
    3   1

    This function checks the graph to see if a given edge has already been added and removes
    duplicates.
    """
    if graph.is_directed_graph():
        raise ValueError("graph was a directed graph; function only defined for undirected graphs")
    cnt = Counter()
    for t in edge_tuples:
        cnt[t] += 1

    for t in cnt:
        if graph.has_edge(t):
            edge_tuples = [x for x in edge_tuples if x != t]

    return edge_tuples
