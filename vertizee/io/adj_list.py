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

Function summary:
    * :func:`read_adj_list` - Reads an adjacency list from a text file and populates ``new_graph``.
    * :func:`read_weighted_adj_list` - Reads an adjacency list from a text file and populates
      ``new_graph``.
    * :func:`write_adj_list_to_file` - Writes a graph to file as an adjacency list.

Adjacency List
##############

The adjacency list format consists of lines containing vertex labels. The first label in a line
is the source vertex. Further labels in the line are considered destination vertices and are added
to the graph along with an edge between the source vertex and destination vertex.

In the case of undirected graphs, potential edges are first checked against the graph, and only
added if the edge is not present.

**Example: Undirected Graph Adjacency List**

.. code-block:: none

    1 2 2 3
    2 1 1
    3 1

Represents undirected edges:

* :math:`(1, 2)` - two parallel edges
* :math:`(1, 3)` - one edge


**Example: Directed Graph Adjacency List**

.. code-block:: none

    1 2 2 3
    2 1
    3

Represents directed edges:

* :math:`(1,\\ 2)` - two parallel edges
* :math:`(2,\\ 1)` - one edge [parallel to the two edges from :math:`(1,\\ 2)`, but a separate
  :class:`DiEdge <vertizee.classes.edge.DiEdge>` object]
* :math:`(1,\\ 3)` - one edge

Note that directed edges :math:`(1,\\ 2)` and :math:`(2,\\ 1)` would be stored as separate
:class:`DiEdge <vertizee.classes.edge.DiEdge>` objects. In the case of ``DiEdge(1, 2)``, the
``parallel_edge_count`` would be 1. In the case of
``DiEdge(2, 1)``, the ``parallel_edge_count`` would be 0, since there is only one directed edge
with ``tail`` 2 and ``head`` 1.


Weighted Adjacency List
#######################

The weighted adjacency list format consists of lines starting with a source vertex followed
by pairs of destination vertices and their associated edge weights.

**Example: Undirected Graph Adjacency List**

.. code-block:: none

    1   2,100   3,50
    2   3,25
    3

Represents undirected edges:

* :math:`(1,\\ 2,\\ 100)` - edge between vertices 1 and 2 with weight 100
* :math:`(1,\\ 3,\\ 50)` - edge between vertices 1 and 3 with weight 50
* :math:`(2,\\ 3,\\ 25)` - edge between vertices 2 and 3 with weight 25

Alternatively, the same graph could be represented with the following adjacency list:

.. code-block:: none

    1   2,100   3,50
    2   1,100   3,25
    3   1,50    2,25
"""

from __future__ import annotations
from collections import Counter
import re
from typing import Dict, List, Set, TYPE_CHECKING, Tuple

from vertizee.classes.edge import create_label
from vertizee.exception import GraphTypeNotSupported

if TYPE_CHECKING:
    from vertizee.classes.edge import Edge
    from vertizee.classes.graph import G
    from vertizee.classes.primitives_parsing import EdgeTuple
    from vertizee.classes.vertex import Vertex


def read_adj_list(path: str, new_graph: "G", delimiters: str = r",\s*|\s+") -> None:
    """Reads an adjacency list from a text file and populates ``new_graph``.

    The ``new_graph`` is cleared and then vertices and edges are added from the adjacency list.
    The adjacency list is interpreted as either directed or undirected based on the type of
    ``new_graph`` (e.g. :class:`Graph <vertizee.classes.graph.Graph>`,
    :class:`DiGraph <vertizee.classes.digraph.DiGraph>`,
    :class:`MultiDiGraph <vertizee.classes.digraph.MultiDiGraph>`).

    Args:
        path: The adjacency list file path.
        new_graph: The new graph to create from the adjacency list.
        delimiters: Optional; Delimiters used to separate values. Defaults to commas and/or
            Unicode whitespace characters.
    """
    new_graph.clear()
    with open(path, mode="r") as f:
        lines = f.read().splitlines()
    if not lines:
        return

    # Keep track of the source vertex (column one of adj. list) for each edge to avoid duplicates.
    edge_label_to_source: Dict[str, str] = {}
    for line in lines:
        vertices = re.split(delimiters, line)
        vertices = [v for v in vertices if len(v) > 0]

        if not vertices:
            continue
        if len(vertices) == 1:
            new_graph.add_vertex(vertices[0])
            continue

        source = vertices[0]
        destination_vertices = vertices[1:]
        edge_tuples: List[EdgeTuple] = [(source, t) for t in destination_vertices]
        if not edge_tuples:
            new_graph.add_vertex(source)
            continue
        if not new_graph.is_directed():
            edge_tuples = _remove_duplicate_undirected_edges(
                source, edge_tuples, edge_label_to_source
            )
        new_graph.add_edges_from(edge_tuples)


def read_weighted_adj_list(path: str, new_graph: "G") -> None:
    """Reads an adjacency list from a text file and populates ``new_graph``.

    The ``new_graph`` is cleared and then vertices and edges are added from the adjacency list.
    The adjacency list is interpreted as either directed or undirected based on the type of
    ``new_graph`` (e.g. :class:`Graph <vertizee.classes.graph.Graph>`,
    :class:`DiGraph <vertizee.classes.digraph.DiGraph>`,
    :class:`MultiDiGraph <vertizee.classes.digraph.MultiDiGraph>`).

    Args:
        path: The adjacency list file path.
        new_graph: The new graph to create from the adjacency list.
    """
    new_graph.clear()
    with open(path, mode="r") as f:
        lines = f.read().splitlines()
    if not lines:
        return

    # Keep track of the source vertex (column one of adj. list) for each edge to avoid duplicates.
    edge_label_to_source: Dict[str, str] = {}
    for line in lines:
        source_match = re.search(r"\w+\b", line)
        if not source_match:
            continue
        source = source_match.group(0)
        line = line[len(source) :]

        edge_tuples: List[EdgeTuple] = []
        for match in re.finditer(r"\b(\w+)\s*,\s*(\w+)", line):
            edge_tuples.append((source, match.group(1), float(match.group(2))))

        if not edge_tuples:
            new_graph.add_vertex(source)
            continue
        if not new_graph.is_directed():
            edge_tuples = _remove_duplicate_undirected_edges(
                source, edge_tuples, edge_label_to_source
            )
        new_graph.add_edges_from(edge_tuples)


def write_adj_list_to_file(
    path: str,
    graph: "G",
    delimiter: str = "\t",
    include_weights: bool = False,
    weights_are_integers: bool = False,
):
    """Writes a graph to a file as an adjacency list.

    If ``include_weights`` is True, then the adjacency list output format is::

        source1     destination1,weight1     destination2,weight2
        source2     destination3,weight3     destination4,weight4
        ...

    Args:
        path: Path to the output file.
        graph: The graph to write out.
        delimiter: Optional; The delimiter to use. Defaults to the tab character.
        include_weights: Optional; If True, write out the edge weights with the vertices.
        weights_are_integers: Optional; If True, floating point weights are converted to int.
    """
    lines = []

    vertices = graph.vertices
    if all([x.label.isdecimal() for x in vertices]):
        sorted_vertices = sorted(vertices, key=lambda v: int(v.label))
    else:
        sorted_vertices = sorted(vertices, key=lambda v: v.label)

    for vertex in sorted_vertices:
        source_vertex_label = vertex.label
        line = f"{source_vertex_label}"
        if vertex.loops is not None:
            loop_edge: "Edge" = vertex.loops
            line = _add_loop_edges_to_line(
                line, loop_edge, delimiter, include_weights, weights_are_integers
            )

        incident_edges = _get_incident_edges_excluding_loops(graph, vertex)
        if incident_edges is None:
            lines.append(line)
            continue

        sorted_edges = sorted(incident_edges, key=lambda e: e.__str__())
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
    edge: "Edge",
    source_vertex: "Vertex",
    delimiter: str = "\t",
    include_weights: bool = False,
    weights_are_integers: bool = False,
) -> str:

    if edge.vertex1 == source_vertex:
        destination_label = edge.vertex2.label
    else:
        destination_label = edge.vertex1.label

    if include_weights:
        if weights_are_integers:
            weight = str(int(edge.weight))
        else:
            weight = str(edge.weight)

        line += f"{delimiter}{destination_label},{weight}"
        while len(edge._parallel_edge_weights) < edge.parallel_edge_count:
            edge._parallel_edge_weights.append(1)
        for i in range(0, edge.parallel_edge_count):
            if weights_are_integers:
                weight = str(int(edge._parallel_edge_weights[i]))
            else:
                weight = str(edge._parallel_edge_weights[i])
            line += f"{delimiter}{destination_label},{weight}"
    else:  # Exclude edge weights.
        line += f"{delimiter}{destination_label}"
        for i in range(0, edge.parallel_edge_count):
            line += f"{delimiter}{destination_label}"
    return line


def _add_loop_edges_to_line(
    line: str,
    loop_edge: "Edge",
    delimiter: str = "\t",
    include_weights: bool = False,
    weights_are_integers: bool = False,
) -> str:

    source_vertex_label = loop_edge.vertex1.label

    if include_weights:
        if weights_are_integers:
            weight = str(int(loop_edge.weight))
        else:
            weight = str(loop_edge.weight)
        line += f"{delimiter}{source_vertex_label},{weight}"

        # If parallel self-loops are missing weights, set to default weight 1.
        while len(loop_edge._parallel_edge_weights) < loop_edge.parallel_edge_count:
            loop_edge._parallel_edge_weights.append(1)
        for i in range(0, loop_edge.parallel_edge_count):
            if weights_are_integers:
                weight = str(int(loop_edge._parallel_edge_weights[i]))
            else:
                weight = str(loop_edge._parallel_edge_weights[i])
            line += f"{delimiter}{source_vertex_label},{weight}"
    else:
        line += f"{delimiter}{source_vertex_label}"
        for i in range(0, loop_edge.parallel_edge_count):
            line += f"{delimiter}{source_vertex_label}"
    return line


def _get_incident_edges_excluding_loops(
    graph: "G", vertex: "Vertex", reverse_graph: bool = False
) -> Set["Edge"]:
    """Helper function to retrieve the incident edges of a vertex, excluding self loops.

    If `reverse_graph` is True and it is a directed graph, then the child's incoming adjacency
    edges are returned rather than the outgoing edges. This is equivalent to reversing the
    direction of all edges in the digraph.

    Args:
        graph (G): The graph to search.
        vertex (Vertex): The vertex whose incident edges are to be retrieved.
        reverse_graph (bool, optional): For directed graphs, setting to True will yield a traversal
            as if the graph were reversed (i.e. the reverse/transpose/converse graph). Defaults to
            False.
    """
    if graph.is_directed():
        if reverse_graph:
            return vertex.incident_edges_incoming
        return vertex.incident_edges_outgoing

    # undirected graph
    if vertex.loops is not None:
        loop_edges = vertex.loops
        edges = vertex.incident_edges
        edges.remove(loop_edges)
        return edges
    return vertex.incident_edges


def _remove_duplicate_undirected_edges(
    source_vertex_label: str, edge_tuples: List[EdgeTuple], edge_label_to_source: Dict[str, str]
) -> List[EdgeTuple]:
    """For undirected graphs, adjacency lists generally repeat edge entries for each endpoint.
    For example, edges (1, 2), (1, 3) would appear as:

    1   2   3
    2   1
    3   1

    This function removes duplicates, where a duplicate is defined as an edge with the same
    edge label (as defined by the function :func:`create_label
    <vertizee.classes.edge.create_label>`) that maps to a different source vertex. Source
    vertice are the vertices defined by the first column of an adjacency list file.
    """
    cnt: Counter = Counter()
    for t in edge_tuples:
        cnt[t] += 1

    unique_edge_tuples = []
    for t in cnt:
        edge_label = create_edge_label(t[0], t[1], is_directed=False)
        if edge_label not in edge_label_to_source:
            edge_label_to_source[edge_label] = source_vertex_label

        if edge_label_to_source[edge_label] == source_vertex_label:
            for _ in range(cnt[t]):
                unique_edge_tuples.append(t)

        # if graph.has_edge(t):
        #     edge_tuples = [x for x in edge_tuples if x != t]

    return unique_edge_tuples
