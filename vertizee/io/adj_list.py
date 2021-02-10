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
==============
Adjacency list
==============

Subroutines for reading and writing :term:`graph` adjacency lists.


Adjacency list format
=====================

The adjacency list format consists of lines containing :term:`vertex` labels. The first label in a
line is the source vertex. Further labels in the line are considered destination vertices and are
added to the :term:`graph` along with an :term:`edge` between the source vertex and destination
vertex.

In the case of :term:`undirected graphs <undirected graph>`, potential edges are first checked
against the graph, and only added if the edge is not present.

**Example**: Undirected :term:`multigraph` adjacency list

.. code-block:: none

    1 2 2 3
    2 1 1
    3 1

Represents :term:`undirected edges <undirected edge>`:

* :math:`(1, 2)` - two parallel edges
* :math:`(1, 3)` - one edge


**Example**:  :term:`multidigraph` adjacency list

.. code-block:: none

    1 2 2 3
    2 1
    3

Represents :term:`directed edges <directed edge>`:

* :math:`(1, 2)` - two parallel edges
* :math:`(2, 1)` - one edge [same :term:`endpoints <endpoint>` as :math:`(1, 2)`, but a separate
  :class:`MultiDiEdge <vertizee.classes.edge.MultiDiEdge>` object pointed in the opposite direction]
* :math:`(1, 3)` - one edge


Weighted adjacency list
=======================

The :term:`weighted` adjacency list format consists of edges starting with a source vertex followed
by pairs of destination vertices and their associated edge weights.

**Example**: Undirected, :term:`weighted` :term:`graph` adjacency list

.. code-block:: none

    1   2,100   3,50
    2   3,25
    3

Represents :term:`undirected edges <undirected edge>`:

* :math:`(1, 2, 100)` - edge between vertices 1 and 2 with weight 100
* :math:`(1, 3, 50)` - edge between vertices 1 and 3 with weight 50
* :math:`(2, 3, 25)` - edge between vertices 2 and 3 with weight 25

Alternatively, the same graph could be represented with the following adjacency list:

.. code-block:: none

    1   2,100   3,50
    2   1,100   3,25
    3   1,50    2,25


Function summary
================

* :func:`read_adj_list` - Reads an adjacency list from a text file and initializes a graph object
  with the vertices and edges represented by the adjency list.
* :func:`read_weighted_adj_list` - Reads a weighted adjacency list from a text file and initializes
  a graph object with the vertices and edges represented by the adjency list.
* :func:`write_adj_list_to_file` - Writes a graph to file as an adjacency list.


Detailed documentation
======================
"""

from __future__ import annotations
import collections
import re
from typing import Any, Counter, Dict, List, Tuple, TYPE_CHECKING, TypeVar, Union

from vertizee.classes import edge as edge_module
from vertizee.classes.edge import EdgeBase, EdgeTuple, EdgeTupleWeighted, MultiEdgeBase

if TYPE_CHECKING:
    from vertizee.classes.graph import GraphBase
    from vertizee.classes.vertex import V, VertexBase


def read_adj_list(path: str, new_graph: "GraphBase[V]", delimiters: str = r",\s*|\s+") -> None:
    """Reads an adjacency list from a text file and populates ``new_graph``.

    The ``new_graph`` is cleared and then :term:`vertices <vertex>` and :term:`edges <edge>` are
    added from the adjacency list. The adjacency list is interpreted as either directed or
    undirected based on the type of ``new_graph`` (for example,
    :class:`Graph <vertizee.classes.graph.Graph>`,
    :class:`MultiDiGraph <vertizee.classes.graph.MultiDiGraph>`).

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


def read_weighted_adj_list(path: str, new_graph: "GraphBase[V]") -> None:
    """Reads a :term:`weighted` adjacency list from a text file and populates ``new_graph``.

    The ``new_graph`` is cleared and then :term:`vertices <vertex>` and :term:`edges <edge>` are
    added from the adjacency list. The adjacency list is interpreted as either directed or
    undirected based on the type of ``new_graph`` (for example,
    :class:`Graph <vertizee.classes.graph.Graph>`,
    :class:`MultiDiGraph <vertizee.classes.graph.MultiDiGraph>`).

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

        edge_tuples: List[EdgeTupleWeighted] = []
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
    graph: "GraphBase[V]",
    delimiter: str = "\t",
    include_weights: bool = False,
    weights_are_integers: bool = False,
) -> None:
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

    vertices = graph.vertices()
    if all([x.label.isdecimal() for x in vertices]):
        sorted_vertices = sorted(vertices, key=lambda v: int(v.label))
    else:
        sorted_vertices = sorted(vertices, key=lambda v: v.label)

    for vertex in sorted_vertices:
        source_vertex_label = vertex.label
        line = f"{source_vertex_label}"

        if vertex.incident_edges() is None:
            lines.append(line)
            continue

        sorted_edges = sorted(vertex.incident_edges())
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
    edge: EdgeBase[VertexBase],
    source_vertex: VertexBase,
    delimiter: str = "\t",
    include_weights: bool = False,
    weights_are_integers: bool = False,
) -> str:

    if edge.vertex1 == source_vertex:
        destination_label = edge.vertex2.label
    else:
        destination_label = edge.vertex1.label

    if include_weights:
        if isinstance(edge, MultiEdgeBase):
            for connection in edge.connections():
                if weights_are_integers:
                    weight = str(int(connection.weight))
                else:
                    weight = str(connection.weight)
                line += f"{delimiter}{destination_label},{weight}"
        else:
            if weights_are_integers:
                weight = str(int(edge.weight))
            else:
                weight = str(edge.weight)
            line += f"{delimiter}{destination_label},{weight}"

    else:  # Exclude edge weights.
        line += f"{delimiter}{destination_label}"
        if isinstance(edge, MultiEdgeBase):
            for _ in range(1, edge.multiplicity):
                line += f"{delimiter}{destination_label}"
    return line


T = TypeVar("T", bound=Union[Tuple[Any, Any], Tuple[Any, Any, Any]])


def _remove_duplicate_undirected_edges(
    source_vertex_label: str, edge_tuples: List[T], edge_label_to_source: Dict[str, str]
) -> List[T]:
    """For undirected graphs, adjacency lists generally repeat edge entries for each endpoint.
    For example, edges (1, 2), (1, 3) would appear as:

    1   2   3
    2   1
    3   1

    This function removes duplicates, where a duplicate is defined as an edge with the same
    edge label (as defined by the function :func:`create_label
    <vertizee.classes.edge.create_label>`) that maps to a different source vertex. Source
    vertices are defined by the first column of an adjacency list file.
    """
    cnt: Counter[Any] = collections.Counter()
    for t in edge_tuples:
        cnt[t] += 1

    unique_edge_tuples = []
    for t in cnt:
        edge_label = edge_module.create_edge_label(t[0], t[1], is_directed=False)
        if edge_label not in edge_label_to_source:
            edge_label_to_source[edge_label] = source_vertex_label

        if edge_label_to_source[edge_label] == source_vertex_label:
            for _ in range(cnt[t]):
                unique_edge_tuples.append(t)

    return unique_edge_tuples
