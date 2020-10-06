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

"""Type aliases and utility methods to handle graph primitives
(:class:`Vertex <vertizee.classes.vertex.Vertex>` and :class:`Edge <vertizee.classes.edge.Edge>`)
when used as function or method arguments.

The graph-primitives data model enables specifying vertices using integer or string
labels as well as :class:`Vertex <vertizee.classes.vertex.Vertex>` objects. Edges may be specified
as tuples of integers or strings as well as :class:`Edge <vertizee.classes.edge.Edge>` objects. In
addition, these primitives may be passed as iterables.

Classes and type aliases:
    * :class:`GraphPrimitive` - Type alias for primitive graph data types such as
      :class:`Vertex <vertizee.classes.vertex.Vertex>` and :class:`Edge
      <vertizee.classes.edge.Edge>`. In addition, this alias includes edges defined as
      ``Tuple[VertexType, VertexType]`` as well as ``Tuple[VertexType, VertexType, EdgeWeight]``,
      where ``VertexType`` is the type alias ``Union[int, str, Vertex]`` and ``EdgeWeight`` is a
      type alias for ``float``. Iterables over the aforementioned types are also defined by the
      ``GraphPrimitive`` alias.
    * :class:`ParsedPrimitives` - Container for the results of parsing arguments that fall under
      the ``GraphPrimitive`` type alias.

Function summary:
    * :func:`get_all_edge_tuples_from_parsed_primitives` - Gets all edge tuples from a
      ``ParsedPrimitives`` object.
    * :func:`get_all_vertices_from_parsed_primitives` - Gets vertex keys (string labels) from a
      ``ParsedPrimitives`` object.
    * :func:`get_edge_tuple_from_parsed_primitives` - Gets one edge tuple from a
      ``ParsedPrimitives`` object.
    * :func:`parse_graph_primitives` - Parses a list of ``GraphPrimitive`` arguments.
"""

from __future__ import annotations
import collections.abc
from typing import Iterable, List, Optional, Set, Tuple, TYPE_CHECKING, Union

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.edge import EdgeType
    from vertizee.classes.vertex import VertexType

# Type aliases
EdgeWeight = float
EdgeVertexPair = Tuple["VertexType", "VertexType"]
EdgeTupleWeighted = Tuple["VertexType", "VertexType", EdgeWeight]
EdgeTuple = Union[EdgeVertexPair, EdgeTupleWeighted]
GraphPrimitiveTerminal = Union["EdgeType", EdgeTuple, "VertexType"]
GraphPrimitive = Union[GraphPrimitiveTerminal, Iterable[GraphPrimitiveTerminal]]


class ParsedPrimitives:
    """Container for storing the results of :func:`parse_graph_primitives`.

    Standalone vertices (i.e. vertices passed as an ``int``, ``str``, or :class:`Vertex
    <vertizee.classes.vertex.Vertex>` object) are added in order to the list ``vertex_labels``.
    Vertices that comprise edges (i.e. passed as a 2-tuple, 3-tuple, or :class:`Edge
    <vertizee.classes.edge.Edge>`) are added to the set ``edge_tuples`` (non-weighted) or
    ``edge_tuples_weighted`` if an edge weight was specified.

    Attributes:
        edge_tuples: A list of tuples of the form ``Tuple[VertexType, VertexType]``, where
            ``VertexType`` is the type alias ``Union[int, str, Vertex]``.
        edge_tuples_weighted: A list of tuples of the form
            ``Tuple[VertexType, VertexType, EdgeWeight]``, where ``EdgeWeight`` is a type alias for
            ``float``.
        vertex_labels: A list of vertex labels, where all labels are of type ``str``.
    """

    def __init__(self) -> None:
        # Pairs of vertices (no weights).
        self.edge_tuples: List[Tuple[str, str]] = list()
        # 3-tuples with two vertices and an edge weight.
        self.edge_tuples_weighted: List[Tuple[str, str, float]] = list()
        # Standalone vertices (not part of a tuple or Edge object).
        self.vertex_labels: List[str] = list()


def get_all_edge_tuples_from_parsed_primitives(
    parsed_primitives: "ParsedPrimitives",
) -> List["EdgeTuple"]:
    """Gets all edge tuples from a :class:`ParsedPrimitives` object.

    If the number of vertices in ``vertex_labels`` is odd, then the last vertex is ignored.

    Args:
        parsed_primitives: The parsed primitive vertex labels to check.

    Returns:
        List[EdgeTuple]: A list of tuples containing vertex labels, or an empty list if no vertices
        found.
    """
    tuples: List[EdgeTuple] = list()
    tuples += parsed_primitives.edge_tuples
    tuples += parsed_primitives.edge_tuples_weighted

    vertex_prev = None
    for vertex_current in parsed_primitives.vertex_labels:
        if vertex_prev is not None:
            tuples.append((vertex_prev, vertex_current))
            vertex_prev = None
        else:
            vertex_prev = vertex_current

    return tuples


def get_all_vertices_from_parsed_primitives(parsed_primitives: "ParsedPrimitives") -> Set[str]:
    """Gets all vertex labels from a :class:`ParsedPrimitives` object.

    Args:
        parsed_primitives: The parsed primitive vertex labels to check.

    Returns:
        Set[str]: A set of vertex labels, or an empty set if no vertices found.
    """
    vertices: Set[str] = set()

    while len(parsed_primitives.edge_tuples) > 0:
        t = parsed_primitives.edge_tuples.pop()
        vertices.add(t[0])
        vertices.add(t[1])
    while len(parsed_primitives.edge_tuples_weighted) > 0:
        t_weighted = parsed_primitives.edge_tuples_weighted.pop()
        vertices.add(t_weighted[0])
        vertices.add(t_weighted[1])

    for vertex_label in parsed_primitives.vertex_labels:
        vertices.add(vertex_label)

    return vertices


def get_edge_tuple_from_parsed_primitives(parsed_primitives: "ParsedPrimitives") -> Optional[Tuple]:
    """Gets a tuple of vertex labels from a :class:`ParsedPrimitives` object.

    If no vertex labels are found, returns None. If only one vertex label found, then the second
    element of the returned tuple will be None, e.g. ``(v1, None)``.

    Args:
        parsed_primitives: The parsed primitive vertex labels to check.

    Returns:
        Tuple: A tuple containing either one or two vertex labels, or None if no vertex labels
        found.
    """
    t: Union[Tuple[str, Optional[str]], Tuple[str, str, float], None] = None
    if len(parsed_primitives.edge_tuples) > 0:
        t = parsed_primitives.edge_tuples[0]
    elif len(parsed_primitives.edge_tuples_weighted) > 0:
        t = parsed_primitives.edge_tuples_weighted[0]
    elif len(parsed_primitives.vertex_labels) == 1:
        t = (parsed_primitives.vertex_labels[0], None)
    elif len(parsed_primitives.vertex_labels) > 1:
        t = (parsed_primitives.vertex_labels[0], parsed_primitives.vertex_labels[1])
    return t


def parse_graph_primitives(*args: "GraphPrimitive") -> "ParsedPrimitives":
    """Parses a list of arguments, which may be any combination of graph primitives.

    Args:
        *args: A sequence of graph primitives. See the definition of ``GraphPrimitive`` above.

    Returns:
        The parsed graph primitives.

    See Also:
        :class:`ParsedPrimitives`
    """
    parsed_primitives = ParsedPrimitives()

    for arg in args:
        is_edge_vertex_or_tuple = _parse_non_iterable_primitive(arg, parsed_primitives)

        if not is_edge_vertex_or_tuple:
            if isinstance(arg, collections.abc.Iterable):
                for x in arg:
                    non_iterable_primitive = _parse_non_iterable_primitive(x, parsed_primitives)
                    if not non_iterable_primitive:
                        raise ValueError(
                            "Expected Iterable of GraphPrimitive to contain non-iterable graph "
                            f"primitives. Found type {type(x).__name__}."
                        )
            else:
                raise ValueError(
                    "Expected arg to be a GraphPrimitive or an Iterable of "
                    f"GraphPrimitive. Found type {type(arg).__name__}."
                )
    return parsed_primitives


def _parse_non_iterable_primitive(
    arg: "GraphPrimitive", parsed_primitives: "ParsedPrimitives"
) -> bool:
    """Parses a single graph primitive ``arg`` and adds it to ``parsed_primitives`` object.

    Args:
        arg: A graph-primitive.
        parsed_primitives: Collection of parsed primitives to which to add the results of parsing
            ``arg``.

    Returns:
        bool: True if ``arg`` was a non-iterable graph primitive, False otherwise.
    """
    # Import locally to avoid circular dependency errors.
    # pylint: disable=import-outside-toplevel
    from vertizee.classes.edge import Edge
    from vertizee.classes.vertex import Vertex

    if isinstance(arg, int):
        parsed_primitives.vertex_labels.append(str(arg))
    elif isinstance(arg, str):
        parsed_primitives.vertex_labels.append(arg)
    elif isinstance(arg, Vertex):
        parsed_primitives.vertex_labels.append(arg.label)
    elif isinstance(arg, Edge):
        edge: Edge = arg
        if edge.weight != 0.0:
            parsed_primitives.edge_tuples_weighted.append(
                (edge.vertex1.label, edge.vertex2.label, edge.weight)
            )
        else:
            parsed_primitives.edge_tuples.append((edge.vertex1.label, edge.vertex2.label))
    elif isinstance(arg, tuple):
        if len(arg) == 3:
            parsed_primitives.edge_tuples_weighted.append((str(arg[0]), str(arg[1]), arg[2]))
        elif len(arg) == 2:
            parsed_primitives.edge_tuples.append((str(arg[0]), str(arg[1])))
        else:
            raise ValueError(
                "GraphPrimitive tuples must have either two items (vertex 1, vertex 2) "
                "or three items (vertex 1, vertex 2, edge weight). "
                f"Found {len(arg)} items."
            )
    else:
        return False

    return True
