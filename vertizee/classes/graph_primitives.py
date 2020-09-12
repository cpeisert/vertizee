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

"""Type aliases and utility methods to handle graph primitives (vertices and edges) in the context
of function arguments.

The graph primitives enable flexible function calls by specifying vertices using integer or string
labels as well as Vertex objects. Edges may be specified as tuples of integers or strings as well
as Edge objects. In addition, these primitives may be part of an Iterable, which enables passing
collections of vertices and edges.

* get_all_edge_tuples_from_parsed_primitives - Gets all edge tuples from `ParsedPrimitives`.
* get_all_vertices_from_parsed_primitives - Gets vertex keys from `ParsedPrimitives`.
* get_edge_tuple_from_parsed_primitives - Gets one edge tuple from `ParsedPrimitives`.
* parse_graph_primitives(*args: GraphPrimitive) - Parses a list of arguments, which may be any
        combination of graph primitives.
* GraphPrimitive - Union type for primitive graph data types (i.e. combinations of vertices and
        edges).
* ParsedPrimitives - Container for the results of parsing function arguments of type
    GraphPrimitive.
"""

import collections.abc
from typing import Iterable, List, Optional, Set, Tuple, TYPE_CHECKING, Union

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.edge import Edge
    from vertizee.classes.vertex import Vertex

# Type aliases
VertexKey = Union[int, str]
EdgeWeight = float
EdgeVertexPair = Tuple[VertexKey, VertexKey]
EdgeTupleWeighted = Tuple[VertexKey, VertexKey, EdgeWeight]
EdgeTuple = Union[EdgeVertexPair, EdgeTupleWeighted]
GraphPrimitiveTerminal = Union["Edge", EdgeTuple, "Vertex", VertexKey]
GraphPrimitive = Union[GraphPrimitiveTerminal, Iterable[GraphPrimitiveTerminal]]


class ParsedPrimitives:
    """Results from `~graph_base.parse_graph_primitives`.

    Standalone vertices (i.e. vertices passed as an int, str, or Vertex object) are added
    in order to the list `vertex_keys`. Vertices that comprise edges (i.e. passed as a 2-tuple,
    3-tuple, or Edge) are added to the set `edge_tuples` (non-weighted) or
    `edge_tuples_weighted` if an edge weight was specified."""

    def __init__(self):
        # Pairs of vertices (no weights).
        self.edge_tuples: List[EdgeTuple] = list()
        # 3-tuples with two vertices and an edge weight.
        self.edge_tuples_weighted: List[EdgeTupleWeighted] = list()
        # Standalone vertices (not part of a tuple or Edge object).
        self.vertex_keys: List[str] = list()


def get_all_edge_tuples_from_parsed_primitives(
    parsed_primitives: "ParsedPrimitives",
) -> List[EdgeTuple]:
    """Gets all edge tuples from a `ParsedPrimitive`object.

    If the number of vertices in `vertex_keys` is odd, then the last vertex is ignored.

    Args:
        parsed_primitives ([ParsedPrimitives]): The parsed primitive vertex keys to check.

    Returns:
        List[EdgeTuple]: A list of tuples containing vertex keys, or an
            empty list if no vertices found.
    """
    tuples: List["EdgeTuple"] = list()
    tuples += parsed_primitives.edge_tuples
    tuples += parsed_primitives.edge_tuples_weighted

    vertex_prev = None
    for vertex_current in parsed_primitives.vertex_keys:
        if vertex_prev is not None:
            tuples.append((vertex_prev, vertex_current))
            vertex_prev = None
        else:
            vertex_prev = vertex_current

    return tuples


def get_all_vertices_from_parsed_primitives(parsed_primitives: "ParsedPrimitives") -> Set[str]:
    """Gets all vertex keys from a `ParsedPrimitive`object.

    Args:
        parsed_primitives (ParsedPrimitives): The parsed primitive vertex keys to check.

    Returns:
        Set[str]: A set of vertex keys, or an empty set if no vertices found.
    """
    vertices: Set[str] = set()

    while len(parsed_primitives.edge_tuples) > 0:
        t = parsed_primitives.edge_tuples.pop()
        vertices.add(t[0])
        vertices.add(t[1])
    while len(parsed_primitives.edge_tuples_weighted) > 0:
        t = parsed_primitives.edge_tuples_weighted.pop()
        vertices.add(t[0])
        vertices.add(t[1])

    for vertex_key in parsed_primitives.vertex_keys:
        vertices.add(vertex_key)

    return vertices


def get_edge_tuple_from_parsed_primitives(parsed_primitives: "ParsedPrimitives") -> Optional[Tuple]:
    """Gets a tuple of vertex keys from a `ParsedPrimitive`object.

    If no vertex keys are found, returns None. If only one vertex key found, then the second
    element of the returned tuple will be None[e.g. (v1, None)].

    Args:
        parsed_primitives ([ParsedPrimitives]): The parsed primitive vertex keys to check.

    Returns:
        Tuple: A tuple containing either one or two vertex keys, or None if no vertex keys found.
    """
    t = None
    if len(parsed_primitives.edge_tuples) > 0:
        t = parsed_primitives.edge_tuples[0]
    elif len(parsed_primitives.edge_tuples_weighted) > 0:
        t = parsed_primitives.edge_tuples_weighted[0]
    elif len(parsed_primitives.vertex_keys) == 1:
        t = (parsed_primitives.vertex_keys[0], None)
    elif len(parsed_primitives.vertex_keys) > 1:
        t = (parsed_primitives.vertex_keys[0], parsed_primitives.vertex_keys[1])
    return t


def parse_graph_primitives(*args: GraphPrimitive) -> "ParsedPrimitives":
    """Parse a list of arguments, which may be any combination of graph primitives.

    Graph Primitive Data Types
    * VertexKey = Union[int, str]
    * Vertex
    * Edge
    * EdgeVertexPair = Tuple[VertexKey, VertexKey]
    * EdgeTupleWeighted = Tuple[VertexKey, VertexKey, EdgeWeight]
    * EdgeTuple = Union[EdgeVertexPair, EdgeTupleWeighted]
    * GraphPrimitiveTerminal = Union['Edge', EdgeTuple, Vertex, VertexKey]
    * GraphPrimitive = Union[GraphPrimitiveTerminal, Iterable[GraphPrimitiveTerminal]]

    Args:
        *args (GraphPrimitive): A sequence of graph primitives (vertices, edges, and iterables
            over vertices and edges).

    See Also:
        `~graph_base.ParsedPrimitives`

    :return: The parsed graph primitives.
    """
    parsed_primitives = ParsedPrimitives()

    for arg in args:
        non_iterable_primitive = _parse_non_iterable_graph_primitive(arg, parsed_primitives)

        if not non_iterable_primitive:
            if isinstance(arg, collections.abc.Iterable):
                for x in arg:
                    non_iterable_primitive = _parse_non_iterable_graph_primitive(
                        x, parsed_primitives
                    )
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


def _parse_non_iterable_graph_primitive(
    arg: GraphPrimitiveTerminal, parsed_primitives: "ParsedPrimitives"
) -> bool:
    """Parse a single graph primitive `arg` and add it to `parsed_primitives`.

    Args:
        arg (GraphPrimitiveTerminal): A graph primitive terminal item (i.e. non-iterable).
        parsed_primitives (ParsedPrimitives): Collection of parsed primitives to which to add the
            results of parsing `arg`.

    Returns:
        bool: True if `arg` was a non-iterable graph primitive, False otherwise.
    """
    # Import locally to avoid circular dependency errors.
    # pylint: disable=import-outside-toplevel
    from vertizee.classes.edge import Edge
    from vertizee.classes.vertex import Vertex

    if isinstance(arg, int):
        parsed_primitives.vertex_keys.append(str(arg))
    elif isinstance(arg, str):
        parsed_primitives.vertex_keys.append(arg)
    elif isinstance(arg, Vertex):
        parsed_primitives.vertex_keys.append(arg.key)
    elif isinstance(arg, Edge):
        edge: Edge = arg
        if edge.weight != 0.0:
            parsed_primitives.edge_tuples_weighted.append(
                (edge.vertex1.key, edge.vertex2.key, edge.weight)
            )
        else:
            parsed_primitives.edge_tuples.append((edge.vertex1.key, edge.vertex2.key))
    elif isinstance(arg, tuple):
        if len(arg) == 3:
            parsed_primitives.edge_tuples_weighted.append((str(arg[0]), str(arg[1]), arg[2]))
        elif len(arg) == 2:
            parsed_primitives.edge_tuples.append((str(arg[0]), str(arg[1])))
        else:
            raise ValueError(
                "GraphPrimitive tuples must have either two items (vertex key 1, vertex key 2) "
                "or three items (vertex key 1, vertex key 2, edge weight). "
                f"Found {len(arg)} items."
            )
    else:
        return False

    return True
