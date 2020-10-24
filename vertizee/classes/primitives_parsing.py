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

"""Provides functionality for parsing vertices and edges in a ``GraphPrimitive`` context, that is,
where vertices and edges may be objects, string literals, or iterables of objects and literals.

The graph-primitives data model enables specifying vertices using integer or string
labels as well as :class:`Vertex <vertizee.classes.vertex.Vertex>` objects. Edges may be specified
as tuples of integers or strings as well as :class:`Edge <vertizee.classes.edge.Edge>` objects. In
addition, these primitives may be passed as iterables.

Classes and type aliases:
    * :class:`GraphPrimitive` - Type alias for primitive graph data types such as
      :class:`Vertex <vertizee.classes.vertex.Vertex>` and :class:`Edge
      <vertizee.classes.edge.Edge>`. In addition, this alias includes edges defined as
      ``Tuple[VertexType, VertexType]`` as well as ``Tuple[VertexType, VertexType, EdgeWeight]``,
      and vertices defined by integer or string labels.
    * :class:`PrimitivesParser` - Class for parsing graph primitives into data objects.
"""

from __future__ import annotations
import collections.abc
import copy
from dataclasses import dataclass, field
import numbers
from typing import Iterable, List, TYPE_CHECKING, Union

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.edge import EdgeClass, EdgeType
    from vertizee.classes.vertex import VertexClass, VertexType

# Type aliases
GraphPrimitive = Union["EdgeType", "VertexType"]


@dataclass
class _VertexData:
    """Lightweight data class to store parsed primitives representing a vertex. If the parsed
    primitive is an instance of a Vertex class, then ``vertex_object`` is set to the object
    reference and ``label`` and ``attr`` are left uninitialized."""
    label: str
    attr: dict = field(default_factory=dict)
    vertex_object: VertexClass

    def get_label(self) -> str:
        """Returns the vertex label."""
        if self.vertex_object:
            return self.vertex_object.label
        return self.label


@dataclass
class _EdgeData:
    """Lightweight data class to store parsed primitives representing an edge. If the parsed
    primitive is an instance of an Edge class, then ``edge_object`` is set to the object
    reference and other attributes are left uninitialized."""
    vertex1: _VertexData
    vertex2: _VertexData
    weight: float = 1.0
    parallel_edge_count: int = 0
    parallel_edge_weights: List[float] = field(default_factory=list)
    attr: dict = field(default_factory=dict)
    edge_object: EdgeClass

    def get_label(self, is_directed: bool = False) -> str:
        """Returns the edge label."""
        if self.edge_object:
            return self.edge_object.label
        if not is_directed:
            if self.vertex1 > self.vertex2:
                return f"({self.vertex2}, {self.vertex1})"
        return f"({self.vertex1}, {self.vertex2})"


@dataclass
class ParsedEdgeAndVertexData:
    """Container class to hold lists of parsed edge and vertex data."""
    edges: List[_EdgeData] = field(default_factory=list)
    vertices: List[_VertexData] = field(default_factory=list)



def parse_edge_type(edge_type: "EdgeType") -> _EdgeData:
    """Parses an ``EdgeType``, which is defined as ``Union[DiEdge, Edge, EdgeLiteral]``, where
    edge literals are tuples specifying vertex endpoints and optional weights and attributes.

    Args:
        edge_type: The edge to parse.

    Returns:
        _EdgeData: Returns the parsed edge.
    """
    # Local import to avoid circular reference.
    # pylint: disable=import-outside-toplevel
    from vertizee.classes.edge import Edge

    edge_data = _EdgeData()
    if isinstance(edge_type, Edge):
        edge_data.edge_object = edge_type
    elif isinstance(edge_type, tuple):
        if len(edge_type) < 2 or len(edge_type) > 4:
            raise ValueError("an edge tuple must contain 2, 3, or 4 items, found tuple of "
                f"length {len(edge_type)}; see 'EdgeType' type alias for more information")
        edge_data.vertex1 = parse_vertex_type(edge_type[0])
        edge_data.vertex2 = parse_vertex_type(edge_type[1])

        if len(edge_type) == 3:
            # Tuple["VertexType", "VertexType", Weight]
            # Tuple["VertexType", "VertexType", AttributesDict]
            if isinstance(edge_type[2], collections.abc.Mapping):
                edge_data.attr = copy.deepcopy(edge_type[2])
            elif isinstance(edge_type[2], numbers.Number):
                edge_data.weight = float(edge_type[2])
            else:
                raise ValueError("the third item in an edge 3-tuple must be an attribute "
                    "mapping (usually a dict) or an edge weight (number); found "
                    f"{type(edge_type[2])}")
        elif len(edge_type) == 4:
            # Tuple["VertexType", "VertexType", Weight, AttributesDict]
            if isinstance(edge_type[2], numbers.Number):
                edge_data.weight = float(edge_type[2])
            else:
                raise ValueError("the third item in an edge 4-tuple must be an edge weight "
                    f"(number); found {type(edge_type[2])}")

            if isinstance(edge_type[3], collections.abc.Mapping):
                edge_data.attr = copy.deepcopy(edge_type[3])
            else:
                raise ValueError("the fourth item in an edge 4-tuple must be an attribute "
                    f"mapping (usually a dict); found {type(edge_type[3])}")
    else:
        raise ValueError("an edge must be specified as a tuple or an object;"
            f" found {type(edge_type)}")

    return edge_data


def parse_graph_primitive(graph_primitive: "GraphPrimitive") -> ParsedEdgeAndVertexData:
    """Parses a graph primitive (an edge or vertex).

    Args:
        graph_primitive: The graph primitive (edge or vertex) to parse.

    Returns:
        ParsedEdgeAndVertexData: Returns the parsed graph primitive.
    """
    # Local import to avoid circular reference.
    # pylint: disable=import-outside-toplevel
    from vertizee.classes import vertex

    parsed_primitive = ParsedEdgeAndVertexData()
    if vertex.is_vertex_type(graph_primitive):
        vertex_data = parse_vertex_type(graph_primitive)
        parsed_primitive.vertices.append(vertex_data)
    else:
        edge_data = parse_edge_type(graph_primitive)
        parsed_primitive.edges.append(edge_data)
    return parsed_primitive


def parse_graph_primitives_from(
    graph_primitives: Iterable["GraphPrimitive"]
) -> ParsedEdgeAndVertexData:
    """Parses a collection of graph primitives (edges and/or vertices).

    Args:
        graph_primitives: The graph primitives to parse.

    Returns:
        ParsedEdgeAndVertexData: Returns the parsed graph primitives.
    """

    parsed_primitives = ParsedEdgeAndVertexData()
    if isinstance(graph_primitives, collections.abc.Iterable):
        for primitive in graph_primitives:
            parsed = parse_graph_primitive(primitive)
            parsed_primitives.vertices.extend(parsed.vertices)
            parsed_primitives.edges.extend(parsed.edges)
    else:
        raise ValueError("graph_primitives must be iterable")

    return parsed_primitives


def parse_vertex_type(vertex_type: "VertexType") -> _VertexData:
    """Parses a ``VertexType``, which is defined as
    ``Union[VertexLabel, VertexTupleAttr, Vertex]`` and where ``VertexTupleAttr`` is defined as
    ``Tuple[VertexLabel, AttributesDict]``. A vertex labels may be a string or integer.

    Args:
        vertex_type: The vertex to parse.

    Returns:
        _VertexData: Returns the parsed vertex and saves the object in ``self.vertices``.
    """
    # Local import to avoid circular reference.
    # pylint: disable=import-outside-toplevel
    from vertizee.classes.vertex import Vertex

    vertex_data = _VertexData()
    if isinstance(vertex_type, Vertex):
        vertex_data.label = vertex_type.label
        vertex_data.attr = copy.deepcopy(vertex_type.attr)
    elif isinstance(vertex_type, tuple):
        if len(vertex_type) != 2:
            raise ValueError("a vertex specified as a tuple must be a 2-tuple of the form "
                "Tuple[VertexLabel, AttributesDict]")
        if not isinstance(vertex_type[0], int) and not isinstance(vertex_type[0], str):
            raise ValueError("a vertex label must be a string or integer; "
                f"found {type(vertex_type[0])}")
        if not isinstance(vertex_type[1], collections.abc.Mapping):
            raise ValueError("a vertex attr dictionary must be a mapping (usually a dict); "
                f"found {type(vertex_type[1])}")
        vertex_data.label = str(vertex_type[0])
        vertex_data.attr = copy.deepcopy(vertex_type[1])
    else:
        if not isinstance(vertex_type, (str, int)):
            raise ValueError("a vertex label must be a string or integer; found "
                f"{type(vertex_type)}")
        vertex_data.label = str(vertex_type)

    return vertex_data
