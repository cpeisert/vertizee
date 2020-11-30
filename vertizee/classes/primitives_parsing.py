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
from typing import Iterable, List, Optional, TYPE_CHECKING, Union

from vertizee import exception
from vertizee.classes import edge as edge_module
from vertizee.classes import vertex as vertex_module
from vertizee.classes.edge import Connection, MultiConnection
from vertizee.classes.vertex import VertexBase

# pylint: disable=cyclic-import
if TYPE_CHECKING:
    from vertizee.classes.edge import EdgeClass, EdgeType
    from vertizee.classes.vertex import VertexClass, VertexType

# Type aliases
GraphPrimitive = Union["EdgeType", "VertexType"]


class VertexData:
    """Lightweight class to store parsed primitives representing a vertex. If the parsed
    primitive is an instance of a vertex class, then ``vertex_object`` is set to the object
    reference. The property getters retrieve data from ``vertex_object`` if it exists."""

    def __init__(self, label: str):
        self.label = label
        self._attr = dict()
        self._vertex_object: Optional[VertexClass] = None

    @classmethod
    def from_vertex_obj(cls, vertex: VertexClass) -> VertexData:
        """Factory class method to create VertexData from a vertex object."""
        vertex_data = VertexData(vertex.label)
        vertex_data._vertex_object = vertex
        return vertex_data

    @property
    def attr(self) -> Optional[dict]:
        """The attributes dictionary."""
        if self._vertex_object and self._vertex_object.has_attributes_dict():
            return self._vertex_object.attr
        return self._attr

    @property
    def vertex_object(self) -> Optional[VertexClass]:
        """Optional vertex object, if a vertex object was parsed."""
        return self._vertex_object


class EdgeData:
    """Lightweight class to store parsed primitives representing an edge. If the parsed
    primitive is an instance of an edge class, then ``edge_object`` is set to the object
    reference. The property getters retrieve data from ``edge_object`` if it exists."""

    def __init__(self, vertex1: VertexData, vertex2: VertexData):
        self._vertex1 = vertex1
        self._vertex2 = vertex2
        self._weight: float = edge_module.DEFAULT_WEIGHT
        self._attr = dict()
        self._edge_object: Optional[EdgeClass] = None

    @property
    def attr(self) -> dict:
        """The attributes dictionary."""
        if self._edge_object and self._edge_object.has_attributes_dict():
            return self._edge_object.attr
        return self._attr

    @property
    def edge_object(self) -> Optional[EdgeClass]:
        """Optional edge object, if an edge object was parsed."""
        return self._edge_object

    @classmethod
    def from_edge_obj(cls, edge: EdgeClass) -> EdgeData:
        """Factory class method to create EdgeData from an edge object."""
        vertex1 = VertexData.from_vertex_obj(edge.vertex1)
        vertex2 = VertexData.from_vertex_obj(edge.vertex2)
        edge_data = EdgeData(vertex1, vertex2)
        edge_data._edge_object = edge
        return edge_data

    def get_label(self, is_directed: bool) -> str:
        """Returns the edge label."""
        if not is_directed:
            if self.vertex1.label > self.vertex2.label:
                return f"({self.vertex2.label}, {self.vertex1.label})"
        return f"({self.vertex1.label}, {self.vertex2.label})"

    @property
    def vertex1(self) -> VertexData:
        """The first vertex."""
        return self._vertex1

    @property
    def vertex2(self) -> VertexData:
        """The second vertex."""
        return self._vertex2

    @property
    def weight(self) -> float:
        """The edge weight."""
        if self._edge_object:
            return self._edge_object.weight
        return self._weight


@dataclass
class ParsedEdgeAndVertexData:
    """Container class to hold lists of parsed edge and vertex data."""
    edges: List[EdgeData] = field(default_factory=list)
    vertices: List[VertexData] = field(default_factory=list)


def parse_edge_type(edge: "EdgeType") -> EdgeData:
    """Parses an ``EdgeType``, which is defined as ``Union[EdgeClass, EdgeLiteral]``, where
    edge literals are tuples specifying vertex endpoints and optional weights and attributes.

    Args:
        edge: The edge to parse.

    Returns:
        EdgeData: Returns the parsed edge.
    """
    if isinstance(edge, (Connection, MultiConnection)):
        edge_data = EdgeData.from_edge_obj(edge)
    elif isinstance(edge, tuple):
        if len(edge) < 2 or len(edge) > 4:
            raise exception.VertizeeException("an edge tuple must contain 2, 3, or 4 items, found "
                f"tuple of length {len(edge)}; see 'EdgeType' type alias for more information")
        vertex1 = parse_vertex_type(edge[0])
        vertex2 = parse_vertex_type(edge[1])
        edge_data = EdgeData(vertex1, vertex2)

        if len(edge) == 3:
            # Tuple["VertexType", "VertexType", Weight]
            # Tuple["VertexType", "VertexType", AttributesDict]
            if isinstance(edge[2], collections.abc.Mapping):
                edge_data._attr = copy.deepcopy(edge[2])
            elif isinstance(edge[2], numbers.Number):
                edge_data._weight = float(edge[2])
            else:
                raise exception.VertizeeException("the third item in an edge 3-tuple must be an "
                    "attribute dictionary or an edge weight (number); found "
                    f"{type(edge[2]).__name__}")
        elif len(edge) == 4:
            # Tuple["VertexType", "VertexType", Weight, AttributesDict]
            if isinstance(edge[2], numbers.Number):
                edge_data._weight = float(edge[2])
            else:
                raise exception.VertizeeException("the third item in an edge 4-tuple must be an "
                    f" edge weight (number); found {type(edge[2]).__name__}")

            if isinstance(edge[3], collections.abc.Mapping):
                edge_data._attr = copy.deepcopy(edge[3])
            else:
                raise exception.VertizeeException("the fourth item in an edge 4-tuple must be an "
                    f"attribute dictionary; found {type(edge[3]).__name__}")
    else:
        raise exception.VertizeeException("an edge must be specified as a tuple or an object;"
            f" found {type(edge).__name__}")

    return edge_data


def parse_graph_primitive(graph_primitive: "GraphPrimitive") -> ParsedEdgeAndVertexData:
    """Parses a graph primitive (an edge or vertex).

    Args:
        graph_primitive: The graph primitive (edge or vertex) to parse.

    Returns:
        ParsedEdgeAndVertexData: Returns the parsed graph primitive.
    """
    parsed_primitive = ParsedEdgeAndVertexData()
    if vertex_module.is_vertex_type(graph_primitive):
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
        raise exception.VertizeeException("graph_primitives must be iterable")

    return parsed_primitives


def parse_vertex_type(vertex_type: "VertexType") -> VertexData:
    """Parses a ``VertexType``, which is defined as
    ``Union[VertexLabel, VertexTupleAttr, Vertex]`` and where ``VertexTupleAttr`` is defined as
    ``Tuple[VertexLabel, AttributesDict]``. A vertex labels may be a string or integer.

    Args:
        vertex_type: The vertex to parse.

    Returns:
        VertexData: Returns the parsed vertex and saves the object in ``self.vertices``.
    """
    if isinstance(vertex_type, VertexBase):
        vertex_data = VertexData(vertex_type.label)
        vertex_data._vertex_object = vertex_type
        if vertex_type._attr:
            vertex_data._attr = copy.deepcopy(vertex_type.attr)
    elif isinstance(vertex_type, tuple):
        if len(vertex_type) != 2:
            raise exception.VertizeeException("a vertex specified as a tuple must be a 2-tuple of "
                "the form Tuple[VertexLabel, AttributesDict]")
        if not isinstance(vertex_type[0], int) and not isinstance(vertex_type[0], str):
            raise exception.VertizeeException("a vertex label must be a string or integer; "
                f"found {type(vertex_type[0]).__name__}")
        if not isinstance(vertex_type[1], collections.abc.Mapping):
            raise exception.VertizeeException("a vertex attr dictionary must be a mapping "
                f"(usually a dict); found {type(vertex_type[1]).__name__}")
        vertex_data = VertexData(str(vertex_type[0]))
        vertex_data._attr = copy.deepcopy(vertex_type[1])
    else:
        if not isinstance(vertex_type, (str, int)):
            raise exception.VertizeeException("a vertex label must be a string or integer; found "
                f"{type(vertex_type).__name__}")
        vertex_data = VertexData(str(vertex_type))

    return vertex_data
