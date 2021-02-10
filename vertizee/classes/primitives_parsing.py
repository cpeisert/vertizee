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
==================
Primitives parsing
==================

Provides functionality for parsing :term:`vertices <vertex>` and :term:`edges <edge>`.

Vertices may be specified using integer or string labels, tuples, or instances of classes derived
from :class:`VertexBase <vertizee.classes.vertex.VertexBase>`. Edges may be specified as tuples of
integers or strings as well as instances of classes derived from either :class:`Connection
<vertizee.classes.edge.Connection>` or :class:`MultiConnection
<vertizee.classes.edge.MultiConnection>`.

Class and type-alias summary
============================

* :mod:`EdgeType <vertizee.classes.edge>` - A type alias defined as ``Union[E, EdgeLiteral]``. See
  definition in :mod:`vertizee.classes.edge.EdgeType <vertizee.classes.edge>`.
* :mod:`VertexType <vertizee.classes.vertex>` - A type alias defined as
  ``Union[V, VertexLabel, VertexTuple]``. See
  definition in :mod:`vertizee.classes.vertex.VertexType <vertizee.classes.vertex>`.
* :class:`GraphPrimitive` - Type alias defined as ``Union["EdgeType", "VertexType"]``.
* :class:`VertexData` - Lightweight class to store parsed primitives representing a vertex.
* :class:`EdgeData` - Lightweight class to store parsed primitives representing an edge.
* :class:`ParsedEdgeAndVertexData` - Container class to hold lists of parsed edges and vertices
  stored as :class:`EdgeData` and/or :class:`VertexData` objects.

Function summary
================

* :func:`parse_edge_type` - Parses an ``EdgeType``. See definition above.
* :func:`parse_graph_primitive` - Parses a graph primitive (an ``EdgeType`` or ``VertexType``). See
  definitions above.
* :func:`parse_graph_primitives_from` - Parses an iterable or collection of graph primitives.
* :func:`parse_vertex_type` - Parses a ``VertexType``. See definition above.

Detailed documentation
======================
"""

from __future__ import annotations
import collections.abc
import copy
from dataclasses import dataclass, field
import numbers
from typing import Any, cast, Dict, Iterable, List, Optional, Union

from vertizee import exception
from vertizee.classes import edge as edge_module
from vertizee.classes import vertex as vertex_module
from vertizee.classes.edge import DiEdge, Edge, MutableEdgeBase, EdgeType
from vertizee.classes.vertex import VertexBase, VertexType

# Type aliases
GraphPrimitive = Union[EdgeType, VertexType]


class VertexData:
    """Lightweight class to store parsed primitives representing a vertex."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._attr: Dict[str, Any] = dict()
        self._vertex_object: Optional[VertexBase] = None

    @classmethod
    def from_vertex_obj(cls, vertex: VertexBase) -> VertexData:
        """Factory method to create a ``VertexData`` instance from a vertex object."""
        vertex_data: VertexData = VertexData(vertex.label)
        vertex_data._vertex_object = vertex
        return vertex_data

    @property
    def attr(self) -> Dict[str, Any]:
        """The attributes dictionary. If a vertex object was parsed that did not contain an
        attributes dictionary, an new empty ``dict`` is returned."""
        if self._vertex_object and self._vertex_object.has_attributes_dict():
            return self._vertex_object.attr
        return self._attr

    @property
    def vertex_object(self) -> Optional[VertexBase]:
        """If an vertex object was parsed, refers to original object, otherwise None."""
        return self._vertex_object


class EdgeData:
    """Lightweight class to store parsed primitives representing an edge."""

    def __init__(self, vertex1: "VertexData", vertex2: "VertexData") -> None:
        self._vertex1 = vertex1
        self._vertex2 = vertex2
        self._weight: float = edge_module.DEFAULT_WEIGHT
        self._attr: Dict[str, Any] = dict()
        self._edge_object: Optional[MutableEdgeBase[VertexBase]] = None

    @property
    def attr(self) -> Dict[str, Any]:
        """The attributes dictionary. If an edge object was parsed that did not contain an
        attributes dictionary, an new empty ``dict`` is returned."""
        if isinstance(self._edge_object, (Edge, DiEdge)):
            # cast below are due to MyPy bug https://github.com/python/mypy/issues/8252
            return cast(Edge, self._edge_object).attr
        return self._attr

    @property
    def edge_object(self) -> Optional[MutableEdgeBase[VertexBase]]:
        """If an edge object was parsed, refers to original object, otherwise None."""
        return self._edge_object

    @classmethod
    def from_edge_obj(cls, edge: MutableEdgeBase[VertexBase]) -> EdgeData:
        """Factory method to create an ``EdgeData`` instance from an edge object."""
        vertex1 = VertexData.from_vertex_obj(edge.vertex1)
        vertex2 = VertexData.from_vertex_obj(edge.vertex2)
        edge_data: EdgeData = EdgeData(vertex1, vertex2)
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
        """The first vertex (type :class:`VertexData`)."""
        return self._vertex1

    @property
    def vertex2(self) -> VertexData:
        """The second vertex (type :class:`VertexData`)."""
        return self._vertex2

    @property
    def weight(self) -> float:
        """The edge weight (type ``float``)."""
        if self._edge_object:
            return self._edge_object.weight
        return self._weight


@dataclass
class ParsedEdgeAndVertexData:
    """Container class to hold lists of parsed edges and vertices stored as :class:`EdgeData`
    and/or :class:`VertexData` objects."""

    edges: List["EdgeData"] = field(default_factory=list)
    vertices: List["VertexData"] = field(default_factory=list)


def parse_edge_type(edge: "EdgeType") -> "EdgeData":
    """Parses an ``EdgeType``, which is defined as ``Union[E, EdgeLiteral]``. See
    definition in :mod:`vertizee.classes.edge.EdgeType <vertizee.classes.edge>`.

    Args:
        edge: The edge to parse.

    Returns:
        EdgeData: The parsed edge.
    """
    if isinstance(edge, MutableEdgeBase):
        edge_data = EdgeData.from_edge_obj(cast(MutableEdgeBase[VertexBase], edge))
    elif isinstance(edge, tuple):
        if len(edge) < 2 or len(edge) > 4:
            raise exception.VertizeeException(
                "an edge tuple must contain 2, 3, or 4 items, found "
                f"tuple of length {len(edge)}; see 'EdgeType' type alias for more information"
            )
        vertex1 = parse_vertex_type(edge[0])
        vertex2 = parse_vertex_type(edge[1])
        edge_data = EdgeData(vertex1, vertex2)

        if len(edge) == 3:
            # Tuple["VertexType", "VertexType", Weight]
            # Tuple["VertexType", "VertexType", AttributesDict]
            if isinstance(edge[2], collections.abc.Mapping):  # type: ignore
                edge_data._attr = copy.deepcopy(edge[2])  # type: ignore
            elif isinstance(edge[2], numbers.Number):  # type: ignore
                edge_data._weight = float(edge[2])  # type: ignore
            else:
                raise TypeError(
                    "the third item in an edge 3-tuple must be an "  # type: ignore
                    "attribute dictionary or an edge weight (float); "
                    f"{type(edge[2]).__name__} found"
                )
        elif len(edge) == 4:
            # Tuple["VertexType", "VertexType", Weight, AttributesDict]
            if isinstance(edge[2], numbers.Number):  # type: ignore
                edge_data._weight = float(edge[2])  # type: ignore
            else:
                raise TypeError(
                    "the third item in an edge 4-tuple must be an "  # type: ignore
                    f" edge weight (float); {type(edge[2]).__name__} found"
                )

            if isinstance(edge[3], collections.abc.Mapping):  # type: ignore
                edge_data._attr = cast(dict, copy.deepcopy(edge[3]))  # type: ignore
            else:
                raise TypeError(
                    "the fourth item in an edge 4-tuple must be an "  # type: ignore
                    f"attribute dictionary; {type(edge[3]).__name__} found"
                )
    else:
        raise TypeError("expected tuple edge object instance;" f" found {type(edge).__name__}")

    return edge_data


def parse_graph_primitive(graph_primitive: "GraphPrimitive") -> "ParsedEdgeAndVertexData":
    """Parses a graph primitive (an ``EdgeType`` or ``VertexType``).

    Args:
        graph_primitive: The graph primitive (edge or vertex) to parse.

    Returns:
        ParsedEdgeAndVertexData: The parsed graph primitives.
    """
    parsed_primitive = ParsedEdgeAndVertexData()
    if vertex_module.is_vertex_type(graph_primitive):
        vertex_data = parse_vertex_type(cast(VertexType, graph_primitive))
        parsed_primitive.vertices.append(vertex_data)
    else:
        edge_data = parse_edge_type(cast(EdgeType, graph_primitive))
        parsed_primitive.edges.append(edge_data)
    return parsed_primitive


def parse_graph_primitives_from(
    graph_primitives: Iterable["GraphPrimitive"],
) -> "ParsedEdgeAndVertexData":
    """Parses an iterable or collection of graph primitives, where the primitives are ``EdgeType``
    and/or ``VertexType``.

    Args:
        graph_primitives: The graph primitives to parse.

    Returns:
        ParsedEdgeAndVertexData: The parsed graph primitives.
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


def parse_vertex_type(vertex_type: "VertexType") -> "VertexData":
    """Parses a ``VertexType``, which is defined as ``Union[V, VertexLabel, VertexTuple]``. See
    definition in :mod:`vertizee.classes.vertex.VertexType <vertizee.classes.vertex>`.

    Args:
        vertex_type: The vertex to parse.

    Returns:
        VertexData: The parsed vertex.
    """
    if isinstance(vertex_type, VertexBase):
        vertex_data: VertexData = VertexData.from_vertex_obj(vertex_type)
        if vertex_type._attr:
            vertex_data._attr = copy.deepcopy(vertex_type.attr)
    elif isinstance(vertex_type, tuple):
        if len(vertex_type) != 2:
            raise TypeError(
                "vertex tuple expected length 2 of the form Tuple[VertexLabel, AttributesDict];"
                f" tuple of length {len(vertex_type)} found"
            )
        if not isinstance(vertex_type[0], int) and not isinstance(vertex_type[0], str):
            raise TypeError(
                "vertex label expected (str or int instance); "
                f"{type(vertex_type[0]).__name__} found"
            )
        if not isinstance(vertex_type[1], collections.abc.Mapping):
            raise TypeError(
                "vertex attr dictionary expected Mapping instance; "
                f"{type(vertex_type[1]).__name__} found"
            )
        vertex_data = VertexData(str(vertex_type[0]))
        vertex_data._attr = copy.deepcopy(vertex_type[1])
    else:
        if not isinstance(vertex_type, (str, int)):
            raise TypeError(
                "vertex label expected (str or int instance); "
                f"{type(vertex_type).__name__} found"
            )
        vertex_data = VertexData(str(vertex_type))

    return vertex_data
