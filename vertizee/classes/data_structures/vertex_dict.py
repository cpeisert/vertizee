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

"""A dictionary mapping vertices to values."""

from __future__ import annotations
from collections.abc import Iterable, Mapping
from typing import Dict, TYPE_CHECKING, TypeVar

from vertizee.classes.vertex import Vertex

if TYPE_CHECKING:
    from vertizee.classes.vertex import VertexType

#:Type variable for values in a generic VertexDict.
VT = TypeVar("VT")


class VertexDict(dict, Dict["VertexType", VT]):
    """A dictionary mapping vertices to values.

    The dictionary keys are of type :mod:`VertexType <vertizee.classes.vertex>`, which is an
    alias for ``int``, ``str``, or :class:`Vertex <vertizee.classes.vertex.Vertex>`. Internally,
    all dictionary vertex keys are coverted to ``str``. The dictionary values may be of any type,
    and can be explicitly specified using type hints, such as::

        vertex_to_path: VertexDict[ShortestPath] = VertexDict()

    Args:
        iterable_or_mapping: If a positional argument is given and it is a mapping object, a
            dictionary is created with the same key-value pairs as the mapping object. Otherwise,
            the positional argument must be an iterable object. Each item in the iterable must
            itself be an iterable with exactly two objects.
        **kwargs: Keyword arguments and their values to be added to the dictionary.

    Example:
        >>> import vertizee as vz
        >>> g = vz.Graph()
        >>> g.add_vertex(1)
        >>> g.add_vertex('2')
        >>> d: vz.VertexDict[str] = vz.VertexDict()
        >>> d[g[1]] = 'one'
        >>> d[2] = 'two'
        >>> d['3'] = 'three'
        >>> print(d['1'])
        one
        >>> print(d[g[2]])
        two
        >>> print(d[3])
        three
    """

    def __init__(self, iterable_or_mapping=None, **kwargs) -> None:
        if iterable_or_mapping is None:
            super().__init__(kwargs)
        else:
            parsed_arg = parse_iterable_or_mapping_arg(iterable_or_mapping)
            if len(kwargs) == 0:
                super().__init__(parsed_arg)
            else:
                super().__init__(kwargs.update(parsed_arg))

    def __contains__(self, vertex: "VertexType") -> bool:
        return super().__contains__(_normalize_vertex(vertex))

    def __getitem__(self, vertex: "VertexType") -> VT:
        """Supports index accessor notation to retrieve items.

        Example:
            x.__getitem__(y) :math:`\\Longleftrightarrow` x[y]
        """
        return super().__getitem__(_normalize_vertex(vertex))

    def __setitem__(self, vertex: "VertexType", val: VT) -> None:
        super().__setitem__(_normalize_vertex(vertex), val)

    def __repr__(self) -> str:
        dictrepr = super().__repr__()
        return f"{type(self).__name__}({dictrepr})"

    def update(self, iterable_or_mapping) -> None:
        """Updates the dictionary from an iterable or mapping object.

        Args:
            iterable_or_mapping: An iterable or mapping object over key-value pairs, where the keys
                represent vertices.
        """
        parsed_arg = parse_iterable_or_mapping_arg(iterable_or_mapping)
        super().update(parsed_arg)


def parse_iterable_or_mapping_arg(iterable_or_mapping) -> dict:
    """Helper method to parse initialization arguments given in the form of an iterable or mapping
    object.

    Args:
        iterable_or_mapping: An iterable or mapping object (e.g. a dictionary).

    Returns:
        dict: A dictionary containing the parsed arguments.
    """
    parsed_arg = dict()

    if isinstance(iterable_or_mapping, Mapping):
        for k, v in iterable_or_mapping.items():
            parsed_arg[_normalize_vertex(k)] = v
    elif isinstance(iterable_or_mapping, Iterable):
        for i in iterable_or_mapping:
            if len(i) != 2:
                raise ValueError(
                    "Each item in the iterable must itself be an iterable "
                    f"with exactly two objects. Found {len(i)} objects."
                )
            vertex = None
            for obj in i:
                if vertex is None:
                    vertex = obj
                else:
                    parsed_arg[_normalize_vertex(vertex)] = obj
    else:
        raise TypeError(f"{type(iterable_or_mapping).__name__} object is not iterable")
    return parsed_arg


def _normalize_vertex(vertex: "VertexType") -> str:
    if isinstance(vertex, Vertex):
        return vertex.label
    return str(vertex)
