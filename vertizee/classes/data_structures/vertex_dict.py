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
from collections import abc
import copy
from typing import Dict, Iterable, Iterator, MutableMapping, TYPE_CHECKING, TypeVar

from vertizee.classes.vertex import Vertex

if TYPE_CHECKING:
    from vertizee.classes.vertex import VertexType

#:Type variable for values in a generic VertexDict.
VT = TypeVar("VT")


class VertexDict(MutableMapping["VertexType", VT]):
    """A dictionary mapping vertices to values.

    The dictionary keys are of type :mod:`VertexType <vertizee.classes.vertex>`, which is an
    alias for ``Union[VertexClass, VertexLabel, VertexTupleAttr]``. This means that vertex keys may
    be specified as integers, strings, vertex tuples, or vertex objects. Internally, all vertex keys
    are coverted to ``str``. The dictionary values may be of any type, and can be explicitly
    specified using type hints as shown below.

    .. code-block:: python

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
    __slots__ = ("data",)

    def __init__(self, dictionary=None, /, **kwargs) -> None:
        self.data: Dict[str, VT] = {}
        if dictionary is not None:
            self.update(dictionary)
        if kwargs:
            self.update(kwargs)

    # Intentionally omitted `vertex` type-hint due to incompatibility with supertype "Mapping".
    def __contains__(self, vertex) -> bool:
        key = _normalize_vertex(vertex)
        return key in self.data

    def __delitem__(self, vertex: "VertexType") -> None:
        key = _normalize_vertex(vertex)
        del self.data[key]

    def __getitem__(self, vertex: "VertexType") -> VT:
        """Supports index accessor notation to retrieve items.

        Example:
            x.__getitem__(y) :math:`\\Longleftrightarrow` x[y]
        """
        key = _normalize_vertex(vertex)
        if key in self.data:
            return self.data[key]
        raise KeyError(vertex)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator["VertexType"]:
        return iter(self.data)

    def __setitem__(self, vertex: "VertexType", value: VT) -> None:
        key = _normalize_vertex(vertex)
        self.data[key] = value

    # Now, add the methods in dicts but not in MutableMapping
    def __repr__(self) -> str:
        return repr(self.data)

    def __copy__(self) -> "VertexDict":
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        # Create a copy and avoid triggering descriptors
        inst.__dict__["data"] = self.__dict__["data"].copy()
        return inst

    def copy(self) -> "VertexDict":
        """Make a copy of this VertexDict."""
        if self.__class__ is VertexDict:
            return VertexDict(self.data.copy())

        data = self.data
        try:
            self.data = {}
            c = copy.copy(self)
        finally:
            self.data = data
        c.update(self)
        return c

    @classmethod
    def fromkeys(cls, iterable: Iterable["VertexType"], value: VT) -> "VertexDict":
        """Create a new dictionary with keys from ``iterable`` and values set to ``value``.

        Args:
            iterable: A collection of vertices.
            value: The default value. All of the values refer to just a single instance,
                so it generally does not make sense for ``value`` to be a mutable object such as an
                empty list. To get distinct values, use a dict comprehension instead.

        Returns:
            VertexDict: A new VertexDict.
        """
        d = cls()
        for vertex in iterable:
            key = _normalize_vertex(vertex)
            d[key] = value
        return d

    def update(self, other=(), /, **kwds) -> None:
        """Updates the dictionary from an iterable or mapping object.

        Args:
            other: An iterable or mapping object over key-value pairs, where the keys
                represent vertices.
            **kwds: Keyword arguments.
        """
        if isinstance(other, abc.Mapping):
            for vertex in other:
                self.data[_normalize_vertex(vertex)] = other[vertex]
        elif hasattr(other, "keys"):
            for vertex in other.keys():
                self.data[_normalize_vertex(vertex)] = other[vertex]
        else:
            for vertex, value in other:
                self.data[_normalize_vertex(vertex)] = value
        for vertex, value in kwds.items():
            self.data[_normalize_vertex(vertex)] = value


def _normalize_vertex(vertex: "VertexType") -> str:
    if isinstance(vertex, Vertex):
        return vertex.label
    return str(vertex)
