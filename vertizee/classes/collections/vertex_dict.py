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

"""VertexDict is a dictionary mapping VertexKeyValue (int, str, Vertex) keys to arbitrary generic
type values."""

from collections.abc import Iterable, Mapping
from typing import Dict, TypeVar

from vertizee.classes.vertex import Vertex, VertexKeyType

VT = TypeVar('VT')


class VertexDict(dict, Dict[VertexKeyType, VT]):
    """VertexDict is a dictionary mapping VertexKeyValue (int, str, Vertex) keys to arbitrary
    generic type values.

    The dictionary keys are stored internally as strings, but when getting and setting dictionary
    values, keys may be given as any one of int, str, or Vertex.

    Example:
        >>> v1 = Vertex(1)
        >>> v2 = Vertex('2')
        >>> v3 = Vertex(3)

        >>> d = VertexDict()
        >>> d['1'] = 'one'
        >>> d[2] = 'two'
        >>> d[v3] = 'three'

        >>> print(d[v1])
        'one'
        >>> print(d['2'])
        'two'
        >>> print(d[3])
        'three'
    """
    def __init__(self, iterable_or_mapping=None, **kwargs):
        if iterable_or_mapping is None:
            super().__init__(kwargs)
        else:
            parsed_arg = parse_iterable_or_mapping_arg(iterable_or_mapping)
            if len(kwargs) == 0:
                super().__init__(parsed_arg)
            else:
                super().__init__(kwargs.update(parsed_arg))

    def __contains__(self, key: VertexKeyType) -> bool:
        return super().__contains__(_normalize_key(key))

    def __getitem__(self, key: VertexKeyType):
        return super().__getitem__(_normalize_key(key))

    def __setitem__(self, key: VertexKeyType, val):
        super().__setitem__(_normalize_key(key), val)

    def __repr__(self):
        dictrepr = super().__repr__()
        return f'{type(self).__name__}({dictrepr})'

    def update(self, iterable_or_mapping):
        parsed_arg = parse_iterable_or_mapping_arg(iterable_or_mapping)
        super().update(parsed_arg)


def parse_iterable_or_mapping_arg(iterable_or_mapping) -> dict:
    parsed_arg = dict()

    if isinstance(iterable_or_mapping, Mapping):
        for k, v in iterable_or_mapping.items():
            parsed_arg[_normalize_key(k)] = v
    elif isinstance(iterable_or_mapping, Iterable):
        for i in iterable_or_mapping:
            if len(i) != 2:
                raise ValueError('Each item in the iterable must itself be an iterable '
                                 f'with exactly two objects. Found {len(i)} objects.')
            key = None
            for obj in i:
                if key is None:
                    key = obj
                else:
                    parsed_arg[_normalize_key(key)] = obj
    else:
        raise TypeError(f"{type(iterable_or_mapping).__name__} object is not iterable")
    return parsed_arg


def _normalize_key(key: VertexKeyType) -> str:
    if isinstance(key, Vertex):
        return key.key
    else:
        return str(key)
