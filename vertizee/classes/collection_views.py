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
================
Collection views
================

Collection view classes providing dynamic views of underlying collections.

Collection views are wrappers around standard collections such as dictionaries, lists, and sets.
The views provide APIs similar to the underlying data structures, but only implement methods that
do not alter the state of the underlying collection. That is, views do not provide methods for
adding, removing, or changing the values of items. However, if the items in a collection are
mutable objects, then the objects themselves may be modified.

For example, a :class:`ListView` does not allow a different element to be assigned to a specific
index in the underlying ``list``. However, if the ``list`` is comprised of mutable objects, such as
:term:`vertices <vertex>`, then the vertex ``attr`` dictionaries may be modified by adding,
removing, or changing custom attributes.

Collection views provide the following advantages:

* **Iterator**. This is especially important when working with large data structures. For example,
  if not all :term:`vertices <vertex>` or :term:`edges <edge>` are required for a particular
  application, iterators allow the consumer to only use those items that are needed rather than
  traversing all items in the collection.
* **Dynamic**. Views enable changes in an underlying collection to be reflected immediately in the
  view, while still protecting the collection from direct write access.
* **Collection-like operations**. Views provide collection-like operations, such as *contains* and
  *length*. These operations are provided without requiring a copy to be made of the underlying
  collection.

The collection view classes are modelled after the Python standard library ``MappingView`` class
(and its descendants) found in the module ``collections.abc``.

Class summary
=============

* :class:`CollectionView[T]] <CollectionView>` - Generic, abstract base class defining a dynamic,
  immutable view of a collection.
* :class:`ItemsView[KT, VT] <ItemsView>` - Generic class defining a dynamic, immutable, set-like
  view of items in a dictionary.
* :class:`ListView[T] <ListView>` - Generic class defining a dynamic, immutable, list-like view
  of a collection. All of the nonmutating list operations are supported.
* :class:`SetView[T] <SetView>` - Generic class defining a dynamic, immutable, set-like view of a
  collection. All of the nonmutating set operations are supported.

Detailed documentation
======================
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import collections.abc
from typing import (
    cast,
    Collection,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    overload,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union
)

from vertizee.classes.comparable import Comparable

T = TypeVar("T", bound=Comparable, covariant=True)  # Any type.
KT = TypeVar("KT", covariant=True)  # Key type.
VT = TypeVar("VT", covariant=True)  # Value type.


class CollectionView(ABC, collections.abc.Collection[T], Comparable, Generic[T]):
    """Generic, abstract base class defining a dynamic, immutable view of a collection."""

    __slots__ = ("_collection",)

    def __init__(self, collection: Collection[T]) -> None:
        self._collection = collection
        super().__init__()

    def __contains__(self, x: object) -> bool:
        """Returns True if the collection contains ``x``, otherwise False."""
        return x in self._collection

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Returns True if ``self`` is equal to ``other``."""

    @abstractmethod
    def __lt__(self, other: object) -> bool:
        """Returns True if ``self`` is less than ``other``."""

    def __iter__(self) -> Iterator[T]:
        """Yields the elements in the collection."""
        yield from self._collection

    def __len__(self) -> int:
        """Returns the number of elements in the collection."""
        return len(self._collection)

    def __repr__(self) -> str:
        """An unambiguous string representation of the collection."""
        return f"{self.__class__.__name__}({repr(self._collection)})"

    def __str__(self) -> str:
        """A string representation of the collection."""
        return self.__repr__()

    @classmethod
    def _from_iterable(cls: Type[CollectionView[T]], it: Iterator[T]) -> CollectionView[T]:
        """Construct an instance of the class from any iterable input."""
        return cls(list(it))


class ItemsView(collections.abc.Set[VT], Generic[KT, VT]):
    """Generic class defining a dynamic, immutable, set-like view of items in a dictionary."""

    __slots__ = ("_mapping",)

    def __init__(self, dictionary: Dict[KT, VT]) -> None:
        self._mapping = dictionary

    def __contains__(self, item: object) -> bool:
        """Returns True if the dictionary contains a key-value pair matching ``item``."""
        if not isinstance(item, tuple):
            raise TypeError(f"expected tuple; {type(item).__name__} found")
        key, value = item
        try:
            v = self._mapping[key]
        except KeyError:
            return False
        else:
            return v is value or v == value

    def __iter__(self) -> Iterator[Tuple[KT, VT]]:  # type: ignore
        """Yields key-value pair tuples from the dictionary."""
        for key in self._mapping:
            yield (key, self._mapping[key])

    def __len__(self) -> int:
        """Returns the number of items in the dictionary."""
        return len(self._mapping)

    def __repr__(self) -> str:
        """An unambiguous string representation of the items view."""
        return f"{self.__class__.__name__}({repr(self._mapping)})"

    @classmethod
    def _from_iterable(
        cls: Type[ItemsView[KT, VT]], it: Iterator[Tuple[KT, VT]]
    ) -> ItemsView[KT, VT]:
        return cls(dict(it))


class ListView(CollectionView[T], Generic[T]):
    """Generic class defining a dynamic, immutable, list-like view of a collection. All of the
    nonmutating list operations are supported."""

    __slots__ = ()

    def __init__(self, list_collection: List[T]) -> None:
        if not isinstance(list_collection, list):
            raise TypeError(f"expected list instance; {type(list_collection).__name__} found")
        super().__init__(collection=list_collection)

    @overload
    def __getitem__(self, key: int) -> T:
        ...

    @overload
    def __getitem__(self, key: slice) -> ListView[T]:
        ...

    def __getitem__(self, key: Union[int, slice]) -> Union[T, ListView[T]]:
        """Supports index accessor notation to retrieve items by index or by using a slice."""
        if isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"index {key} is out of range")
            return cast(List[T], self._collection)[key]
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return ListView([self[i] for i in range(start, stop, step)])
        raise TypeError(f"expected int or slice instance; {type(key).__name__} found")

    def __len__(self) -> int:
        """Returns the number of items in the list."""
        return len(self._collection)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, list):
            other_list = cast(List[T], other)
        elif isinstance(other, ListView):
            other_list = cast(List[T], other._collection)
        else:
            return NotImplemented
        return self._collection == other_list

    def __lt__(self, other: object) -> bool:
        if isinstance(other, list):
            other_list = cast(List[T], other)
        elif isinstance(other, ListView):
            other_list = cast(List[T], other._collection)
        else:
            return NotImplemented
        return cast(List[T], self._collection) <  other_list

    @classmethod
    def _from_iterable(cls: Type[ListView[T]], it: Iterator[T]) -> ListView[T]:
        """Construct an instance of the class from any iterable input."""
        return cls(list(it))

    def count(self, x: object) -> int:
        """Returns the number of times ``x`` appears in the list."""
        return cast(List[object], self._collection).count(x)

    def index(self, x: object, start: Optional[int] = None, end: Optional[int] = None) -> int:
        """Returns zero-based index in the list of the first item whose value is equal to ``x``.
        Raises a ``ValueError`` if there is no such item.

        The optional arguments ``start`` and ``end`` are interpreted as in the slice notation and
        are used to limit the search to a particular subsequence of the list. The returned index is
        computed relative to the beginning of the full sequence rather than the start argument."""
        if not start:
            start = 0
        if not end:
            end = len(self._collection)
        return cast(List[object], self._collection).index(x, start, end)


class SetView(collections.abc.Set[T], CollectionView[T], Generic[T]):
    """Generic class defining a dynamic, immutable, set-like view of a collection. All of the
    nonmutating set operations are supported."""

    __slots__ = ()

    def __init__(self, set_collection: Set[T]) -> None:
        super().__init__(collection=set_collection)

    def __contains__(self, x: object) -> bool:
        """Returns True if the set contains ``x``, otherwise False."""
        return x in self._collection

    def __eq__(self, other: object) -> bool:
        return self._collection.__eq__(other)

    def __lt__(self, other: object) -> bool:
        if not isinstance(self._collection, set):
            return False
        return self._collection.__lt__(cast(Set[object], other))

    def __gt__(self, other: object) -> bool:
        return (not self < other) and self != other

    def __le__(self, other: object) -> bool:
        return self < other or self == other

    def __ge__(self, other: object) -> bool:
        return not self < other

    def __iter__(self) -> Iterator[T]:
        """Yields the elements in the collection."""
        yield from self._collection

    def __len__(self) -> int:
        """Returns the number of items in the set."""
        return len(self._collection)

    @classmethod
    def _from_iterable(cls: Type[SetView[T]], it: Iterator[T]) -> SetView[T]:
        return cls(set(it))
