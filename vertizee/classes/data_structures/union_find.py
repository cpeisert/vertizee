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
==========
Union Find
==========

:term:`Union-find <union find>` data structure (a.k.a. disjoint-set data structure) for
maintaining a collection of disjoint, dynamic sets.
"""

# pytype: disable=not-supported-yet
import collections.abc

from collections import defaultdict
from typing import Dict, Generic, Iterator, Optional, Set, TypeVar

T = TypeVar("T", bound=collections.abc.Hashable)


class UnionFind(Generic[T]):
    """:term:`Union-find <union find>` data structure (a.k.a. disjoint-set data structure) for
    maintaining a collection of disjoint, dynamic sets.

    This class has a generic type parameter ``T``, which supports the type-hint usage
    ``UnionFind[T]``.

    ``T = TypeVar("T", bound=collections.abc.Hashable)``

    The dynamic sets are comprised of objects of type :class:`T`.

    Note:
        The objects stored in UnionFind must be hashable.

    **Traditional operations:**

        * :func:`find_set` - Returns the representative item of the set containing the given item.
          Implemented as :meth:`__getitem__` to enable index accessor notation.
        * :meth:`make_set` - Creates a new set containing the given item.
        * :meth:`union` - Unites the dynamic sets that contain the given items.

    **Bonus operations:**

        * :meth:`__iter__` - Returns an iterator over all items in the data structure.
        * :meth:`__len__` - Returns the number of items in the data structure.
        * :meth:`get_set` - Returns the set containing ``item``.
        * :meth:`get_sets` - Returns the sets contained in the data structure.
        * :meth:`in_same_set` - Returns True if the given items are in the same set.
        * :attr:`set_count` - Returns the number of sets.

    Note:
        This implementation is based on the **disjoint-set forest** presented by Cormen, Leiserson,
        Rivest, and Stein :cite:`2009:clrs` as well as the NetworkX :cite:`2008:hss` ``UnionFind``
        implementation, which was in turn based on work by D. Eppstein. :cite:`2015:eppstein`

    Args:
        *args (Generic[T]): Optional; Items to initialize as disjoint sets. Each item is added to
            its own set.

    Example:
        >>> uf: UnionFind[int] = UnionFind(1, 2, 3, 4, 5)
        >>> len(uf)
        5
        >>> uf.set_count
        5
        >>> uf.make_set(6)
        >>> uf.make_set(7)
        >>> uf.make_set(8)
        >>> len(uf)
        8
        >>> uf.set_count
        8
        >>> uf.union(3, 4)
        >>> uf.union(4, 5)
        >>> len(uf)
        8
        >>> uf.set_count
        6
        >>> uf.in_same_set(3, 5)
        True
        >>> uf[3] == uf[5]
        True
        >>> uf.in_same_set(1, 3)
        False
        >>> uf[1] == uf[3]
        False
    """

    __slots__ = ("_parents", "_paths_compressed", "_ranks", "_set_count", "_sets_dict")

    def __init__(self, *args: T) -> None:
        self._parents: Dict[T, T] = dict()
        """A dictionary mapping an object to its parent. If the parent of an object is itself,
        then the the object is the root of a tree in the disjoint-set forest."""

        self._ranks: Dict[T, int] = dict()
        """A dictionary mapping an object, x, to its rank, which is an upper bound on its height
        (i.e. the longest simple path from a descendent leaf to x). The rank is only relevant for
        root nodes and is only updated during a union operation in which the two tree roots have
        equal rank, in which case the root node chosen as the new parent has its rank incremented.
        """

        self._set_count: int = 0

        # _paths_compressed and _sets_dict support the methods get_set() and get_sets().
        self._paths_compressed = False
        self._sets_dict: Optional[Dict[T, Set[T]]] = None

        for arg in args:
            self.make_set(arg)

    def __getitem__(self, item: T) -> T:
        """Returns the representative item of the set containing specified item. The representative
        item may change after a ``union`` operation.

        Args:
            item (T): The item whose set is to be found.

        Returns:
            T: The representative item of the set.
        """
        path = []
        root = item
        while self._parents[root] != root:
            path.append(root)
            root = self._parents[root]

        # Compress path such that for each item in the path, its parent is the root of the tree.
        for i in path:
            self._parents[i] = root
        return root

    def __iter__(self) -> Iterator[T]:
        """Iterates all items in this data structure."""
        return iter(self._parents)

    def __len__(self) -> int:
        return len(self._parents)

    def get_set(self, item: T) -> Set[T]:
        """Returns the set containing ``item``.

        Note:
            This is a computationally expensive operation that involves path compression of the
            entire UnionFind data structure. However, this price is paid the first time it is called
            and subsequent calls are relatively cheap, unless new sets are subsequently added or
            merged. To get the representative item of a set, use :meth:``__getitem__``.
        """
        self.get_sets()  # Called for side effects.
        assert self._sets_dict is not None
        return self._sets_dict[self[item]]

    def get_sets(self) -> Iterator[Set[T]]:
        """Returns the sets contained in the data structure.

        Note:
            This is a computationally expensive operation that involves path compression of the
            entire UnionFind data structure. However, this price is paid the first time it is called
            and subsequent calls are relatively cheap, unless new sets are subsequently added or
            merged.
        """
        # Compress all tree paths, so that every item's parent is the root of its tree.
        if not self._paths_compressed:
            for item in self._parents:
                _ = self[item]  # Evaluate for path-compression side-effect.
            self._paths_compressed = True

            self._sets_dict = defaultdict(set)
            for k, root in self._parents.items():
                self._sets_dict[root].add(k)

        assert self._sets_dict is not None
        return iter(self._sets_dict.values())

    def in_same_set(self, item1: T, item2: T) -> bool:
        """Returns True if the items are elements of the same set."""
        return self[item1] == self[item2]

    def make_set(self, item: T) -> None:
        """Creates a new set containing the item."""
        self._paths_compressed = False
        self._set_count += 1
        self._parents[item] = item
        self._ranks[item] = 0

    @property
    def set_count(self) -> int:
        """The number of sets. This value is decremented after calling ``union``
        on disjoint sets."""
        return self._set_count

    def union(self, item1: T, item2: T) -> None:
        """Unites the dynamic sets that contain ``item1`` and ``item2``."""
        self._paths_compressed = False
        x = self[item1]
        y = self[item2]
        if x != y:
            self._set_count -= 1
            if self._ranks[x] > self._ranks[y]:
                self._parents[y] = x
            else:
                self._parents[x] = y
                if self._ranks[x] == self._ranks[y]:
                    self._ranks[y] = self._ranks[y] + 1
