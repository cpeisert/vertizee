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

"""Union-find data structure (a.k.a. disjoint-set data structure) for maintaining a collection of
disjoint, dynamic sets."""

# pytype: disable=not-supported-yet

from collections import defaultdict
from typing import Dict, Generic, Iterator, Set, TypeVar

T = TypeVar("T")


class UnionFind(Generic[T]):
    """Union-find data structure (a.k.a. disjoint-set data structure) for maintaining a collection
    of disjoint, dynamic sets.

    The dynamic sets are comprised of objects of generic type 'T'.
    IMPORTANT: The objects stored in UnionFind must be hashable.

    Traditional operations:
        * find_set(x) - returns the representative item of the set containing x.
            Implemented as `__getitem__()` to enable index accessor notation.
        * make_set(x) - creates a new set containing x.
        * union(x, y) - unites the dynamic sets that contain elements x and y.

    Bonus operations:
        * __iter__() - returns an iterator over all items in the data structure.
        * __len__() - returns the number of items in the data structure.
        * in_same_set(x, y) - returns True if x and y are elements of the same set.
        * set_count - returns the number of sets
        * to_sets() - returns an iterator over the sets contained in the data structure.

    This implementation is based on the "disjoint-set forest" presented by Cormen, Leiserson,
    Rivest, and Stein [1] as well as the NetworkX [2] UnionFind implementation, which was in turn
    based on work by Josiah Carlson [3] and D. Eppstein [4].

    Args:
        *args (T, optional): Items to initialize as disjoint sets. Each item is added to its own
            set.

    Example::

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

    References:
        [1] Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein.
            Introduction to Algorithms: Third Edition, pages 568-571. The MIT Press, 2009.
        [2] NetworkX Python package: networkx.utils.union_find.py
            https://github.com/networkx/networkx/blob/master/networkx/utils/union_find.py
        [3] Carlson, Josiah. http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
        [4] Eppstein, D. http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py
    """

    def __init__(self, *args: T):
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

        for arg in args:
            self.make_set(arg)

    def __getitem__(self, item: T) -> T:
        """Returns the representative item of the set containing item. The representative item may
        change after a `union` operation.

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

    def in_same_set(self, item1: T, item2: T) -> bool:
        """Returns True if the items are elements of the same set."""
        return self[item1] == self[item2]

    def make_set(self, item: T):
        """Creates a new set containing the item."""
        self._set_count += 1
        self._parents[item] = item
        self._ranks[item] = 0

    @property
    def set_count(self) -> int:
        return self._set_count

    def to_sets(self) -> Iterator[Set[T]]:
        """Returns an iterator over all the sets contained in the data structure. Warning: This
        is the most computationally expensive operation of UnionFind."""
        # Compress all tree paths, so that every item's parent is the root of its tree.
        for item in self._parents.keys():
            _ = self[item]  # Evaluate for path-compression side-effect.

        dict_of_sets: Dict[T, Set[T]] = defaultdict(set)
        for k, root in self._parents.items():
            dict_of_sets[root].add(k)
        return iter(dict_of_sets.values())

    def union(self, item1: T, item2: T):
        """Unites the dynamic sets that contain elements item1 and item2."""
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
