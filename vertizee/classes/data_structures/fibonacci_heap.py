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
==============
Fibonacci Heap
==============

:term:`Fibonacci heap` data structure that serves the lowest priority item as defined by a
priority function.
"""

# pytype: disable=not-supported-yet

from __future__ import annotations
import collections.abc
import math
from typing import Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

NEG_INFINITY = float("-inf")

T = TypeVar("T", bound=collections.abc.Hashable)


class _FibonacciNode(Generic[T]):

    __slots__ = ("children", "item", "marked", "parent", "priority")

    def __init__(self, priority: Union[float, int], item: T) -> None:
        self.children: Set[_FibonacciNode[T]] = set()
        self.item: T = item
        self.marked: bool = False
        self.parent: Optional[_FibonacciNode[T]] = None
        self.priority = priority

    @property
    def degree(self) -> int:
        """Returns the degree of the node, which is the number of children it has."""
        return len(self.children)


class FibonacciHeap(Generic[T]):
    """A :term:`Fibonacci heap` data structure that obeys the min-heap-property: the priority of a
    node is greater than or equal to the priority of its parent, where priorities are defined by a
    priority function.

    This class has a generic type parameter ``T``, which supports the type-hint usage
    ``FibonacciHeap[T]``.

    ``T = TypeVar("T", bound=collections.abc.Hashable)``

    The priority function accepts an item of type ``T`` and returns a numeric priority (int or
    float). The default is the identity function (i.e. returns its argument), which is suitable
    for heaps of floats or integers.

    Note:
        Items stored in the heap must be hashable.

    The :term:`Fibonacci heap` asymptotic performance compares to a :term:`binary heap <heap>` as
    follows :cite:`2009:clrs`::

                    | Binary heap  | Fibonacci heap
      Procedure     | (worst-case) | (amortized)
      ---------------------------------------------
      MAKE-HEAP     | Θ(1)         | Θ(1)
      INSERT        | Θ(lg n)      | Θ(1)
      MINIMUM       | Θ(1)         | Θ(1)
      EXTRACT-MIN   | Θ(lg n)      | O(lg n)
      UNION         | Θ(n)         | Θ(1)
      DECREASE-KEY  | Θ(lg n)      | Θ(1)
      DELETE        | Θ(lg n)      | O(lg n)

    Note:
        This implementation is based on *Introduction to Algorithms: Third Edition*
        :cite:`2009:clrs` and the original paper "Fibonacci heaps and their uses in improved
        network optimization algorithms." :cite:`1987:fredman`

    Args:
        priority_function: The item/node priority function. If omitted, the identity function is
            used (i.e. returns its argument). If type ``T`` is not float or integer, then a priority
            function must be provided to ensure correct operation of the heap. Defaults to None.

    Example:
        >>> fh: FibonacciHeap[int] = FibonacciHeap()
        >>> for i in range(10): fh.insert(i)
        >>> fh.minimum
        0
        >>> len(fh)
        10
        >>> fh.extract_min()
        0
        >>> len(fh)
        9
        >>> fh.delete(7)
        >>> len(fh)
        8
        >>> fh2: FibonacciHeap[int] = FibonacciHeap()
        >>> for i in range(10, 20): fh2.insert(i)
        >>> len(fh2)
        10
        >>> fh.union(fh2)
        >>> len(fh)
        18
    """

    __slots__ = ("_item_to_node", "_length", "_min", "_priority_function", "_roots")

    def __init__(
        self, priority_function: Optional[Callable[[T], Union[float, int]]] = None
    ) -> None:
        self._item_to_node: Dict[T, _FibonacciNode[T]] = dict()
        """Maintain a mapping between items and their _FibonacciNode wrappers to facilitate
        efficient DECREASE-KEY operation (see :meth:`update_item_with_decreased_priority`)."""

        self._length = 0
        self._min: Optional[_FibonacciNode[T]] = None
        if priority_function:
            self._priority_function: Callable[[T], Union[float, int]] = priority_function
        else:
            self._priority_function = lambda x: x  # type: ignore

        self._roots: Set[_FibonacciNode[T]] = set()

    def __len__(self) -> int:
        return self._length

    def delete(self, item: T) -> None:
        """Deletes the specified item from the heap.

        Args:
            item: The item to delete.
        """
        self.update_item_with_decreased_priority(item, NEG_INFINITY)
        self.extract_min()

    def extract_min(self) -> Optional[T]:
        """Returns the minimum priority item from the heap and removes it."""
        z = self._min
        if z is not None:
            self._length -= 1
            self._roots.remove(z)
            for x in z.children:
                self._roots.add(x)
                x.parent = None
            if not self._roots:  # if z == z.right (see: Introduction to Algorithms [1])
                self._min = None
            else:
                self._consolidate()
            return z.item
        return None

    def insert(self, item: T) -> None:
        """Inserts a new item into the heap."""
        self._length += 1
        x = _FibonacciNode(self._priority_function(item), item)
        self._item_to_node[item] = x
        self._roots.add(x)
        if self._min is None:
            self._min = x
        elif x.priority < self._min.priority:
            self._min = x

    @property
    def minimum(self) -> Optional[T]:
        """Returns the minimum item from the heap, or ``None`` if the heap is empty."""
        if self._min is not None:
            return self._min.item
        return None

    def union(self, other: "FibonacciHeap[T]") -> None:
        """Merge ``other`` Fibonacci heap into this heap."""
        self._roots.update(other._roots)
        if self._min is None or (
            other._min is not None and other._min.priority < self._min.priority
        ):
            self._min = other._min
        self._length += other._length

    def update_item_with_decreased_priority(
        self, item: T, priority: Optional[float] = None
    ) -> None:
        """If the result of ``priority_function(item)`` is a lower priority than when ``item`` was
        first inserted into the heap, then this method must be called in order to update the item's
        position in the data structure.

        Note:
            Item priorities are only allowed to decrease.

        In the literature, this operation is called: **decrease key** or **FIB-HEAP-DECREASE-KEY**.

        Args:
            item: The item whose priority has decreased.
            priority: Optional; The new priority for the item. By default, the
                ``priority_function`` is used to get the new priority value. Defaults to None.
        """
        if priority:
            new_priority = priority
        else:
            new_priority = self._priority_function(item)
        x: _FibonacciNode[T] = self._item_to_node[item]
        if new_priority > x.priority:
            raise ValueError("new priority is greater than current priority")
        x.priority = new_priority
        y: Optional[_FibonacciNode[T]] = x.parent
        if y is not None and x.priority < y.priority:
            self._cut(x, y)
            self._cascading_cut(y)
        if self._min is None or x.priority < self._min.priority:
            self._min = x

    def _cascading_cut(self, y: _FibonacciNode[T]) -> None:
        """Move up the tree until finding either a root or an unmarked node."""
        while y.parent is not None:
            if not y.marked:
                y.marked = True
                break

            z = y.parent
            self._cut(y, z)
            y = z

    def _consolidate(self) -> None:
        """Consolidate the root list.

        Consolidation works as follows:

            * Find any two trees with roots of the same degree (i.e. the same number of
                children) and link them together. The new root has degree one greater than before.
            * Once there are no two trees with roots of the same degree, find the root with
                minimum key to serve as the minimum node of the modified heap.
        """
        if not self._roots:
            return
        max_degree = (int(math.log2(self._length)) + 2) * 2
        roots_indexed_by_degree: List[Optional[_FibonacciNode[T]]] = [
            None for x in range(max_degree)
        ]
        for w in self._roots.copy():
            x: _FibonacciNode[T] = w
            d = x.degree
            while roots_indexed_by_degree[d] is not None:
                y: Optional[_FibonacciNode[T]] = roots_indexed_by_degree[d]
                if y is not None and x.priority > y.priority:
                    x, y = y, x
                self._link(y, x)
                roots_indexed_by_degree[d] = None
                d = d + 1
            roots_indexed_by_degree[d] = x

        self._min = next(iter(self._roots))
        for node in self._roots:
            if node and node.priority < self._min.priority:
                self._min = node

    def _cut(self, x: _FibonacciNode[T], y: _FibonacciNode[T]) -> None:
        """Cut the link between x and its parent y, making x a root."""
        y.children.remove(x)
        self._roots.add(x)
        x.parent = None
        x.marked = False

    def _link(self, y: Optional[_FibonacciNode[T]], x: Optional[_FibonacciNode[T]]) -> None:
        if y is None or x is None:
            return
        self._roots.remove(y)
        x.children.add(y)
        y.parent = x
        y.marked = False
