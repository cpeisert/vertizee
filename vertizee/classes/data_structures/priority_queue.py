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
Priority Queue
==============

:term:`Priority-queue <priority queue>` data structure that returns the lowest (or highest)
priority item, where the priority is defined by a priority function.
"""

import collections.abc
import heapq
import itertools
from typing import Callable, Dict, Final, Generic, List, TypeVar, Union

ITEM_REMOVED: Final[str] = "__priority_queue_item_removed__"

T = TypeVar("T", bound=collections.abc.Hashable)


class _PriorityQueueItem(Generic[T]):
    """Generic wrapper for items stored in a :term:`priority queue`."""

    __slots__ = ("priority", "insertion_count", "item")

    def __init__(self, priority: Union[float, int], insertion_count: int, item: T) -> None:
        self.priority = priority
        self.insertion_count = insertion_count
        self.item: T = item

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PriorityQueueItem):
            return False
        return self.priority == other.priority and self.insertion_count == other.insertion_count

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _PriorityQueueItem):
            return False
        if self.priority < other.priority:
            return True
        return self.priority == other.priority and self.insertion_count < other.insertion_count

    def __gt__(self, other: object) -> bool:
        return (not self < other) and self != other

    def __le__(self, other: object) -> bool:
        return self < other or self == other

    def __ge__(self, other: object) -> bool:
        return not self < other

    def __hash__(self) -> int:
        return hash((self.priority, self.insertion_count))


class PriorityQueue(Generic[T]):
    """A :term:`priority queue` that always serves the item with the lowest (or highest) priority
    based on the priority returned by a priority function.

    ``PriorityQueue`` may be initialized as a minimum (default) or maximum priority queue.

    This class has a generic type parameter ``T``, which supports the type-hint usage
    ``PriorityQueue[T]``.

    ``T = TypeVar("T", bound=collections.abc.Hashable)``

    The priority function accepts an item of type ``T`` and returns a numeric priority (int or
    float). If two items have the same priority, they are served in the order inserted (first in
    first out).

    This implementation uses the Python standard library ``heapq``, which is a list-based binary
    :term:`heap`.

    Note:
        Items stored in a priority queue must be hashable.

    Args:
        priority_function: The item priority function.
        minimum: Optional; If True, priority queue is a minimum priority queue, otherwise a maximum
            priority queue. Defaults to True.

    Example:
        >>> PRIORITY = 'priority_key'
        >>> def priority_function(vertex: Vertex) -> int:
        >>>     return vertex.attr[PRIORITY]
        >>> g = Graph([(1, 2), (2, 3)])
        >>> g[1].attr[PRIORITY] = 100
        >>> g[2].attr[PRIORITY] = 90
        >>> g[3].attr[PRIORITY] = 80
        >>> pq: PriorityQueue[Vertex] = PriorityQueue(priority_function)
        >>> pq.add_or_update(g[2])
        >>> pq.add_or_update(g[1])
        >>> pq.add_or_update(g[3])
        >>> pq.pop()
        3
        >>> pq.pop()
        2
        >>> pq.pop()
        1
    """

    __slots__ = (
        "_heap_item_finder",
        "_insertion_counter",
        "_length",
        "_priority_function",
        "_priority_queue",
    )

    def __init__(
        self, priority_function: Callable[[T], Union[float, int]], minimum: bool = True
    ) -> None:
        if minimum:
            self._priority_function: Callable[[T], Union[float, int]] = priority_function
        else:

            def max_priority_func(item: T) -> float:
                return priority_function(item) * -1

            self._priority_function = max_priority_func
        self._priority_queue: List[_PriorityQueueItem[T]] = []

        self._heap_item_finder: Dict[T, _PriorityQueueItem[T]] = {}
        self._insertion_counter = itertools.count()
        self._length = 0

    def __len__(self) -> int:
        # return sum(1 for x in self._priority_queue if x.item is not ITEM_REMOVED)
        return self._length

    def add_or_update(self, item: T) -> None:
        """Adds a new item or updates an existing item with a new priority.

        Args:
            item: The item to add or update.
        """
        # If already in heap, mark as removed and re-add to maintain heap structure invariants.
        if item in self._heap_item_finder:
            self._mark_item_removed(item)
            self._length = self._length - 1

        self._length = self._length + 1
        insertion_count = next(self._insertion_counter)
        priority = self._priority_function(item)
        queue_item = _PriorityQueueItem(
            priority=priority, insertion_count=insertion_count, item=item
        )
        self._heap_item_finder[item] = queue_item
        heapq.heappush(self._priority_queue, queue_item)

    def pop(self) -> T:
        """Removes and returns the lowest (or highest) priority item. Raises ``KeyError`` if
        the priority queue is empty."""
        while self._priority_queue:
            item: _PriorityQueueItem[T] = heapq.heappop(self._priority_queue)
            if item.item is not ITEM_REMOVED:
                self._length = self._length - 1
                self._heap_item_finder.pop(item.item)
                return item.item
        raise KeyError("pop from an empty priority queue")

    def _mark_item_removed(self, item: T) -> None:
        """Mark an existing item as removed."""
        queue_item: _PriorityQueueItem[T] = self._heap_item_finder.pop(item)
        queue_item.item = ITEM_REMOVED  # type: ignore
