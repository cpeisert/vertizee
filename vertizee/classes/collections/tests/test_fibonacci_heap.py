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

"""Tests for Fibonacci heap data structure."""

from typing import Optional
import pytest

from vertizee.classes.collections.fibonacci_heap import FibonacciHeap

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip."
)


class Item:
    def __init__(self, name: str, priority: int):
        self.name = name
        self.priority = priority

    def __eq__(self, other: "Item"):
        return self.name == other.name and self.priority == other.priority

    def __hash__(self):
        return hash((self.name))


@pytest.mark.usefixtures()
class TestFibonacciHeap:
    def test_insert_and_length_and_minimum(self):
        fh: FibonacciHeap[Item] = FibonacciHeap(lambda i: i.priority)
        first = Item("first", 1)
        second = Item("second", 2)
        third = Item("third", 3)
        fourth = Item("fourth", 4)
        fifth = Item("fifth", 5)
        fh.insert(first)
        fh.insert(second)
        fh.insert(third)
        fh.insert(fourth)
        fh.insert(fifth)

        assert len(fh) == 5, "Fibonacci heap should contain 5 items."
        assert fh.minimum == first, "Min item should be `first`."

        zeroeth = Item("zeroeth", 0)
        fh.insert(zeroeth)
        assert len(fh) == 6, "Fibonacci heap should contain 5 items."
        assert fh.minimum == zeroeth, "Min item should be `zeroeth`."

    def test_union(self):
        fh: FibonacciHeap[Item] = FibonacciHeap(lambda i: i.priority)
        first = Item("first", 1)
        second = Item("second", 2)
        fh.insert(first)
        fh.insert(second)

        assert len(fh) == 2, "Fibonacci heap should contain 2 items."
        assert fh.minimum == first, "Min item should be `first`."

        neg_fh: FibonacciHeap[Item] = FibonacciHeap(lambda i: i.priority)
        neg_one = Item("neg_one", -1)
        neg_two = Item("neg_two", -2)
        neg_fh.insert(neg_two)
        neg_fh.insert(neg_one)

        fh.union(neg_fh)
        assert len(fh) == 4, "After Fibonacci heap union should contain 4 items."
        assert fh.minimum == neg_two, "Min item should be `neg_two`."

    def test_extract_min(self):
        fh: FibonacciHeap[Item] = FibonacciHeap(lambda i: i.priority)
        first = Item("first", 1)
        second = Item("second", 2)
        third = Item("third", 3)
        fourth = Item("fourth", 4)
        fifth = Item("fifth", 5)
        fh.insert(first)
        fh.insert(second)
        fh.insert(third)
        fh.insert(fourth)
        fh.insert(fifth)

        min_item: Optional[Item] = fh.extract_min()
        assert min_item == first, "Extracted min item should be `first`."
        assert fh.minimum == second, "Heap minimum should be `second` after extract_min()."
        assert len(fh) == 4, "Length of heap should be 4 after extract_min()."
        assert len(fh._roots) == 1, "Heap should have one tree after extract min."

        min_item = fh.extract_min()
        assert min_item == second, "Extracted min item should be `second`."
        assert fh.minimum == third, "Heap minimum should be `third` after extract_min()."
        assert len(fh) == 3, "Length of heap should be 3 after extract_min()."

        zeroeth = Item("zeroeth", 0)
        fh.insert(zeroeth)
        min_item = fh.extract_min()
        assert min_item == zeroeth, "Extracted min item should be `zeroeth`."
        assert fh.minimum == third, "Heap minimum should be `third` after extract_min()."
        assert len(fh) == 3, "Length of heap should be 3 after extract_min()."

        fh.extract_min()
        fh.extract_min()
        min_item = fh.extract_min()
        assert min_item == fifth, "Extracted min item should be `fifth`."
        assert fh.minimum is None, "Heap minimum should be `None` after extract_min()."
        assert len(fh) == 0, "Length of heap should be 0 after extract_min()."

        min_item = fh.extract_min()
        assert min_item is None, "Extracted min item from empty heap should be `None`."

    def test_update_item_with_decreased_priority(self):
        fh: FibonacciHeap[Item] = FibonacciHeap(lambda i: i.priority)
        first = Item("first", 1)
        second = Item("second", 2)
        third = Item("third", 3)
        fourth = Item("fourth", 4)
        fifth = Item("fifth", 5)
        fh.insert(first)
        fh.insert(second)
        fh.insert(third)
        fh.insert(fourth)
        fh.insert(fifth)

        assert fh.minimum == first, "Heap minimum should be `first`."
        second.priority = -2
        fh.update_item_with_decreased_priority(second)
        assert fh.minimum == second, "Heap minimum should be `second`."
        fh.extract_min()
        assert fh.minimum == first, "Heap minimum should be `first`."
        fifth.priority = -5
        fh.update_item_with_decreased_priority(fifth)
        assert fh.minimum == fifth, "Heap minimum should be `fifth`."
        fourth.priority = -16
        fh.update_item_with_decreased_priority(fourth)
        assert fh.minimum == fourth, "Heap minimum should be `fourth`."

    def test_delete(self):
        fh: FibonacciHeap[Item] = FibonacciHeap(lambda i: i.priority)
        first = Item("first", 1)
        second = Item("second", 2)
        third = Item("third", 3)
        fourth = Item("fourth", 4)
        fifth = Item("fifth", 5)
        fh.insert(first)
        fh.insert(second)
        fh.insert(third)
        fh.insert(fourth)
        fh.insert(fifth)

        assert len(fh) == 5, "Length of heap should be 5 after inserting 5 items."
        fh.delete(first)
        assert len(fh) == 4, "Length of heap should be 4 after deleting `first` item."
        fh.extract_min()
        fh.delete(fifth)
        assert len(fh) == 2, "Length of heap should be 2 after extract_min and delete."
        assert fh.minimum == third, "Heap minimum should be `third`."
        fh.delete(third)
        assert fh.minimum == fourth, "Heap minimum should be `fourth`."
