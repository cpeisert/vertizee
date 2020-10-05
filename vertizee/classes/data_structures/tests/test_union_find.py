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

"""Tests for UnionFind data structure."""

import pytest

from vertizee.classes.data_structures.union_find import UnionFind

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip."
)


@pytest.mark.usefixtures()
class TestUnionFind:
    def test_init_make_set(self):
        uf: UnionFind[int] = UnionFind(1, 2, 3, 4, 5)

        assert len(uf) == 5, "Union-find should contain 5 items."
        assert uf.set_count == 5, "Union-find should contain 5 sets."

        uf.make_set(6)
        uf.make_set(7)
        uf.make_set(8)

        assert len(uf) == 8, "Union-find should contain 8 items."
        assert uf.set_count == 8, "Union-find should contain 8 sets."

    def test_find_set_and_union_and_iter(self):
        uf: UnionFind[int] = UnionFind(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

        item1 = None
        for item in uf:
            if item1 is None:
                item1 = item
            else:
                assert uf[item1] != uf[item], "Item 1 should not be contained in any other set."

        uf.union(1, 2)
        uf.union(3, 4)
        uf.union(5, 6)
        uf.union(7, 8)
        uf.union(9, 10)
        assert len(uf) == 15, "Union-find should contain 15 items."
        assert uf.set_count == 10, "Union-find should contain 10 sets."
        assert uf.in_same_set(1, 2), "Items 1 and 2 should be in the same set."
        assert uf[1] == uf[2], "Items 1 and 2 should be in the same set."
        uf.union(2, 4)
        uf.union(4, 6)
        uf.union(6, 8)
        uf.union(8, 10)
        assert len(uf) == 15, "Union-find should contain 15 items."
        assert uf.set_count == 6, "Union-find should contain 6 sets."
        assert uf[1] == uf[10], "Items 1 and 10 should be in the same set."
        assert uf.in_same_set(1, 10), "Items 1 and 10 should be in the same set."
        assert not uf.in_same_set(1, 15), "Items 1 and 15 should not be in the same set."

    def test_to_sets(self):
        uf: UnionFind[int] = UnionFind(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

        uf.union(1, 2)
        uf.union(3, 4)
        uf.union(5, 6)
        uf.union(7, 8)
        uf.union(9, 10)
        assert len(uf) == 15, "Union-find should contain 15 items."
        assert uf.set_count == 10, "Union-find should contain 10 sets."
        uf.union(2, 4)
        uf.union(4, 6)

        uf.union(12, 13)
        uf.union(12, 14)
        uf.union(12, 15)
        assert len(uf) == 15, "Union-find should contain 15 items."
        assert uf.set_count == 5, "Union-find should contain 5 sets."

        set_iter = uf.to_sets()

        item_count = 0
        set_count = 0
        for s in set_iter:
            item_count += len(s)
            set_count += 1
        assert item_count == len(uf), "Length of sets in UnionFind should equal len(UnionFind)."
        assert (
            set_count == uf.set_count
        ), "Number of sets in UnionFind should equal number of sets returned by to_sets()."
