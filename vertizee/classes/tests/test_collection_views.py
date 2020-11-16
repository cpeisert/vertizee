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

"""Tests for views on collections."""
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

from vertizee.classes.collection_views import ItemsView, ListView, SetView


class TestCollectionViews:
    """Tests for functions defined in the collection_views module."""

    def test_items_view(self):
        d = {1: "one", 2: "two", 3: "three"}
        items_view = ItemsView(d)
        assert (1, "one") in items_view
        assert (2, "three") not in items_view

        keys = set()
        values = list()
        for k, v in items_view:
            keys.add(k)
            values.append(v)
        assert len(keys) == 3
        assert 1 in keys
        assert len(values) == 3
        assert "one" in values
        assert repr(items_view) == "ItemsView({1: 'one', 2: 'two', 3: 'three'})"
        assert str(items_view) == repr(items_view)

        items_view2 = ItemsView._from_iterable([("five", 5), ("six", 6)])
        assert len(items_view2) == 2
        assert ("five", 5) in items_view2
        assert ("five", 4) not in items_view2

    def test_list_view(self):
        list1 = [5, 6, 7, 5]
        list_view = ListView(list1)

        assert len(list_view) == 4
        assert 7 in list_view

        list_copy = list()
        for x in list_view:
            list_copy.append(x)
        assert list_view == list_copy
        assert list_view < [8, 9, 10, 11]
        assert list_view > [0, 1, 2, 3]
        assert list_view < [8, 9]
        assert list_view > []
        assert list_view < [5, 6, 7, 5, 0]
        assert list_view > [5, 6, 7]
        assert list_view[0] == 5
        assert list_view[2] == 7
        assert list_view.count(5) == 2
        assert list_view.index(6) == 1
        assert list_view.index(5, start=2, end=4) == 3
        assert repr(list_view) == "ListView([5, 6, 7, 5])"
        assert str(list_view) == repr(list_view)

        list_view2 = ListView._from_iterable({10, 11, 12})
        assert len(list_view2) == 3
        assert isinstance(list_view2, ListView)
        assert list_view < list_view2
        assert list_view2 == ListView._from_iterable([10, 11, 12])
        assert bool(list_view2), "Non-empty list view should evaluate to True"
        assert not bool(ListView._from_iterable([])), "Empty list view should evaluate to False"

    def test_set_view(self):
        set1 = {10, 11, 12}
        set_view = SetView(set1)

        assert len(set_view) == 3
        assert 10 in set_view
        assert 13 not in set_view
        assert set_view & {9, 10, 11} == {10, 11}
        assert set_view | {12, 13, 14} == {10, 11, 12, 13, 14}
        assert repr(set_view) == "SetView({10, 11, 12})"
        assert str(set_view) == repr(set_view)

        set_view2 = SetView._from_iterable(["a", "b", "c", "d", "d"])
        assert len(set_view2) == 4
        assert "d" in set_view2
        assert isinstance(set_view2, SetView)
