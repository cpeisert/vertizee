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

"""Tests for :term:`spanning arborescence` and :term:`branching` algorithms."""
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

import pytest

from vertizee import exception
from vertizee.algorithms.spanning import directed
from vertizee.classes.graph import DiGraph, Graph, MultiDiGraph


test_edges_acyclic = [
    ("a", "b", 9),
    ("a", "e", 2),
    ("b", "c", 6),
    ("c", "d", 3),
    ("c", "f", 1),
    ("c", "h", 10),
    ("d", "f", 8),
    ("e", "g", 7),
    ("f", "g", 5),
    ("g", "h", 4),
]

test_edges_cyclic = [
    ("a", "b", 2),
    ("b", "a", 6),
    ("b", "c", 4),
    ("b", "g", 5),
    ("c", "a", 3),
    ("c", "d", 5),
    ("d", "c", 3),
    ("d", "e", 1),
    ("e", "c", 2),
    ("e", "f", 3),
    ("f", "h", 2),
    ("g", "h", 3),
    ("h", "d", 6)
]

test_edges_multi_tree = [
    ("a", "b", 2),
    ("b", "c", 5),
    ("c", "d", 1),
    ("d", "a", 3),
    ("b", "e", -10),
    ("c", "f", 10),
    ("g", "c", -2),
    ("g", "h", -4),
    ("h", "i", -5),
    ("i", "j", -1),
    ("j", "g", -3)
]


class TestEdmonds:
    """Tests for Edmonds' algorithm to find spanning arborescenes and branchings."""

    def test_edmonds_empty_graph(self):
        g = DiGraph()

        # Edmonds' algorithm does not work on empty graphs.
        with pytest.raises(exception.Unfeasible):
            for _ in directed.edmonds(g):
                pass

    def test_edmonds_undirected_graph(self):
        g = Graph([("s", "t", 10), ("s", "y", 5), ("t", "y", 2)])

        # Edmonds' algorithm does not work on undirected graphs.
        with pytest.raises(exception.GraphTypeNotSupported):
            for _ in directed.edmonds(g):
                pass

    def test_edmonds_max_spanning_arborescence(self):
        g = DiGraph(test_edges_acyclic)
        spanning_edges_max_arborescence = {
            ("a", "e"),
            ("a", "b"),
            ("b", "c"),
            ("c", "h"),
            ("c", "d"),
            ("d", "f"),
            ("e", "g")
        }

        arborescence = next(directed.edmonds(g, minimum=False, find_spanning_arborescence=True))
        for edge_label in spanning_edges_max_arborescence:
            assert edge_label in arborescence

        assert arborescence.weight == 45, "max spanning arborescence weight should be 45"

    def test_edmonds_min_spanning_arborescence(self):
        g = DiGraph(test_edges_acyclic)
        spanning_edges_min_arborescence = {
            ("a", "e"),
            ("a", "b"),
            ("b", "c"),
            ("g", "h"),
            ("f", "g"),
            ("c", "f"),
            ("c", "d")
        }
        arborescence = next(directed.edmonds(g, minimum=True, find_spanning_arborescence=True))
        for edge_label in spanning_edges_min_arborescence:
            assert edge_label in arborescence

        assert arborescence.weight == 30, "min spanning arborescence weight should be 30"

    def test_edmonds_max_spanning_arborescence_cyclic_graph(self):
        g = DiGraph(test_edges_cyclic)
        spanning_edges_max_arborescence = {
            ("b", "a"),
            ("b", "c"),
            ("b", "g"),
            ("g", "h"),
            ("h", "d"),
            ("d", "e"),
            ("e", "f")
        }

        arborescence = next(directed.edmonds(g, minimum=False, find_spanning_arborescence=True))

        for edge_label in spanning_edges_max_arborescence:
            assert edge_label in arborescence

        assert arborescence.weight == 28, "max spanning arborescence weight should be 28"

    def test_edmonds_min_spanning_arborescence_cyclic_graph(self):
        g = DiGraph(test_edges_cyclic)
        spanning_edges_min_arborescence = {
            ("d", "e"),
            ("e", "f"),
            ("f", "h"),
            ("e", "c"),
            ("c", "a"),
            ("a", "b"),
            ("b", "g")
        }

        arborescence = next(directed.edmonds(g, minimum=True, find_spanning_arborescence=True))

        for edge_label in spanning_edges_min_arborescence:
            assert edge_label in arborescence

        assert arborescence.weight == 18, "min spanning arborescence weight should be 18"

    def test_edmonds_max_branching(self):
        g = DiGraph(test_edges_multi_tree)
        main_arborescence = {
            ("d", "a"),
            ("a", "b"),
            ("b", "c"),
            ("c", "f")
        }

        count = 0
        weight = 0
        for arborescence in directed.edmonds(g, minimum=False, find_spanning_arborescence=False):
            count += 1
            weight += arborescence.weight
            if len(arborescence) > 1:
                for edge_label in main_arborescence:
                    assert edge_label in arborescence

        assert count == 6, "max branching should have 6 arborescences"
        assert weight == 20

    def test_edmonds_min_branching(self):
        g = DiGraph(test_edges_multi_tree)
        main_arborescence = {
            ("j", "g"),
            ("g", "h"),
            ("h", "i"),
            ("g", "c")
        }

        count = 0
        weight = 0
        for arborescence in directed.edmonds(g, minimum=True, find_spanning_arborescence=False):
            count += 1
            weight += arborescence.weight
            if len(arborescence) > 2:
                for edge_label in main_arborescence:
                    assert edge_label in arborescence

        assert count == 5, "min branching should have 5 arborescences"
        assert weight == -24

    def test_edmonds_max_arborescence_multigraph(self):
        g = MultiDiGraph([("a", "b", 1), ("a", "b", 2), ("b", "c", 3), ("b", "c", 6), ("c", "d", 5),
            ("c", "d", 10), ("d", "a", 2), ("d", "a", 4)])
        spanning_edges_max_arborescence = {
            ("b", "c"),
            ("c", "d"),
            ("d", "a")
        }

        arborescence = next(directed.edmonds(g, minimum=False, find_spanning_arborescence=True))

        for edge_label in spanning_edges_max_arborescence:
            assert edge_label in arborescence

        assert len(arborescence.edges()) == 3, "max spanning arborescence should have 3 edges"

        weight = 0
        for edge in spanning_edges_max_arborescence:
            weight += max(c.weight for c in arborescence[edge].connections())
        assert weight == 20

    def test_edmonds_min_arborescence_multigraph(self):
        g = MultiDiGraph([("a", "b", 1), ("a", "b", 2), ("b", "c", 3), ("b", "c", 6), ("c", "d", 5),
            ("c", "d", 10), ("d", "a", 2), ("d", "a", 4)])
        spanning_edges_min_arborescence = {
            ("d", "a"),
            ("a", "b"),
            ("b", "c")
        }

        arborescence = next(directed.edmonds(g, minimum=True, find_spanning_arborescence=True))

        for edge_label in spanning_edges_min_arborescence:
            assert edge_label in arborescence

        assert len(arborescence.edges()) == 3, "min spanning arborescence should have 3 edges"

        weight = 0
        for edge in spanning_edges_min_arborescence:
            weight += min(c.weight for c in arborescence[edge].connections())
        assert weight == 6
