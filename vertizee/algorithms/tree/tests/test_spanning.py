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

"""Tests for spanning tree algorithms."""

import pytest

from vertizee.algorithms.tree.spanning import spanning_tree_kruskal
from vertizee.algorithms.tree.spanning import spanning_tree_prim
from vertizee.algorithms.tree.spanning import spanning_tree_prim_fibonacci
from vertizee.classes.digraph import DiGraph
from vertizee.classes.graph import Graph

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip."
)


test_edges = [
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


@pytest.mark.usefixtures()
class TestSpanningTrees:
    def test_kruskal_directed_graph(self):
        g = DiGraph([("s", "t", 10), ("s", "y", 5), ("t", "y", 2)])

        # Kruskal algorithm does not work on directed graphs.
        with pytest.raises(ValueError):
            for _ in spanning_tree_kruskal(g):
                pass

    def test_kruskal_max_spanning_tree(self):
        g = Graph(test_edges)
        kruskal_edges_max = [
            {"c", "h"},
            {"a", "b"},
            {"d", "f"},
            {"e", "g"},
            {"b", "c"},
            {"f", "g"},
            {"g", "h"},
        ]
        tree_weight = 0
        for i, edge in enumerate(spanning_tree_kruskal(g, minimum=False)):
            tree_weight += edge.weight
            assert (
                edge.vertex1 in kruskal_edges_max[i] and edge.vertex2 in kruskal_edges_max[i]
            ), f"Kruskal max spanning tree edge {i} should have vertices {kruskal_edges_max[i]}"
        assert tree_weight == 49, "Max spanning tree weight should be 49."

    def test_kruskal_min_spanning_tree(self):
        g = Graph(test_edges)
        kruskal_edges_min = [
            {"c", "f"},
            {"a", "e"},
            {"c", "d"},
            {"g", "h"},
            {"f", "g"},
            {"b", "c"},
            {"e", "g"},
        ]
        tree_weight = 0
        for i, edge in enumerate(spanning_tree_kruskal(g)):
            tree_weight += edge.weight
            assert (
                edge.vertex1 in kruskal_edges_min[i] and edge.vertex2 in kruskal_edges_min[i]
            ), f"Kruskal min spanning tree edge {i} should have vertices {kruskal_edges_min[i]}"
        assert tree_weight == 28, "Min spanning tree weight should be 28."

    def test_prim_max_spanning_tree(self):
        g = Graph(test_edges)
        prim_edges_max = [
            {"a", "b"},
            {"b", "c"},
            {"c", "h"},
            {"g", "h"},
            {"e", "g"},
            {"f", "g"},
            {"d", "f"},
        ]
        tree_weight = 0
        for i, edge in enumerate(spanning_tree_prim(g, root="a", minimum=False)):
            tree_weight += edge.weight
            assert (
                edge.vertex1 in prim_edges_max[i] and edge.vertex2 in prim_edges_max[i]
            ), f"Prim max spanning tree edge {i} should have vertices {prim_edges_max[i]}"
        assert tree_weight == 49, "Max spanning tree weight should be 49."

    def test_prim_min_spanning_tree(self):
        g = Graph(test_edges)
        prim_edges_min = [
            {"a", "e"},
            {"e", "g"},
            {"g", "h"},
            {"f", "g"},
            {"c", "f"},
            {"c", "d"},
            {"b", "c"},
        ]
        tree_weight = 0
        for i, edge in enumerate(spanning_tree_prim(g, root="a")):
            tree_weight += edge.weight
            assert (
                edge.vertex1 in prim_edges_min[i] and edge.vertex2 in prim_edges_min[i]
            ), f"Prim min spanning tree edge {i} should have vertices {prim_edges_min[i]}"
        assert tree_weight == 28, "Min spanning tree weight should be 28."

    def test_prim_max_spanning_tree_fibonacci(self):
        g = Graph(test_edges)
        prim_edges_max = [
            {"a", "b"},
            {"b", "c"},
            {"c", "h"},
            {"g", "h"},
            {"e", "g"},
            {"f", "g"},
            {"d", "f"},
        ]
        tree_weight = 0
        for i, edge in enumerate(spanning_tree_prim_fibonacci(g, root="a", minimum=False)):
            tree_weight += edge.weight
            # print(f'DEBUG Prim found edge {edge} with weight {edge.weight}')
            assert (
                edge.vertex1 in prim_edges_max[i] and edge.vertex2 in prim_edges_max[i]
            ), f"Prim max spanning tree edge {i} should have vertices {prim_edges_max[i]}"
        assert tree_weight == 49, "Max spanning tree weight should be 49."

    def test_prim_min_spanning_tree_fibonacci(self):
        g = Graph(test_edges)
        prim_edges_min = [
            {"a", "e"},
            {"e", "g"},
            {"g", "h"},
            {"f", "g"},
            {"c", "f"},
            {"c", "d"},
            {"b", "c"},
        ]
        tree_weight = 0
        for i, edge in enumerate(spanning_tree_prim_fibonacci(g, root="a")):
            tree_weight += edge.weight
            # print(f'DEBUG Prim found edge {edge} with weight {edge.weight}')
            assert (
                edge.vertex1 in prim_edges_min[i] and edge.vertex2 in prim_edges_min[i]
            ), f"Prim min spanning tree edge {i} should have vertices {prim_edges_min[i]}"
        assert tree_weight == 28, "Min spanning tree weight should be 28."
