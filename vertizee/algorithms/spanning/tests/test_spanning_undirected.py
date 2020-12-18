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
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

import pytest

from vertizee import exception
from vertizee.algorithms.spanning import undirected
from vertizee.classes.graph import DiGraph, Graph, MultiGraph


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


class TestKruskal:
    """Tests for Kruskal's algorithm to find spanning trees."""

    def test_kruskal_directed_graph(self):
        g = DiGraph([("s", "t", 10), ("s", "y", 5), ("t", "y", 2)])

        # Kruskal algorithm does not work on directed graphs.
        with pytest.raises(exception.GraphTypeNotSupported):
            for _ in undirected.kruskal_spanning_tree(g):
                pass

    def test_kruskal_max_spanning_tree(self):
        g = Graph(test_edges)
        kruskal_edges_max = [
            ("c", "h"),
            ("a", "b"),
            ("d", "f"),
            ("e", "g"),
            ("b", "c"),
            ("f", "g"),
            ("g", "h"),
        ]
        tree_weight = 0
        for i, edge in enumerate(undirected.kruskal_spanning_tree(g, minimum=False)):
            tree_weight += edge.weight
            assert (
                edge.vertex1 in kruskal_edges_max[i] and edge.vertex2 in kruskal_edges_max[i]
            ), f"Kruskal max spanning tree edge {i} should have vertices {kruskal_edges_max[i]}"
        assert tree_weight == 49, "Max spanning tree weight should be 49."

    def test_kruskal_min_spanning_tree(self):
        g = Graph(test_edges)
        kruskal_edges_min = [
            ("c", "f"),
            ("a", "e"),
            ("c", "d"),
            ("g", "h"),
            ("f", "g"),
            ("b", "c"),
            ("e", "g"),
        ]
        tree_weight = 0
        for i, edge in enumerate(undirected.kruskal_spanning_tree(g)):
            tree_weight += edge.weight
            assert (
                edge.vertex1 in kruskal_edges_min[i] and edge.vertex2 in kruskal_edges_min[i]
            ), f"Kruskal min spanning tree edge {i} should have vertices {kruskal_edges_min[i]}"
        assert tree_weight == 28, "Min spanning tree weight should be 28."

    def test_kruskal_multigraph(self):
        g = MultiGraph([("a", "b", 5), ("a", "c", 3), ("b", "d", 6), ("c", "d", 4), ("c", "d", 7)])

        min_tree = list(undirected.kruskal_spanning_tree(g, minimum=True))
        assert len(min_tree) == 3
        assert min_tree[0] == g["a", "c"]
        assert min_tree[1] == g["c", "d"]
        assert min_tree[2] == g["a", "b"]

        max_tree = list(undirected.kruskal_spanning_tree(g, minimum=False))
        assert len(max_tree) == 3
        assert max_tree[0] == g["c", "d"]
        assert max_tree[1] == g["b", "d"]
        assert max_tree[2] == g["a", "b"]

    def test_kruskal_optimum_forest_single_tree(self):
        g = Graph(test_edges)

        spanning_edges = set(undirected.kruskal_spanning_tree(g))
        spanning_tree = next(undirected.kruskal_optimum_forest(g))

        assert spanning_edges == set(spanning_tree.edges())
        assert spanning_tree.weight == 28, "Min spanning tree weight should be 28."

    def test_kruskal_optimum_forest_multiple_trees(self):
        g = Graph(test_edges)
        g.add_edge("x", "y", weight=22)
        g.add_edge("y", "z", weight=20)
        g.add_vertex("isolated")

        count = 0
        total_weight = 0
        for tree in undirected.kruskal_optimum_forest(g):
            count += 1
            total_weight += tree.weight

        assert count == 3, "there should be 3 trees in the spanning forest"
        assert total_weight == 70, "total weight of trees should be 70"


class TestPrim:
    """Tests for Prim's algorithm to find spanning trees."""

    def test_prim_max_spanning_tree(self):
        g = Graph(test_edges)
        prim_edges_max = [
            ("a", "b"),
            ("b", "c"),
            ("c", "h"),
            ("g", "h"),
            ("e", "g"),
            ("f", "g"),
            ("d", "f"),
        ]
        tree_weight = 0
        for i, edge in enumerate(undirected.prim_spanning_tree(g, root="a", minimum=False)):
            tree_weight += edge.weight
            assert (
                edge.vertex1 in prim_edges_max[i] and edge.vertex2 in prim_edges_max[i]
            ), f"Prim max spanning tree edge {i} should have vertices {prim_edges_max[i]}"
        assert tree_weight == 49, "Max spanning tree weight should be 49."

    def test_prim_min_spanning_tree(self):
        g = Graph(test_edges)
        prim_edges_min = [
            ("a", "e"),
            ("e", "g"),
            ("g", "h"),
            ("f", "g"),
            ("c", "f"),
            ("c", "d"),
            ("b", "c"),
        ]
        tree_weight = 0
        for i, edge in enumerate(undirected.prim_spanning_tree(g, root="a")):
            tree_weight += edge.weight
            assert (
                edge.vertex1 in prim_edges_min[i] and edge.vertex2 in prim_edges_min[i]
            ), f"Prim min spanning tree edge {i} should have vertices {prim_edges_min[i]}"
        assert tree_weight == 28, "Min spanning tree weight should be 28."

    def test_prim_multigraph(self):
        g = MultiGraph([("a", "b", 5), ("a", "c", 3), ("b", "d", 6), ("c", "d", 4), ("c", "d", 7)])

        min_tree = list(undirected.prim_spanning_tree(g, root="a", minimum=True))
        assert len(min_tree) == 3
        assert min_tree[0] == g["a", "c"]
        assert min_tree[1] == g["c", "d"]
        assert min_tree[2] == g["a", "b"]

        max_tree = list(undirected.prim_spanning_tree(g, root="a", minimum=False))
        assert len(max_tree) == 3
        assert max_tree[0] == g["a", "b"]
        assert max_tree[1] == g["b", "d"]
        assert max_tree[2] == g["c", "d"]

    def test_prim_fibonacci_max_spanning_tree(self):
        g = Graph(test_edges)
        prim_edges_max = [
            ("a", "b"),
            ("b", "c"),
            ("c", "h"),
            ("g", "h"),
            ("e", "g"),
            ("f", "g"),
            ("d", "f"),
        ]
        tree_weight = 0
        for i, edge in enumerate(undirected.prim_fibonacci(g, root="a", minimum=False)):
            tree_weight += edge.weight
            # print(f'DEBUG Prim found edge {edge} with weight {edge.weight}')
            assert (
                edge.vertex1 in prim_edges_max[i] and edge.vertex2 in prim_edges_max[i]
            ), f"Prim max spanning tree edge {i} should have vertices {prim_edges_max[i]}"
        assert tree_weight == 49, "Max spanning tree weight should be 49."

    def test_prim_fibonacci_min_spanning_tree(self):
        g = Graph(test_edges)
        prim_edges_min = [
            ("a", "e"),
            ("e", "g"),
            ("g", "h"),
            ("f", "g"),
            ("c", "f"),
            ("c", "d"),
            ("b", "c"),
        ]
        tree_weight = 0
        for i, edge in enumerate(undirected.prim_fibonacci(g, root="a")):
            tree_weight += edge.weight
            # print(f'DEBUG Prim found edge {edge} with weight {edge.weight}')
            assert (
                edge.vertex1 in prim_edges_min[i] and edge.vertex2 in prim_edges_min[i]
            ), f"Prim min spanning tree edge {i} should have vertices {prim_edges_min[i]}"
        assert tree_weight == 28, "Min spanning tree weight should be 28."
