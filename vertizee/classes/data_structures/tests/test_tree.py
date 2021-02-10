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

"""Tests for trees."""
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

import pytest

from vertizee import exception
from vertizee.classes.data_structures.tree import Tree
from vertizee.classes.edge import Edge
from vertizee.classes.graph import Graph
from vertizee.classes.vertex import Vertex


class TestTree:
    """Tests for trees."""

    def test__contains__(self):
        g = Graph([(1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])
        tree = Tree(g[1])
        assert g[1] in tree, "vertex 1 should be in tree"
        assert 1 in tree, "vertex specified as int should be in tree"
        assert "1" in tree, "vertex specified as str should be in tree"
        assert 3 not in tree, "vertex 3 should not be in tree"

        tree.add_edge(g.get_edge(1, 2))
        assert g.get_edge(1, 2) in tree
        assert (1, 2) in tree, "edge specified as tuple should be in tree"
        assert ("1", "2") in tree
        assert (1, 3) not in tree

        with pytest.raises(TypeError):
            _ = 4.5 not in g
        with pytest.raises(TypeError):
            _ = (1, 2, 3, 4) not in g

    def test__getitem__(self):
        g = Graph([(1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])
        tree = Tree(g[1])
        assert isinstance(tree[1], Vertex), "tree should have vertex 1"
        assert isinstance(tree["1"], Vertex), "tree should have vertex 1"
        assert isinstance(tree[(1, {})], Vertex), "tree should have vertex 1"
        assert isinstance(tree[1, {}], Vertex), "tree should have vertex 1"
        v1 = tree[1]
        assert isinstance(tree[v1], Vertex), "tree should have vertex 1"
        with pytest.raises(TypeError):
            _ = tree[2.0]
        with pytest.raises(KeyError):
            _ = tree[3]

        tree.add_edge(g.get_edge(1, 2))
        assert isinstance(tree[1, 2], Edge), "tree should have edge (1, 2)"
        assert isinstance(tree["1", "2"], Edge), "tree should have edge (1, 2)"
        assert isinstance(tree[(1, 2, {})], Edge), "tree should have edge (1, 2)"
        assert isinstance(tree[1, 2, {}], Edge), "tree should have edge (1, 2)"
        assert isinstance(tree[(1, 2, 1.0)], Edge), "tree should have edge (1, 2)"
        assert isinstance(tree[1, 2, 1.0], Edge), "tree should have edge (1, 2)"
        assert isinstance(tree[(1, 2, 1.0, {})], Edge), "tree should have edge (1, 2)"
        assert isinstance(tree[1, 2, 1.0, {}], Edge), "tree should have edge (1, 2)"
        edge = tree[1, 2]
        assert isinstance(tree[edge], Edge), "tree should have edge (1, 2)"
        with pytest.raises(TypeError):
            _ = g.get_edge(1.0, 2.0)
        with pytest.raises(KeyError):
            _ = g.get_edge(1, 3)

    def test__iter__(self):
        g = Graph([(1, 2), (2, 3), (3, 4)])
        tree = Tree(g[1])
        tree.add_edge(g.get_edge(1, 2))
        tree.add_edge(g.get_edge(2, 3))
        tree.add_edge(g.get_edge(3, 4))
        count = sum(1 for _ in tree)
        assert count == 4, "tree should iterate over its 4 vertices"
        assert set([tree[1], tree[2], tree[3], tree[4]]) == set(
            tree
        ), "tree should iterate over its 4 vertices"

    def test__len__(self):
        g = Graph([(1, 2), (2, 3), (3, 4)])
        tree = Tree(g[1])
        tree.add_edge(g.get_edge(1, 2))
        tree.add_edge(g.get_edge(2, 3))
        tree.add_edge(g.get_edge(3, 4))
        assert len(tree) == 4, "tree should contain 4 vertices"

    def test_add_edge(self):
        g = Graph([(1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])
        tree = Tree(g[1])
        tree.add_edge(g.get_edge(1, 2))
        assert (1, 2) in tree

        tree.add_edge(g.get_edge(2, 3))
        with pytest.raises(exception.Unfeasible):
            # Raise exception due to (4, 5) not containing a vertex already in the tree.
            tree.add_edge(g.get_edge(4, 5))

        tree.add_edge(g.get_edge(3, 4))
        with pytest.raises(exception.Unfeasible):
            # Raises exception due to cycle.
            tree.add_edge(g.get_edge(1, 4))

    def test_merge(self):
        g = Graph([(1, 2), (2, 3), (1, 4), (3, 4), (4, 5)])
        tree1 = Tree(g[1])
        tree5 = Tree(g[5])

        tree1.add_edge(g.get_edge(1, 4))
        tree1.add_edge(g.get_edge(4, 3))
        tree5.add_edge(g.get_edge(5, 4))
        tree1.merge(tree5)
        assert tree1._vertex_set == {tree1[1], tree1[3], tree1[4], tree1[5]}
