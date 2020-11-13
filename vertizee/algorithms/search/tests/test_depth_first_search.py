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

"""Tests finding the shortest paths in unweighted graphs."""
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

import pytest

from vertizee.algorithms.algo_utils.search_utils import DepthFirstSearchResults, SearchTree
from vertizee.algorithms.search.depth_first_search import (
    depth_first_search,
    dfs_labeled_edge_traversal,
    dfs_postorder_traversal,
    dfs_preorder_traversal,
    Direction,
    Label
)
from vertizee.classes.edge import Edge
from vertizee.classes.graph import DiGraph, Graph, MultiDiGraph


class TestDepthFirstSearch:
    """Tests for depth-first search algorithms and traversals."""

    def test_dfs_undirected_cyclic_graph(self):
        g = Graph()
        g.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (6, 7)])
        dfs: DepthFirstSearchResults = depth_first_search(g, 0)
        t, *_ = dfs.depth_first_search_trees()
        tree: SearchTree = t

        assert tree.root == 0, "DFS tree should be rooted at vertex 0"
        assert (
            len(dfs.depth_first_search_trees()) == 1
        ), "DFS search with source vertex should yield one DFS tree"
        assert (
            len(tree.vertices_in_discovery_order()) == 6
        ), "DFS tree should have 6 vertices (excluding vertices 6 & 7)"
        assert len(tree.edges_in_discovery_order()) == 5, (
            "DFS tree should have 5 edges, since for all trees |E| = |V| - 1"
        )
        assert (
            g[6] not in tree.vertices_in_discovery_order()
        ), "DFS tree should not contain vertex 6"
        assert not dfs.is_acyclic(), "graph should not be acyclic, since it contains 2 cycles"

        assert len(tree.vertices_in_discovery_order()) == len(dfs.vertices_pre_order()), (
            "DFS vertices should match the DFS tree, since only one tree was searched"
        )
        assert len(tree.edges_in_discovery_order()) == len(
            dfs.edges_in_discovery_order()
        ), "DFS edges should match the DFS tree, since only one tree was searched"
        assert len(dfs.back_edges()) > 0, "graph should have back edges, since there are cycles"
        topological_sort = dfs.topological_sort()
        assert topological_sort is None, (
            "Topological sort should be None, since the graph contains cycles"
        )
        first_edge: Edge = dfs.edges_in_discovery_order()[0]
        assert first_edge.vertex1 == 0, "first edge should have vertex1 of 0"
        assert first_edge.vertex2 == 1, "first edge should have vertex2 of 1"

        assert (
            not dfs.cross_edges() and not dfs.forward_edges()
        ), "in an undirected graph, every edge is either a tree edge or a back edge"


    def test_dfs_undirected_cyclic_graph_with_self_loop(self):
        g = Graph([(0, 0), (0, 1), (1, 2), (3, 4)])
        dfs: DepthFirstSearchResults = depth_first_search(g)

        assert len(dfs.depth_first_search_trees()) == 2, "DFS should have discovered two DFS trees"
        assert len(dfs.vertices_pre_order()) == 5, "DFS tree should have 5 vertices"
        assert len(dfs.vertices_pre_order()) == len(
            dfs.vertices_post_order()
        ), "number of vertices should be the same in discovery as well post order"
        assert len(dfs.back_edges()) == 1, "graph should have one self-loop back edge"
        assert len(dfs.cross_edges()) == 0, (
            "graph should have zero cross edges (true for all undirected graphs)"
        )
        assert len(dfs.forward_edges()) == 0, (
            "graph should have zero forward edges (true for all undirected graphs)"
        )
        assert not dfs.is_acyclic(), "graph should not be acyclic, since it contains a self loop"

    def test_dfs_directed_cyclic_graph(self):
        g = MultiDiGraph()
        # This graph is from "Introduction to Algorithms: Third Edition", page 607.
        g.add_edges_from(
            [
                ("y", "x"),
                ("z", "y"),
                ("z", "w"),
                ("s", "z"),
                ("s", "w"),
                ("t", "v"),
                ("t", "u"),
                ("x", "z"),
                ("w", "x"),
                ("v", "w"),
                ("v", "s"),
                ("u", "v"),
                ("u", "t"),
            ]
        )

        # Test DiGraph DFS by specifying source vertex s.
        dfs: DepthFirstSearchResults = depth_first_search(g, "s")

        assert len(dfs.depth_first_search_trees()) == 1, (
            "DFS search should find 1 DFS tree, since source vertex 's' was specified"
        )
        t, *_ = dfs.depth_first_search_trees()
        tree: SearchTree = t

        assert len(tree.edges_in_discovery_order()) == 4, (
            "DFS tree rooted at vertex 's' should have 4 edges"
        )
        assert (
            len(tree.vertices_in_discovery_order()) == 5
        ), "DFS tree rooted at vertex 's' should have 5 vertices"

        assert dfs.vertices_pre_order()[0] == "s", "first vertex should be s"
        assert not dfs.is_acyclic(), "graph should not be acyclic, since it contains cycles"

        # Test DiGraph DFS without specifying a source vertex.
        dfs: DepthFirstSearchResults = depth_first_search(g)

        assert len(dfs.vertices_pre_order()) == len(
            dfs.vertices_post_order()
        ), "graph should equal number of vertices in pre and post order"
        assert len(dfs.vertices_post_order()) == 8, (
            "all vertices should be accounted for when a source vertex is not specified"
        )

        dfs_no_source: DepthFirstSearchResults = depth_first_search(g)

        classified_edge_count = (len(dfs_no_source.back_edges()) + len(dfs_no_source.cross_edges())
            + len(dfs_no_source.forward_edges()) + len(dfs_no_source.tree_edges()))
        assert classified_edge_count == g.edge_count, "classified edges should equal total edges"

    def test_topological_sort(self):
        g = DiGraph([("s", "t"), ("t", "u"), ("u", "v")])

        dfs: DepthFirstSearchResults = depth_first_search(g)
        topo_sorted = dfs.topological_sort()
        assert topo_sorted[0] == "s", "first element of path graph topo sort should be s"
        assert topo_sorted[1] == "t", "second element of path graph topo sort should be t"
        assert topo_sorted[2] == "u", "third element of path graph topo sort should be u"
        assert topo_sorted[3] == "v", "fourth element of path graph topo sort should be v"

        g = DiGraph()
        g.add_edges_from([("s", "v"), ("s", "w"), ("v", "t"), ("w", "t")])

        dfs: DepthFirstSearchResults = depth_first_search(g)
        topo_sorted = dfs.topological_sort()
        assert topo_sorted[0] == "s", "first element of topo sort should be s"
        assert topo_sorted[1] == "v" or topo_sorted[1] == "w", (
            "second element of topo sort " "should be v or w"
        )
        assert topo_sorted[3] == "t", "fourth element topo sort should be t"

    def test_dfs_traversal_undirected_graph(self):
        g = Graph([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (6, 7)])
        edge_iter = dfs_labeled_edge_traversal(g)
        dfs_edge_tuples = list(edge_iter)

        tree_roots = set(parent for parent, child, label, direction in dfs_edge_tuples
            if label == Label.TREE_ROOT)
        assert len(tree_roots) == 2, "there should be two DFS trees"

        vertices = set(child for parent, child, label, direction in dfs_edge_tuples)

        print(f"DEBUG dfs_edge_tuples => {dfs_edge_tuples}")
        print(f"DEBUG vertices => {vertices}")

        assert len(vertices) == 8, "DFS traversal should include all vertices"

        vertices_preorder = list(child for parent, child, label, direction in dfs_edge_tuples
            if direction == Direction.PREORDER)
        vertices_postorder = list(child for parent, child, label, direction in dfs_edge_tuples
            if direction == Direction.POSTORDER)
        assert len(vertices_preorder) == len(vertices_postorder), (
            "the number of preorder vertices should match the number of postorder vertices"
        )

        back_edges = set((parent, child) for parent, child, label, direction in dfs_edge_tuples
            if label == Label.BACK_EDGE)
        cross_edges = set((parent, child) for parent, child, label, direction in dfs_edge_tuples
            if label == Label.CROSS_EDGE)
        forward_edges = set((parent, child) for parent, child, label, direction in dfs_edge_tuples
            if label == Label.FORWARD_EDGE)

        assert len(back_edges) > 0, "graph should have back edges, since there are cycles"
        assert (
            not cross_edges and not forward_edges
        ), "in an undirected graph, every edge is either a tree edge or a back edge"

    def test_dfs_traversal_directed_graph(self):
        g = DiGraph([(0, 1), (1, 2), (2, 1)])

        # from pprint import pprint
        # print('\n\nPretty Print DFS Labeled Edge Traversal\n')
        # pprint(list(dfs_labeled_edge_traversal(g, source=0)))

        tuple_generator = dfs_labeled_edge_traversal(g, source=0)
        parent, child, label, direction = next(tuple_generator)
        assert parent == 0 and child == 0, "Traversal should start with source vertex 0."
        assert label == Label.TREE_ROOT, "Source vertex should be a DFS tree root."
        assert direction == Direction.PREORDER, "Direction should start out as preorder."

        parent, child, label, direction = next(tuple_generator)
        assert parent == 0 and child == 1, "Vertex after 0 should be 1."
        assert label == Label.TREE_EDGE, "Source vertex should be a DFS tree root."
        assert direction == Direction.PREORDER, "Direction should start out as preorder."

        vertex_generator = dfs_postorder_traversal(g, source=0)
        v = next(vertex_generator)
        assert v == 2, "First vertex in postorder should be 2"
        v = next(vertex_generator)
        assert v == 1, "Second vertex in postorder should 1"
        v = next(vertex_generator)
        assert v == 0, "Third vertex in postorder should 0"

        # Test depth limit.
        vertex_generator = dfs_postorder_traversal(g, source=0, depth_limit=2)
        v = next(vertex_generator)
        assert v == 1, "First vertex in postorder should be 1 with depth limited to 2"
        v = next(vertex_generator)
        assert v == 0, "Second vertex in postorder should 0"
        # With depth_limit = 2, StopIteration should be raised on third request to next().
        with pytest.raises(StopIteration):
            v = next(vertex_generator)

        vertex_generator = dfs_preorder_traversal(g, source=0)
        v = next(vertex_generator)
        assert v == 0, "First preorder vertex should be 0"
        v = next(vertex_generator)
        assert v == 1, "Second preorder vertex should be 1"
        v = next(vertex_generator)
        assert v == 2, "Third preorder vertex should be 2"
        with pytest.raises(StopIteration):
            v = next(vertex_generator)

    def test_dfs_reverse_traversal(self):
        g = DiGraph([(0, 1), (1, 2), (2, 0)])

        vertices = list(dfs_preorder_traversal(g, source=0))
        assert vertices == [0, 1, 2], "Preorder vertices should be 0, 1, 2."
        vertices = list(dfs_preorder_traversal(g, source=0, reverse_graph=True))
        assert vertices == [0, 2, 1], "Reverse graph preorder vertices should be 0, 2, 1."

        vertices = list(dfs_postorder_traversal(g, source=0))
        assert vertices == [2, 1, 0], "Postorder vertices should be 2, 1, 0."
        vertices = list(dfs_postorder_traversal(g, source=0, reverse_graph=True))
        assert vertices == [1, 2, 0], "Reverse graph postorder vertices should be 1, 2, 0."

        g = DiGraph([(0, 1), (1, 4), (4, 0), (4, 3), (3, 1), (2, 3), (2, 4)])

        vertices = list(dfs_preorder_traversal(g, source=0))
        assert vertices == [0, 1, 4, 3], "Preorder vertices should be 0, 1, 4, 3."
        vertices = list(dfs_preorder_traversal(g, source=0, reverse_graph=True))

        preorder1 = [0, 4, 1, 3, 2]
        preorder2 = [0, 4, 2, 1, 3]
        assert (
            vertices in (preorder1, preorder2)
        ), f"Reverse graph preorder vertices should be either {preorder1} or {preorder2}."
