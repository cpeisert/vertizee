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

"""Tests for breadth-first search."""
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

import pytest

from vertizee import exception
from vertizee.algorithms.algo_utils.search_utils import Direction, Label, SearchResults
from vertizee.algorithms.search.breadth_first_search import (
    bfs,
    bfs_labeled_edge_traversal,
    bfs_vertex_traversal,
    INFINITY,
)
from vertizee.classes.data_structures.tree import Tree
from vertizee.classes.edge import Edge, MultiDiEdge
from vertizee.classes.graph import DiGraph, Graph, MultiDiGraph
from vertizee.classes.vertex import Vertex, MultiDiVertex


class TestBreadthFirstSearch:
    """Tests for breadth-first search."""

    def test_bfs_undirected_cyclic_graph(self) -> None:
        g = Graph()
        g.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (6, 7)])
        results: SearchResults[Vertex, Edge] = bfs(g, 0)
        tree: Tree[Vertex, Edge] = next(iter(results.graph_search_trees()))

        assert tree.root == 0, "BFS tree should be rooted at vertex 0"
        assert (
            len(results.graph_search_trees()) == 1
        ), "BFS search with source vertex should yield one BFS tree"
        assert len(tree) == 6, "BFS tree should have 6 vertices (excluding vertices 6 & 7)"
        assert (
            len(tree.edges()) == 5
        ), "BFS tree should have 5 edges, since for all trees |E| = |V| - 1"
        assert g[6] not in tree, "BFS tree should not contain vertex 6"

        # Breadth-first search does not support cycle detection.
        with pytest.raises(exception.VertizeeException):
            results.is_acyclic()

        assert (
            not results.has_topological_ordering()
        ), "no topological ordering for undirected graphs"
        with pytest.raises(exception.Unfeasible):
            results.vertices_topological_order()

        assert (
            results.vertices_preorder() == results.vertices_postorder()
        ), "a BFS should yield vertices in same order for both preorder and postorder"
        assert not results.back_edges(), "BFS on undirected graph cannot have back edges"
        assert not results.forward_edges(), "BFS on undirected graph cannot have forward edges"
        assert len(results.cross_edges()) == 2, "tree should have 2 cross edges"

        first_edge: Edge = results.edges_in_discovery_order()[0]
        assert first_edge.vertex1 == 0, "first edge should have vertex1 of 0"
        assert first_edge.vertex2 == 1, "first edge should have vertex2 of 1"

    def test_bfs_undirected_graph_with_self_loop_and_two_trees(self) -> None:
        g = Graph([(0, 0), (0, 1), (1, 2), (3, 4)])
        results: SearchResults[Vertex, Edge] = bfs(g)

        assert len(results.graph_search_trees()) == 2, "BFS should have discovered two BFS trees"
        assert len(results.vertices_preorder()) == 5, "BFS tree should have 5 vertices"
        assert (
            results.vertices_preorder() == results.vertices_postorder()
        ), "a BFS should yield vertices in same order for both preorder and postorder"
        assert len(results.back_edges()) == 1, "graph should have one self-loop back edge"

    def test_bfs_directed_cyclic_graph(self) -> None:
        g = MultiDiGraph()
        g.add_edges_from(
            [
                (1, 2),
                (2, 2),
                (2, 1),
                (1, 3),
                (4, 3),
                (4, 5),
                (4, 7),
                (5, 7),
                (5, 7),
                (7, 5),
                (7, 6),
                (6, 5),
                (7, 8),
                (5, 8),
            ]
        )

        # from pprint import pprint
        # pprint(list(bfs_labeled_edge_traversal(g, 1)))

        results1: SearchResults[MultiDiVertex, MultiDiEdge] = bfs(g, 1)
        search_trees = results1.graph_search_trees()
        assert (
            len(search_trees) == 1
        ), "BFS search should find 1 BFS tree, since source vertex was specified"
        tree: Tree[MultiDiVertex, MultiDiEdge] = search_trees[0]

        assert len(tree.edges()) == 2, "BFS tree rooted at vertex 1 should have 2 tree edges"
        assert len(tree) == 3, "BFS tree rooted at vertex 1 should have 3 vertices"

        assert results1.vertices_preorder()[0] == 1, "first vertex should be 1"
        assert len(results1.tree_edges()) == 2
        assert len(results1.back_edges()) == 2
        assert not results1.forward_edges()
        assert not results1.cross_edges()

        results4: SearchResults[MultiDiVertex, MultiDiEdge] = bfs(g, 4)
        assert len(results4.tree_edges()) == 5
        assert len(results4.back_edges()) == 1
        assert len(results4.cross_edges()) == 2
        assert len(results4.forward_edges()) == 1

        # Test DiGraph BFS without specifying a source vertex.
        results: SearchResults[MultiDiVertex, MultiDiEdge] = bfs(g)

        assert len(results.graph_search_trees()) > 1, "BFS search should find at least 2 BFS trees"
        assert (
            results.vertices_preorder() == results.vertices_postorder()
        ), "vertices should be the same in preorder and postorder"
        assert (
            len(results.vertices_postorder()) == 8
        ), "all vertices should be accounted for when a source vertex is not specified"

        classified_edge_count = (
            len(results.back_edges())
            + len(results.cross_edges())
            + len(results.forward_edges())
            + len(results.tree_edges())
        )
        assert classified_edge_count == len(g.edges()), "classified edges should equal total edges"

    def test_bfs_traversal_undirected_graph(self) -> None:
        g = Graph([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (6, 7)])
        edge_iter = bfs_labeled_edge_traversal(g)
        bfs_edge_tuples = list(edge_iter)

        tree_roots = set(
            child
            for parent, child, label, direction, depth in bfs_edge_tuples
            if label == Label.TREE_ROOT
        )
        assert len(tree_roots) == 2, "there should be two BFS trees"

        vertices = set(child for parent, child, label, direction, depth in bfs_edge_tuples)
        assert len(vertices) == 8, "BFS traversal should include all vertices"

        vertices_preorder = list(
            child
            for parent, child, label, direction, depth in bfs_edge_tuples
            if direction == Direction.PREORDER
        )
        vertices_postorder = list(
            child
            for parent, child, label, direction, depth in bfs_edge_tuples
            if direction == Direction.POSTORDER
        )
        assert (
            vertices_preorder == vertices_postorder
        ), "preorder vertices should match the postorder vertices"

        back_edges = set(
            (parent, child)
            for parent, child, label, direction, depth in bfs_edge_tuples
            if label == Label.BACK_EDGE
        )
        cross_edges = set(
            (parent, child)
            for parent, child, label, direction, depth in bfs_edge_tuples
            if label == Label.CROSS_EDGE
        )
        forward_edges = set(
            (parent, child)
            for parent, child, label, direction, depth in bfs_edge_tuples
            if label == Label.FORWARD_EDGE
        )
        tree_edges = set(
            (parent, child)
            for parent, child, label, direction, depth in bfs_edge_tuples
            if label == Label.TREE_EDGE
        )

        assert not back_edges
        assert not forward_edges
        assert len(cross_edges) == 2
        assert len(tree_edges) == 6

        # Test traversal when source is specified.
        edge_iter = bfs_labeled_edge_traversal(g, source=2)
        bfs_tuples2 = list(edge_iter)
        depth0 = next(
            depth
            for parent, child, label, direction, depth in bfs_tuples2
            if child == 0 and label == Label.TREE_EDGE
        )
        assert depth0 == 2, "vertex 0 should be at depth 2 in the tree relative to source vertex 2"
        depth3 = next(
            depth
            for parent, child, label, direction, depth in bfs_tuples2
            if child == 3 and label == Label.TREE_EDGE
        )
        assert depth3 == 1, "vertex 3 should be at depth 1 in the tree relative to source vertex 2"
        depth2 = next(
            depth
            for parent, child, label, direction, depth in bfs_tuples2
            if child == 2 and label == Label.TREE_ROOT
        )
        assert depth2 == 0, "vertex 2 should be at depth 0 in the BFS tree"
        depth1_3 = next(
            depth
            for parent, child, label, direction, depth in bfs_tuples2
            if child.label in ("1", "3") and label == Label.CROSS_EDGE
        )
        assert depth1_3 == INFINITY, "non-tree edges should report depth as infinity"

    def test_bfs_traversal_directed_graph(self) -> None:
        g = DiGraph([(0, 1), (1, 2), (2, 1)])

        tuple_generator = bfs_labeled_edge_traversal(g, source=0)
        parent, child, label, direction, depth = next(tuple_generator)
        assert parent == 0 and child == 0, "traversal should start with source vertex 0"
        assert label == Label.TREE_ROOT, "source vertex should be a BFS tree root"
        assert direction == Direction.PREORDER, "direction should start out as preorder"
        assert depth == 0

        parent, child, label, direction, depth = next(tuple_generator)
        assert parent == 0 and child == 1, "vertex after 0 should be 1"
        assert label == Label.TREE_EDGE
        assert direction == Direction.PREORDER
        assert depth == 1

        parent, child, label, direction, depth = next(tuple_generator)
        assert parent == 0 and child == 0, "finished visiting tree root 0"
        assert label == Label.TREE_ROOT
        assert direction == Direction.POSTORDER
        assert depth == 0

        parent, child, label, direction, depth = next(tuple_generator)
        assert parent == 1 and child == 2, "discovered vertex 2"
        assert label == Label.TREE_EDGE
        assert direction == Direction.PREORDER
        assert depth == 2

        parent, child, label, direction, depth = next(tuple_generator)
        assert parent == 0 and child == 1, "finished visiting vertex 1"
        assert label == Label.TREE_EDGE
        assert direction == Direction.POSTORDER
        assert depth == 1

        parent, child, label, direction, depth = next(tuple_generator)
        assert parent == 2 and child == 1
        assert label == Label.BACK_EDGE
        assert direction == Direction.ALREADY_DISCOVERED
        assert depth == INFINITY

        # Test depth limit.
        vertex_generator = bfs_vertex_traversal(g, source=0, depth_limit=2)
        v = next(vertex_generator)
        assert v == 0
        v = next(vertex_generator)
        assert v == 1
        # With depth_limit = 2, StopIteration should be raised on third request to next().
        with pytest.raises(StopIteration):
            v = next(vertex_generator)

        vertex_generator = bfs_vertex_traversal(g, source=0)
        v = next(vertex_generator)
        assert v == 0, "first preorder vertex should be 0"
        v = next(vertex_generator)
        assert v == 1, "second preorder vertex should be 1"
        v = next(vertex_generator)
        assert v == 2, "third preorder vertex should be 2"
        with pytest.raises(StopIteration):
            v = next(vertex_generator)

    def test_dfs_reverse_traversal(self) -> None:
        g = DiGraph([(0, 1), (1, 2)])

        vertices = list(bfs_vertex_traversal(g, source=1))
        assert vertices == [g[1], g[2]]
        vertices = list(bfs_vertex_traversal(g, source=1, reverse_graph=True))
        assert vertices == [g[1], g[0]]
