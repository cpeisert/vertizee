#!/usr/bin/env python
#
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

"""Tests for directed graphs: DiGraph, MultiDiGraph."""

from typing import List

import pytest

from vertizee.classes.digraph import DiGraph, MultiDiGraph
from vertizee.classes.edge import DEFAULT_WEIGHT, DiEdge
from vertizee.classes.graph_base import GraphBase
from vertizee.classes.vertex import Vertex


pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip.")


@pytest.mark.usefixtures()
class TestDirectedGraph:

    def test_vertex(self):
        g = MultiDiGraph()
        v0 = g.add_vertex('0')
        assert v0.key == '0', f'DiVertex v0 should have key "0", but had key "{v0.key}"'
        assert v0.degree == 0, f'DiVertex v0 should have degree 0, but had degree {v0.degree}'
        assert len(v0.edges_incoming) == 0, 'DiVertex v0 should have no incoming edges.'

        v1 = g.add_vertex('1')
        assert v1.key == '1', f'DiVertex v1 should have key "1", but had key "{v1.key}"'
        assert v1.degree == 0, f'DiVertex v1 should have degree 0, but had degree {v1.degree}'

    def test_edge(self):
        g = MultiDiGraph()
        v0 = g.add_vertex('0')
        v1 = g.add_vertex('1')
        e1 = g.add_edge(tail=v0, head=v1)
        e_loop = g.add_edge(tail=v0, head=v0)

        assert e1.weight == DEFAULT_WEIGHT, \
            f'Edge e1 should have weight {DEFAULT_WEIGHT} (default), but weight was {e1.weight}'
        assert e1.parallel_edge_count == 0, 'Edge e1 should have zero parallel edges.'
        assert e1.tail == v0, f'Edge e1 should have tail ({v0.key}), but tail was ({e1.tail.key})'
        assert e1.head == v1, f'Edge e1 should have head ({v1.key}), but head was ({e1.head.key})'
        assert e_loop.is_loop(), 'Edge e_loop should be a loop.'
        assert v0.degree == 3, 'DiVertex v0 should have degree 3.'
        assert len(v0.adjacent_vertices_outgoing) == 1, 'Vertex v0 should have 1 outgoing ' \
            'adjacent vertex.'
        adj_vertex = v0.adjacent_vertices_outgoing.pop()
        assert adj_vertex.key == '1', 'Vertex v0 should be adjacent to vertex 1.'

        v2 = g.add_vertex('2')
        e2 = g.add_edge(
            tail=v1, head=v2, weight=1.5, parallel_edge_count=1, parallel_edge_weights=[3])
        assert e2.weight == 1.5, 'Edge e2 should have weight 1.5.'
        assert e2.weight_with_parallel_edges == 4.5, \
            'Edge e2 should have total weight 4.5 including parallel edges.'
        assert e2.parallel_edge_count == 1, 'Edge e2 should have 1 parallel edge.'

    def test_digraph(self):
        g = DiGraph()
        vs0 = g.add_vertex('s0')
        vs1 = g.add_vertex('s1')

        g.add_edge(vs0, vs1)
        # Attempting to add a parallel edge with different tail and head vertices to a digraph
        # should not raise an error. "Two edges are parallel if they connect the same ordered
        # pair of vertices."
        # See: https://algs4.cs.princeton.edu/42digraph/
        g.add_edge(vs1, vs0)

        edges: List[DiEdge] = g.edges
        edge = edges.pop()
        assert isinstance(edge, DiEdge), 'DiGraphs should have edges of type DiEdge.'

    @staticmethod
    def build_parallel_weighted_graph(g: GraphBase, v: List[Vertex]) -> GraphBase:
        # (0, 0, 0.5)
        #   - 6 loop edges
        #   - sum(parallel_edge_weights) => 10
        g.add_edge(tail=v[0], head=v[0], weight=0.5, parallel_edge_count=5,
                   parallel_edge_weights=list(range(5)))
        # (0, 1, 0.5)
        #   - 101 edges
        #   - sum(parallel_edge_weights) => 4950
        g.add_edge(tail=v[0], head=v[1], weight=0.5, parallel_edge_count=100,
                   parallel_edge_weights=list(range(100)))
        # (1, 0, 1.0)
        #   - 5 edges
        #   - sum(parallel_edge_weights) => 6
        g.add_edge(tail=v[1], head=v[0], weight=1.0, parallel_edge_count=4,
                   parallel_edge_weights=list(range(4)))
        # (1, 2, 1.5)
        #   - 1 edge
        g.add_edge(tail=v[1], head=v[2], weight=1.5)
        # (2, 0, 0.1)
        #   - 1 edge
        g.add_edge(tail=v[2], head=v[0], weight=0.1)
        # (3, 2, 1.9)
        #   - 1 edge
        g.add_edge(tail=v[3], head=v[2], weight=1.9)
        # Isolated vertex
        #   - 1 vertex (0 edges)
        g.add_vertex(v[4].key)

        return g
