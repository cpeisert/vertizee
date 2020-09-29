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

"""Test the graph input/output routines for adjency lists."""

import os

import pytest

from vertizee.classes.digraph import MultiDiGraph
from vertizee.classes.graph import MultiGraph
from vertizee.io.adj_list import read_adj_list, write_adj_list_to_file

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip."
)

DIGRAPH_FILE01 = "adj_list_digraph01.txt"
DIGRAPH_OUTPUT_FILE = "adj_list_digraph01_TEST_OUTPUT.txt"
GRAPH_FILE01 = "adj_list_graph01.txt"
GRAPH_FILE02 = "adj_list_graph02.txt"
GRAPH_FILE03 = "adj_list_graph03.txt"
GRAPH_OUTPUT_FILE01 = "adj_list_graph01_TEST_OUTPUT.txt"
TEST_DIR = "vertizee/io/tests"


@pytest.mark.usefixtures()
class TestGraphIO:
    def test_digraph_read_adj_list(self):
        g = MultiDiGraph()
        read_adj_list(os.path.join(os.getcwd(), TEST_DIR, DIGRAPH_FILE01), g)

        assert g.vertex_count == 5, "graph should have 5 vertices"
        assert g[1].loops is not None, "v1 should have a loop"
        assert len(g[1].edges_incoming) == 2, "v1 should have 2 incoming edges (including its loop)"
        assert len(g[1].edges_outgoing) == 3, "v1 edges_outgoing should have length 3"
        assert g[1].degree == 6, "deg(v1) should be 6"
        assert g[2].degree == 4, "deg(v2) should be 4"
        assert len(g[2].edges_outgoing) == 2, "v2 should have 2 outgoing edges"
        assert len(g[3].edges_incoming) == 2, "v3 should have 2 incoming edges"
        assert len(g[3].edges_outgoing) == 1, "v3 should have 1 outgoing edge"
        assert len(g[4].edges_outgoing) == 0, "v4 should have 0 outgoing edges"
        v5_loop_edge = g[5].loops.pop()
        assert v5_loop_edge.parallel_edge_count == 1, "v5 should have 2 loops"
        assert len(g[5].edges_incoming) == 1, "v5 should have 1 incoming edge (self-loop)"

        assert g[4, 3] is None, "graph should not have edge (4, 3)"
        assert g[3, 4] is not None, "graph should have edge (3, 4)"

    def test_digraph_write_adj_list(self):
        g = MultiDiGraph()
        read_adj_list(os.path.join(os.getcwd(), TEST_DIR, DIGRAPH_FILE01), g)
        write_adj_list_to_file(os.path.join(os.getcwd(), TEST_DIR, DIGRAPH_OUTPUT_FILE), g)

    def test_multigraph_read_adj_list01(self):
        g = MultiGraph()
        read_adj_list(os.path.join(os.getcwd(), TEST_DIR, GRAPH_FILE01), g)

        v1_loop_edge = g[1].loops.pop()
        assert v1_loop_edge.parallel_edge_count == 1, "v1 should have 2 loops"
        assert g[1].degree == 6, "deg(v1) should be 6"
        assert len(g[2].edges) == 2, (
            "v2 should have 4 incident edges, 3 of which are parallel and stored in one "
            " Edge object."
        )
        assert g[2].degree == 4, "v2 should have degree 4"
        assert g[3].degree == 3, "v3 should have degree 3"
        assert (
            g[3, 2].parallel_edge_count == 2
        ), "there should be 2 parallel edges between (2, 3) [3 edges total]"
        assert (
            g[2, 3].parallel_edge_count == 2
        ), "there should be 2 parallel edges between (2, 3) [3 edges total]"
        assert g[4].degree == 1, "v4 should have degree 1"
        assert g[5].degree == 0, "v5 should have degree 0 (i.e. isolated vertex)"
        assert g[1, 5] is None, "There should be no edge connected to v5"
        assert g[1, 4] is not None, "There should be an edge (1, 4)"
        assert g[1, 2] is not None, "There should be an edge (1, 2)"

    def test_multigraph_read_adj_list02(self):
        g = MultiGraph()
        read_adj_list(os.path.join(os.getcwd(), TEST_DIR, GRAPH_FILE02), g)

        assert (
            g.edge_count == g.edge_count_ignoring_parallel_edges()
        ), "g should have no parallel edges"
        assert g.edge_count == 6, "graph should have 6 edges"
        assert g.vertex_count == 4, "graph should have 4 vertices"
        assert g[1].degree == g[2].degree, "v1 and v2 should have same degree"
        assert g[1].degree == 3, "v1 should have degree 3"
        assert g[2, 4] is not None, "graph should have edge (2, 4)"
        assert g[1, 3] is not None, "graph should have edge (1, 3)"

    def test_multigraph_read_adj_list03(self):
        g = MultiGraph()
        read_adj_list(os.path.join(os.getcwd(), TEST_DIR, GRAPH_FILE03), g)

        assert (
            g.edge_count == g.edge_count_ignoring_parallel_edges()
        ), "g should have no parallel edges"
        assert g.edge_count == 14, "graph should have 14 edges"
        assert g.vertex_count == 8, "graph should have 8 vertices"
        assert g[1].degree < g[2].degree, "v1 and v2 should have same degree"
        assert g[1].degree == 3, "v1 should have degree 3"
        assert g[2].degree == 4, "v2 should have degree 4"
        assert g[2, 4] is not None, "graph should have edge (2, 4)"
        assert g[2, 6] is None, "graph should not have edge (2, 6)"

    def test_multigraph_write_adj_list01(self):
        g = MultiGraph()
        read_adj_list(os.path.join(os.getcwd(), TEST_DIR, GRAPH_FILE01), g)
        write_adj_list_to_file(os.path.join(os.getcwd(), TEST_DIR, GRAPH_OUTPUT_FILE01), g)
