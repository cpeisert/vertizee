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

"""Edge tests."""

import pytest

from vertizee import DiEdge, DiGraph, Edge, Graph, MultiGraph, MultiDiGraph

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip."
)


@pytest.mark.usefixtures()
class TestEdge:
    """Test suite for the Edge class."""

    def test_edge_str_representation_and_weight(self):
        g = MultiGraph()
        e01: Edge = g.add_edge(0, 1)
        assert str(e01) == "(0, 1)", "Edge string representation should be (0, 1)."
        e02: Edge = g.add_edge(2, 0)
        assert str(e02) == "(0, 2)", "Edge string representation should be (0, 2)."
        g.add_edge(1, 0)
        assert (
            str(e01) == "(0, 1), (0, 1)"
        ), "Multi-edge string representation should be (0, 1), (0, 1)."

        # Graph treated as weighted after adding edge with non-default weight.
        g.add_edge(2, 3, 10.5)
        assert str(g[2, 3]) == "(2, 3, 10.5)", "Edge string representation should be (2, 3, 10.5)."
        assert str(e02) == "(0, 2, 1.0)", "Edge string representation should be (0, 2, 1.0)."
        assert g[2, 3].weight == 10.5, "Edge (2, 3) weight should be 10.5."
        assert g[0, 1].weight == 1, "Default edge weight should be 1.0."

    def test_loop_edges(self):
        g = MultiGraph()
        e00: Edge = g.add_edge(0, 0)
        assert e00.is_loop(), "Edge (0, 0) should be a loop."
        assert str(e00) == "(0, 0)", "String representation should be (0, 0)."
        assert e00.vertex1 == 0 and e00.vertex2 == 0, "Both loop endpoints should be '0'."
        g.add_edge(0, 1)
        assert not g[0, 1].is_loop(), "Edge (0, 1) should not be a loop."

    def test_parallel_edges(self):
        g = MultiGraph()
        g.add_edges_from([(0, 1), (0, 1)])
        assert (
            g[0, 1].multiplicity == 2
        ), "Multi-edge with one parallel edge should have multiplicity 2."
        assert g[0, 1].parallel_edge_count == 1, "Edge should have one parallel edge."
        assert len(g[0, 1].parallel_edge_weights) == 1, "There should be one parallel edge weight."
        g.add_edges_from([(1, 2, 4.5), (1, 2, 7.0)])
        assert len(g[1, 2].parallel_edge_weights) == 1, "There should be one parallel edge weight."

    def test_vertex_order(self):
        g = Graph()
        g.add_edge(2, 1)
        assert g[2, 1] == g[1, 2], "Vertex order should not matter in an undirected graph."
        assert g[1, 2].vertex1 == 2, "Vertex1 should be 2"
        assert g[1, 2].vertex2 == 1, "Vertex2 should be 1"

    def test_edge_attr_dictionary(self):
        g = Graph()
        g.add_edge(0, 1)
        g[0, 1].attr["weight"] = 100
        g[0, 1]["color"] = "blue"
        assert g[0, 1]["weight"] == 100, 'Edge attribute "weight" should be 100.'
        assert g[0, 1].attr["color"] == "blue", 'Edge attribute "color" should be blue.'


@pytest.mark.usefixtures()
class TestDiEdge:
    """Test suite for the DiEdge class."""

    def test_edge_str_representation_and_weight(self):
        g = MultiDiGraph()
        e01: DiEdge = g.add_edge(0, 1)
        assert str(e01) == "(0, 1)", "DiEdge string representation should be (0, 1)."
        e02: DiEdge = g.add_edge(2, 0)
        assert str(e02) == "(2, 0)", "DiEdge string representation should be (2, 0)."
        g.add_edge(1, 0)
        assert (
            str(e01) == "(0, 1)"
        ), "DiEdge string representation should be (0, 1) after adding edge (1, 0)."
        g.add_edge(0, 1)
        assert (
            str(e01) == "(0, 1), (0, 1)"
        ), "DiEdge string representation should be (0, 1), (0, 1) after adding parallel edge."

        # DiGraph treated as weighted after adding edge with non-default weight.
        g.add_edge(2, 3, 10.5)
        assert (
            str(g[2, 3]) == "(2, 3, 10.5)"
        ), "DiEdge string representation should be (2, 3, 10.5)."
        assert str(e02) == "(2, 0, 1.0)", "DiEdge string representation should be (2, 0, 1.0)."
        assert g[2, 3].weight == 10.5, "DiEdge (2, 3) weight should be 10.5."
        assert g[0, 1].weight == 1, "Default edge weight should be 1.0."

    def test_loop_edges(self):
        g = MultiDiGraph()
        e00: DiEdge = g.add_edge(0, 0)
        assert e00.is_loop(), "Edge (0, 0) should be a loop."
        assert str(e00) == "(0, 0)", "String representation should be (0, 0)."
        assert e00.tail == 0 and e00.head == 0, "Both the tail and head endpoints should be '0'."
        g.add_edge(0, 1)
        assert not g[0, 1].is_loop(), "Edge (0, 1) should not be a loop."

    def test_parallel_edges(self):
        g = MultiDiGraph()
        g.add_edges_from([(0, 1), (0, 1)])
        assert (
            g[0, 1].multiplicity == 2
        ), "Multi-edge with one parallel edge should have multiplicity 2."
        assert g[0, 1].parallel_edge_count == 1, "DiEdge should have one parallel edge."
        assert len(g[0, 1].parallel_edge_weights) == 1, "There should be one parallel edge weight."
        g.add_edges_from([(1, 2, 4.5), (1, 2, 7.0)])
        assert len(g[1, 2].parallel_edge_weights) == 1, "There should be one parallel edge weight."

    def test_vertex_order(self):
        g = DiGraph()
        g.add_edge(2, 1)
        g.add_edge(1, 2)
        assert g[2, 1] != g[1, 2], "Vertex order should define a different edge in a digraph."
        assert g[2, 1].vertex1 == 2 and g[2, 1].tail == 2, "Tail vertex should be 2"
        assert g[2, 1].vertex2 == 1 and g[2, 1].head == 1, "Head vertex should be 1"

    def test_edge_attr_dictionary(self):
        g = DiGraph()
        g.add_edge(0, 1)
        g[0, 1].attr["weight"] = 100
        g[0, 1]["color"] = "blue"
        assert g[0, 1]["weight"] == 100, 'DiEdge attribute "weight" should be 100.'
        assert g[0, 1].attr["color"] == "blue", 'DiEdge attribute "color" should be blue.'
