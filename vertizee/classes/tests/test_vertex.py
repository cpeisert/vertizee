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

"""Vertex tests."""

import pytest

from vertizee import DiEdge, DiGraph, Edge, Graph, MultiGraph, Vertex

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip."
)


@pytest.mark.usefixtures()
class TestVertex:
    """Test suite for the Vertex class."""

    def test_vertex_type_and_labels(self):
        """Test the VertexType type alias."""
        g = Graph()
        v0: Vertex = g.add_vertex("0")
        assert v0 == g[0], "Graph should have vertex when VertexType is string."
        v1: Vertex = g.add_vertex(1)
        assert v0 == g["0"], "Graph should have vertex when VertexType is integer."

        g2 = Graph()
        g2_v0: Vertex = g2.add_vertex(v0)
        assert g2_v0 == g2[0], "Graph should have vertex when VertexType is Vertex object."

        assert v0.label == "0", "Vertex 0 should have label '0'."
        assert v1.label == "1", "Vertex 1 should have label '1'."
        assert g2_v0.label == "0", "Vertex 0 in graph 2 should have label '0'."

    def test_vertex_degree(self):
        """Test the degree of vertices."""
        g = MultiGraph()
        v0 = g.add_vertex("0")
        assert v0.degree == 0, "Vertex 0 should have degree 0."
        g.add_edge(1, 1)
        assert g[1].degree == 2, "Vertex with self loop should have degree 2."
        g.add_edge(1, 2)
        assert g[1].degree == 3, "Vertex 1 should have degree 3."
        g.add_edge(2, 1)
        assert g[1].degree == 4, "Vertex 1 should have degree 4."
        assert g[1].indegree == 4, "Undirected graph indegree should match degree."
        assert g[1].outdegree == 4, "Undirected graph outdegree should match degree."

        dg = DiGraph()
        v0 = dg.add_vertex("0")
        assert v0.degree == 0, "Vertex 0 should have degree 0."
        dg.add_edge(1, 1)
        assert dg[1].degree == 2, "Vertex with self loop should have degree 2."
        dg.add_edge(1, 2)
        assert dg[1].degree == 3, "Vertex 1 should have degree 3."
        dg.add_edge(2, 1)
        assert dg[1].degree == 4, "Vertex 1 should have degree 4."

        assert dg[0].indegree == 0, "Vertex should have 0 indegree."
        assert dg[0].outdegree == 0, "Vertex should have 0 outdegree."
        assert dg[1].indegree == 2, "Vertex 1 indegree should be 2."
        assert dg[1].outdegree == 2, "Vertex 1 outdegree should be 2."
        dg.add_edge(3, 3)
        assert dg[3].indegree == 1, "Vertex 3 with self-loop should have indegree 1."
        assert dg[3].outdegree == 1, "Vertex 3 with self-loop should have outdegree 1."

    def test_vertex_loop_edges(self):
        g = Graph([(0, 0)])
        loop: Edge = next(iter(g[0].loops))
        assert (
            loop.vertex1 == g[0] and loop.vertex2 == g[0]
        ), "Loop should have same vertex endpoints."
        edge = next(iter(g[0].edges))
        assert edge == loop, "The only adjacent edge should be the self loop."
        adj_v = next(iter(g[0].adjacent_vertices))
        assert adj_v == g[0], "Vertex 0 should be adjacent to itself."
        assert g[0].delete_loops() == 1, "Should have deleted one loop edge."
        assert (
            len(g[0].adjacent_vertices) == 0
        ), "After loop deletion, there should be no adjacent vertices."
        assert len(g[0].edges) == 0, "After loop deletion, there should be no adjacent edges."

        dg = DiGraph([(1, 1)])
        assert (
            len(dg[1].adjacent_vertices) == 1
        ), "Vertex with directed self-loop should be adjacent to itself."
        assert (
            len(dg[1].adjacent_vertices_incoming) == 1
        ), "Vertex with directed self-loop should be adjacent to itself."
        assert (
            len(dg[1].adjacent_vertices_outgoing) == 1
        ), "Vertex with directed self-loop should be adjacent to itself."

    def test_vertex_adjacent_edges_undirected_graph(self):
        g = Graph([(0, 1)])
        edge: Edge = next(iter(g[0].edges))
        assert (
            edge.vertex1 == g[0] and edge.vertex2 == g[1]
        ), "Edge should have vertex endpoints in instantiation order."

        adj_v = next(iter(g[0].adjacent_vertices))
        assert adj_v == g[1], "Vertex 0 should be adjacent to vertex 1."

        g.add_edge(1, 2)
        adj_search = g[1].get_adj_for_search(parent=g[0])
        assert next(iter(adj_search)) == g[2], "Adj. vertex for searching should be vertex 2."

        assert g[0]._get_edge(1) == g[0, 1], "get_edge should return edge (0, 1)"
        assert (
            g[0].non_loop_edges == g[0].edges
        ), "Non-loop edges should equal edges when there are no self-loops."
        assert g[0].is_incident_edge(g[0, 1]), "Edge (0, 1) should be adjacent to vertex 0."
        g.add_edge(2, 3)
        assert not g[0].is_incident_edge(g[2, 3]), "Edge (2, 3) should not be adjacent to vertex 0."

    def test_vertex_adjacent_edges_directed_graph(self):
        g = DiGraph([(0, 1)])
        edge: DiEdge = next(iter(g[0].edges))
        assert (
            edge.tail == g[0] and edge.head == g[1]
        ), "DiEdge should have vertex endpoints in instantiation order."

        g.add_edge(1, 0)
        g.add_edge(1, 2)
        g.add_edge(3, 1)
        assert g[1].adjacent_vertices == {
            g[0],
            g[2],
            g[3],
        }, "Adj. vertices should include incoming and outgoing edges."

        assert g[1].adjacent_vertices_incoming == {
            g[0],
            g[3],
        }, "Incoming adj. vertices should include incoming edges."

        assert g[1].adjacent_vertices_outgoing == {
            g[0],
            g[2],
        }, "Outgoing adj. vertices should include outgoing edges."

        assert g[1].edges_incoming == {
            g[0, 1],
            g[3, 1],
        }, "Incoming edges should include (0, 1) and (3, 1)."

        assert g[1].edges_outgoing == {
            g[1, 0],
            g[1, 2],
        }, "Outgoing edges should include (1, 0) and (1, 2)."

        assert (
            g[1].get_adj_for_search() == g[1].adjacent_vertices_outgoing
        ), "Adj. vertices for digraph search should be the adjacent outgoing edges."

        assert (
            g[1].get_adj_for_search(reverse_graph=True) == g[1].adjacent_vertices_incoming
        ), "Adj. vertices for search should be the adjacent incoming edges for reverse graph."
        assert g[1].is_incident_edge(g[1, 0]), "DiEdge (1, 0) should be incident to vertex 1."
        g.add_edge(2, 3)
        assert not g[1].is_incident_edge(
            g[2, 3]
        ), "DiEdge (2, 3) should not be incident to vertex 1."

    def test_vertex_attr_dictionary(self):
        g = Graph()
        g.add_vertex(0)
        g[0].attr["weight"] = 100
        g[0]["color"] = "blue"
        assert g[0]["weight"] == 100, 'Vertex attribute "weight" should be 100.'
        assert g[0].attr["color"] == "blue", 'Vertex attribute "color" should be blue.'
