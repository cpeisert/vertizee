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
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

import pytest

from vertizee import (
    Graph,
    Vertex,
    DiGraph,
    DiVertex,
    MultiGraph,
    MultiVertex,
    MultiDiGraph,
    MultiDiVertex,
    VertizeeException,
)

from vertizee.classes import vertex as vertex_module


class TestVertexModuleFunctions:
    """Tests for functions defined in the vertex module."""

    def test_get_vertex_label(self):
        g = Graph()
        v_obj: Vertex = g.add_vertex(1)
        assert vertex_module.get_vertex_label(v_obj) == "1", "vertex object label should be '1'"
        assert vertex_module.get_vertex_label(10) == "10", "int vertex label should be '10'"
        assert vertex_module.get_vertex_label("s") == "s", "str vertex label should be 's'"
        assert (
            vertex_module.get_vertex_label(("s", {"color": "blue"})) == "s"
        ), "vertex tuple should have label 's'"

    def test_is_vertex_type(self):
        g = Graph()
        v_obj: Vertex = g.add_vertex(1)
        assert vertex_module.is_vertex_type(v_obj), "Vertex object should be a VertexType"

        g2 = DiGraph()
        v_obj: DiVertex = g2.add_vertex(1)
        assert vertex_module.is_vertex_type(v_obj), "DiVertex object should be a VertexType"

        g3 = MultiGraph()
        v_obj: MultiVertex = g3.add_vertex(1)
        assert vertex_module.is_vertex_type(v_obj), "MultiVertex object should be a VertexType"

        g4 = MultiDiGraph()
        v_obj: MultiDiVertex = g4.add_vertex(1)
        assert vertex_module.is_vertex_type(v_obj), "MultiDiVertex object should be a VertexType"

        assert vertex_module.is_vertex_type(10), "int vertex label should be a VertexType"
        assert vertex_module.is_vertex_type("s"), "str vertex label should be a VertexType"
        assert vertex_module.is_vertex_type(
            ("s", {"color": "blue"})
        ), "vertex tuple should be a VertexType"
        assert not vertex_module.is_vertex_type(10.99), "float should not be a VertexType"
        assert not vertex_module.is_vertex_type(("s", "t")), "edge tuple should not be a VertexType"
        assert not vertex_module.is_vertex_type(
            ("s", "t", 4.5)
        ), "edge tuple with edge weight should not be a VertexType"
        g.add_edge("s", "t")
        assert not vertex_module.is_vertex_type(
            g["s", "t"]
        ), "edge object should not be a VertexType"


class TestVertexBase:
    """Tests for VertexBase features shared by all vertex classes."""

    def test_comparison_operators(self):
        g = Graph()
        v1 = g.add_vertex("a")
        v2 = g.add_vertex("b")

        assert v1 < v2, "v1 should be less than v2"
        assert v1 <= v2, "v1 should be less than or equal to v2"
        assert v2 > v1, "v2 should be greater than v1"
        assert v2 >= v1, "v2 should be greater than or equal to v1"
        assert v1 == "a", "v1 should be equal to a vertex represented by label 'a'"
        assert v1 < "b", "v1 should be less than a vertex represented by label 'b'"
        assert v1 > 1, "v1 should be great than a vertex represented by label 1"
        assert v1 < ("b", {"color": "blue"}), "v1 should be less than vertex tuple 'b'"

    def test_repr_str_and_label(self):
        g = Graph()
        v1 = g.add_vertex("a")
        v2 = g.add_vertex(2)
        assert v1.label == "a", "v1 label should be 'a'"
        assert v2.label == "2", "v2 label should be '2'"
        assert v1.__repr__() == v1.label, "vertex __repr__() should be its label"
        assert v1.__str__() == v1.label, "vertex __str__() should be its label"
        assert v1.__repr__() == v1.__str__(), "vertex __repr__() should match __str__()"

    def test_adj_vertices(self):
        g = Graph()
        g.add_vertex(1)
        assert not g[1].adj_vertices(), "vertex 1 should have no adjacent vertices"

        g.add_edge(1, 2)
        assert next(iter(g[1].adj_vertices())) == 2, "vertex 1 should adjacent to vertex 2"

        g.add_edge(1, 3)
        assert len(g[1].adj_vertices()) == 2, "vertex 1 should be adjacent to vertices 2 and 3"
        assert next(iter(g[3].adj_vertices())) == g[1], "vertex 3 should be adjacent to vertex 1"

        g.add_edge(2, 4)
        assert g[4] not in g[1].adj_vertices(), "vertex 1 should not be adjacent to vertex 4"
        assert g[2] not in g[2].adj_vertices(), "vertex 2 should not be adjacent to itself"

        g.add_edge(4, 4)
        assert g[4] in g[4].adj_vertices(), "vertex 4 should be adjacent to itself"

    def test_attr_dictionary(self):
        g = Graph()
        v1 = g.add_vertex("a")
        assert v1._attr is None, "attr dictionary should be None by default"
        v1.attr["weight"] = 1000
        assert v1.attr["weight"] == 1000, "v1 should have 'weight' attribute set to 1000"
        assert v1["weight"] == 1000, "'weight' attribute should be accessible with index getter"
        v1["color"] = "blue"
        assert v1["color"] == "blue", "v1 should have color attribute set to 'blue'"

        with pytest.raises(KeyError):
            _ = v1["unknown_key"]

    def test_degree(self):
        g = Graph()
        v0 = g.add_vertex(0)
        assert v0.degree == 0, "vertex 0 should have degree 0"

        g.add_edge(1, 1)
        assert g[1].degree == 2, "vertex 1 with self loop should have degree 2"

        g.add_edge(1, 2)
        assert g[1].degree == 3, "vertex 1 should have degree 3"

    def test_is_isolated(self):
        g = Graph()
        v1 = g.add_vertex(1)
        assert v1.is_isolated(), "v1 should be isolated"

        g.add_edge(v1, v1)
        assert v1.loop_edge, "v1 should have a self loop"
        assert not v1.is_isolated(), "vertex with self-loop should not be considered isolated"
        assert v1.is_isolated(
            ignore_self_loops=True
        ), "vertex with self-loop should be considered semi-isolated"

        g.add_edge(v1, 2)
        assert not v1.is_isolated(), "vertex connected to a different vertex should not be isolated"

        v1.remove_incident_edges()
        assert v1.is_isolated(), "vertex should be isolated after removing incident edges"

    def test_remove_and_edge_removal(self):
        g = Graph()
        v1 = g.add_vertex("a")
        assert g.vertex_count == 1, "graph should have one vertex"
        v1.remove()
        assert g.vertex_count == 0, "graph should have zero vertices after vertex self removal"
        v2 = g.add_vertex("b")
        g.add_edge("b", "c")
        g.add_edge("b", "b")

        with pytest.raises(VertizeeException):
            v2.remove()

        assert len(v2.incident_edges()) == 2, "v2 should have two incident edges"
        v2.remove_loops()
        assert v2.loop_edge is None, "v2 should not have a loop edge after removal"
        assert len(v2.incident_edges()) == 1, "v2 should have one incident after removing loop"
        v2.remove_incident_edges()
        assert len(v2.incident_edges()) == 0, "v2 should not have any incident edges after removal"


class TestVertex:
    """Tests for ``Vertex`` class (concrete implementation ``_Vertex``)."""

    def test_issubclass_and_isinstance(self):
        g = Graph()
        v1 = g.add_vertex(1)
        assert isinstance(
            v1, vertex_module.VertexBase
        ), "v1 should be an instance of superclass VertexBase"
        assert isinstance(v1, Vertex), "v1 should be a Vertex instance"
        assert issubclass(Vertex, vertex_module.VertexBase), "Vertex should be VertexBase subclass"

    def test_incident_edges_and_loop_edge(self):
        g = Graph()
        g.add_vertex(0)
        assert not g[0].loop_edge, "vertex 0 should not have a loop edge"

        g.add_edge(1, 1)
        assert g[1].incident_edges() == {g[1, 1]}, "vertex 1 should have self loop as incident edge"
        assert g[1].loop_edge, "vertex 1 should have a self loop"

        g.add_edge(1, 2)
        assert len(g[1].incident_edges()) == 2, "vertex 1 should have two incident edges"


class TestDiVertex:
    """Tests for ``DiVertex`` class (concrete implementation ``_DiVertex``)."""

    def test_issubclass_and_isinstance(self):
        g = DiGraph()
        v1 = g.add_vertex(1)
        assert isinstance(
            v1, vertex_module.VertexBase
        ), "v1 should be an instance of superclass VertexBase"
        assert isinstance(v1, DiVertex), "v1 should be a DiVertex instance"
        assert issubclass(
            DiVertex, vertex_module.VertexBase
        ), "DiVertex should be VertexBase subclass"

    def test_adj_vertices(self):
        g = DiGraph([(1, 1)])
        assert g[1].adj_vertices() == {
            g[1]
        }, "vertex with directed self-loop should be adjacent to itself"
        assert g[1].adj_vertices_incoming() == {
            g[1]
        }, "vertex with directed self-loop should connect to itself via an incoming edge"
        assert g[1].adj_vertices_outgoing() == {
            g[1]
        }, "vertex with directed self-loop should connect to itself via an outgoing edge"

        g.add_edge(1, 2)
        assert g[1].adj_vertices() == {
            g[1],
            g[2],
        }, "vertex 1 should be adjacent to itself and vertex 2"
        assert g[1].adj_vertices_incoming() == {
            g[1]
        }, "vertex 1 should only have itself as incoming adjacent vertex"
        assert g[1].adj_vertices_outgoing() == {
            g[1],
            g[2],
        }, "vertex 1 should only have itself and vertex 2 as outgoing adjacent vertex"

    def test_degree(self):
        g = DiGraph()
        v0 = g.add_vertex("0")
        assert v0.indegree == 0, "vertex 0 should have indegree 0"
        assert v0.outdegree == 0, "vertex 0 should have outdegree 0"

        g.add_edge(1, 1)
        assert g[1].degree == 2, "vertex with self loop should have degree 2"
        assert g[1].indegree == 1, "vertex with self loop should have indegree 1"
        assert g[1].outdegree == 1, "vertex with self loop should have outdegree 1"

        g.add_edge(1, 2)
        assert g[1].outdegree == 2, "vertex 1 should have outdegree 2"
        assert g[1].indegree == 1, "vertex with self loop and outgoing edge should have indegree 1"
        assert g[2].outdegree == 0, "vertex with no outgoing edges should have outdegree 0"
        assert g[2].indegree == 1, "vertex with one incoming edge should have indegree 1"
        g.add_edge(3, 1)
        assert g[1].indegree == 2, "vertex 1 should have indegree 2"

    def test_incident_edges(self):
        g = DiGraph()
        g.add_edge(1, 0)
        g.add_edge(1, 2)
        g.add_edge(3, 1)
        assert g[1].incident_edges() == {
            g[1, 0],
            g[1, 2],
            g[3, 1],
        }, "incident edges should be (1, 0), (1, 2), and (3, 1)"

        assert g[1].incident_edges_incoming() == {g[3, 1]}, "incoming edge should be (3, 1)"

        assert g[1].incident_edges_outgoing() == {
            g[1, 0],
            g[1, 2],
        }, "outgoing edges should include (1, 0) and (1, 2)"

    def test_loop_edge(self):
        g = DiGraph()
        g.add_vertex(0)
        assert not g[0].loop_edge, "vertex 0 should not have a loop edge"

        g.add_edge(1, 1)
        assert g[1].incident_edges() == {g[1, 1]}, "vertex 1 should have self loop as incident edge"
        assert g[1].loop_edge, "vertex 1 should have a self loop"


class TestMultiVertex:
    """Tests for ``MultiVertex`` class (concrete implementation ``_MultiVertex``)."""

    def test_issubclass_and_isinstance(self):
        g = MultiGraph()
        v1 = g.add_vertex(1)
        assert isinstance(
            v1, vertex_module.VertexBase
        ), "v1 should be an instance of superclass VertexBase"
        assert isinstance(v1, MultiVertex), "v1 should be a MultiVertex instance"
        assert issubclass(
            MultiVertex, vertex_module.VertexBase
        ), "MultiVertex should be VertexBase subclass"

    def test_adj_vertices(self):
        g = MultiGraph([(1, 1), (1, 1), (2, 3), (2, 3)])
        assert g[1].adj_vertices() == {g[1]}, "vertex with self-loops should be adjacent to itself"
        assert g[2].adj_vertices() == {
            g[3]
        }, "vertex 2 (with parallel connections) should be adjacent to vertex 3"

    def test_degree(self):
        g = MultiGraph()
        v0 = g.add_vertex(0)
        assert v0.degree == 0, "vertex 0 should have degree 0"

        g.add_edge(1, 1)
        assert g[1].degree == 2, "vertex with self loop should have degree 2"
        g.add_edge(1, 1)
        assert g[1].degree == 4, "vertex with two parallel self loops should have degree 4"

        g.add_edge(2, 3)
        assert g[2].degree == 1, "vertex 2 should have degree 1"
        g.add_edge(2, 3)
        assert g[2].degree == 2, "vertex 2 with two parallel edge connections should have degree 2"

    def test_incident_edges_and_loop_edge(self):
        g = MultiGraph()
        g.add_vertex(0)
        assert not g[0].loop_edge, "vertex 0 should not have a loop edge"

        g.add_edges_from([(1, 1), (1, 1), (2, 3), (2, 3)])
        assert g[1].incident_edges() == {g[1, 1]}, "vertex 1 should have self loop as incident edge"
        assert g[1].loop_edge, "vertex 1 should have a self loop"

        assert len(g[2].incident_edges()) == 1, "vertex 2 should have one incident multiedge"
        assert (
            next(iter(g[2].incident_edges())).multiplicity == 2
        ), "multiedge (2, 3) should have multiplicity 2"


class TestMultiDiVertex:
    """Tests for ``MultiDiVertex`` class (concrete implementation ``_MultiDiVertex``)."""

    def test_issubclass_and_isinstance(self):
        g = MultiDiGraph()
        v1 = g.add_vertex(1)
        assert isinstance(
            v1, vertex_module.VertexBase
        ), "v1 should be an instance of superclass VertexBase"
        assert isinstance(v1, MultiDiVertex), "v1 should be a MultiDiVertex instance"
        assert issubclass(
            MultiDiVertex, vertex_module.VertexBase
        ), "MultiDiVertex should be VertexBase subclass"

    def test_adj_vertices(self):
        g = MultiDiGraph([(1, 1), (1, 1), (2, 3), (2, 3)])
        assert g[1].adj_vertices() == {
            g[1]
        }, "vertex with directed self-loop should be adjacent to itself"
        assert g[1].adj_vertices_incoming() == {
            g[1]
        }, "vertex with directed self-loop should connect to itself via an incoming edge"
        assert g[1].adj_vertices_outgoing() == {
            g[1]
        }, "vertex with directed self-loop should connect to itself via an outgoing edge"

        assert g[2].adj_vertices() == {g[3]}, "vertex 2 should be adjacent to vertex 3"
        assert not g[
            2
        ].adj_vertices_incoming(), "vertex 2 should have no incoming adjacent vertices"
        assert g[2].adj_vertices_outgoing() == {
            g[3]
        }, "vertex 2 should have vertex 3 as outgoing adjacent vertex"

    def test_degree(self):
        g = MultiDiGraph([(1, 1), (1, 1), (2, 3), (2, 3)])
        assert g[1].degree == 4, "vertex with two self loops should have degree 4"
        assert g[1].indegree == 2, "vertex with two loops should have indegree 2"
        assert g[1].outdegree == 2, "vertex with two loops should have outdegree 2"

        assert g[2].outdegree == 2, "vertex 2 should have outdegree 2"
        assert g[2].indegree == 0, "vertex 2 should have indegree 0"
        assert g[3].indegree == 2, "vertex 3 should have indegree 2"

    def test_incident_edges(self):
        g = MultiDiGraph([(1, 1), (1, 1), (2, 3), (2, 3)])
        assert g[1].incident_edges() == {g[1, 1]}, "vertex 1 should be incident on (1, 1)"
        assert g[1].incident_edges_incoming() == {g[1, 1]}, "incoming edge should be (1, 1)"
        assert g[1].incident_edges_outgoing() == {g[1, 1]}, "outgoing edge should be (1, 1)"

        assert g[2].incident_edges_outgoing() == {g[2, 3]}, "outgoing edges should (2, 3)"
        assert not g[2].incident_edges_incoming(), "should be no incoming edges"

    def test_loop_edge(self):
        g = MultiDiGraph()
        g.add_vertex(0)
        assert not g[0].loop_edge, "vertex 0 should not have a loop edge"

        g.add_edge(1, 1)
        g.add_edge(1, 1)
        assert g[1].incident_edges() == {g[1, 1]}, "vertex 1 should have self loop as incident edge"
        assert g[1].loop_edge, "vertex 1 should have a self loop"
        assert g[1].loop_edge.multiplicity == 2, "vertex 1 should have two loops"
