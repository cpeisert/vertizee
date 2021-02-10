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

"""Tests for the primitives parsing."""
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

from vertizee import Graph, DiGraph, MultiGraph

from vertizee.classes import edge as edge_module
from vertizee.classes import primitives_parsing as pp_module


class TestVertexData:
    """Tests for the VertexData class."""

    def test_from_vertex_obj(self):
        g = Graph()
        g.add_vertex(1, color="blue")

        vertex_data = pp_module.VertexData.from_vertex_obj(g[1])
        assert vertex_data.label == "1", "label should be '1'"
        assert vertex_data.attr["color"] == "blue", "should have 'color' attribute set to 'blue'"
        assert vertex_data.vertex_object, "should have reference to vertex object"

    def test_attr(self):
        vertex_data = pp_module.VertexData("s")
        assert vertex_data.label == "s", "label should be 's'"
        assert not vertex_data.attr, "there should be no attributes by default"
        assert not vertex_data.vertex_object, "there should be no vertex object reference"
        vertex_data.attr["color"] = "blue"
        assert vertex_data.attr["color"] == "blue", "should have 'color' attribute set to 'blue'"


class TestEdgeData:
    """Tests for the EdgeData class."""

    def test_attr(self):
        v1 = pp_module.VertexData("1")
        v2 = pp_module.VertexData("2")
        edge_data = pp_module.EdgeData(v1, v2)
        assert not edge_data.attr, "there should be no attributes by default"
        assert not edge_data.edge_object, "there should be no edge object reference"
        edge_data.attr["color"] = "blue"
        assert edge_data.attr["color"] == "blue", "should have 'color' attribute set to 'blue'"

        g = Graph()
        g.add_edge(1, 2)
        edge_data2 = pp_module.EdgeData.from_edge_obj(g.get_edge(1, 2))
        assert not edge_data2.attr, "there should be no attributes from edge (1, 2)"
        edge_data2.attr["color"] = "blue"
        assert edge_data2.attr["color"] == "blue", "should have 'color' attribute set to 'blue'"

    def test_get_label(self):
        v1 = pp_module.VertexData("1")
        v2 = pp_module.VertexData("2")
        edge_data1 = pp_module.EdgeData(v2, v1)
        assert edge_data1.get_label(is_directed=False) == "(1, 2)", "label should be (1, 2)"
        assert edge_data1.get_label(is_directed=True) == "(2, 1)", "label should be (2, 1)"

        g = DiGraph()
        g.add_edge(2, 1)
        edge_data2 = pp_module.EdgeData.from_edge_obj(g.get_edge(2, 1))
        assert edge_data2.get_label(is_directed=False) == "(1, 2)", "label should be (1, 2)"
        assert edge_data2.get_label(is_directed=True) == "(2, 1)", "label should be (2, 1)"

    def test_vertex1_vertex2(self):
        v1 = pp_module.VertexData("1")
        v2 = pp_module.VertexData("2")
        edge_data1 = pp_module.EdgeData(v2, v1)
        assert edge_data1.vertex1 == v2, "vertex1 should be v2"
        assert edge_data1.vertex2 == v1, "vertex2 should be v1"

        g = DiGraph()
        g.add_edge(2, 1)
        edge_data2 = pp_module.EdgeData.from_edge_obj(g.get_edge(2, 1))
        assert edge_data2.vertex1.label == "2", "vertex1 should be 2"
        assert edge_data2.vertex2.label == "1", "vertex2 should be 1"

    def test_weight(self):
        g = Graph()
        g.add_edge(1, 2)
        edge_data1 = pp_module.EdgeData.from_edge_obj(g.get_edge(1, 2))
        assert edge_data1.weight == edge_module.DEFAULT_WEIGHT, "weight should be default"

        g.add_edge(3, 4, 9.5)
        edge_data2 = pp_module.EdgeData.from_edge_obj(g.get_edge(3, 4))
        assert edge_data2.weight == 9.5, "weight should be 9.5"

        v5 = pp_module.VertexData("5")
        v6 = pp_module.VertexData("6")
        edge_data3 = pp_module.EdgeData(v5, v6)
        assert edge_data3.weight == edge_module.DEFAULT_WEIGHT, "weight should be default"


class TestPrimitivesParsingModuleFunctions:
    """Tests for functions defined in the primitive parsing module."""

    def test_parse_edge_type(self):
        g = Graph()
        g.add_edge(1, 2)
        edge_data1 = pp_module.parse_edge_type(g.get_edge(1, 2))
        assert edge_data1.vertex1.label == "1"
        assert edge_data1.vertex2.label == "2"

        mg = MultiGraph([(3, 4), (3, 4)])
        edge_data2 = pp_module.parse_edge_type(mg.get_edge(3, 4))
        assert edge_data2.vertex1.label == "3"
        assert edge_data2.vertex2.label == "4"

        edge_tuple = (1, 2)
        edge_data3 = pp_module.parse_edge_type(edge_tuple)
        assert edge_data3.vertex1.label == "1"
        assert edge_data3.vertex2.label == "2"

        edge_tuple_weighted = (3, 4, 3.5)
        edge_data4 = pp_module.parse_edge_type(edge_tuple_weighted)
        assert edge_data4.vertex1.label == "3"
        assert edge_data4.vertex2.label == "4"
        assert edge_data4.weight == 3.5

        edge_tuple_attr = (4, 5, {"color": "blue"})
        edge_data5 = pp_module.parse_edge_type(edge_tuple_attr)
        assert edge_data5.vertex1.label == "4"
        assert edge_data5.vertex2.label == "5"
        assert edge_data5.weight == edge_module.DEFAULT_WEIGHT
        assert edge_data5.attr["color"] == "blue"

        edge_tuple_weighted_attr = (6, 7, 9.5, {"k": "v"})
        edge_data6 = pp_module.parse_edge_type(edge_tuple_weighted_attr)
        assert edge_data6.vertex1.label == "6"
        assert edge_data6.vertex2.label == "7"
        assert edge_data6.weight == 9.5
        assert edge_data6.attr["k"] == "v"

    def test_parse_graph_primitive(self):
        edge_tuple_attr = (1, 2, {"color": "blue"})
        data1: pp_module.ParsedEdgeAndVertexData = pp_module.parse_graph_primitive(edge_tuple_attr)
        assert data1.edges[0].vertex1.label == "1"
        assert data1.edges[0].vertex2.label == "2"
        assert data1.edges[0].attr["color"] == "blue"
        assert not data1.vertices, "there should be no parsed vertices"

        vertex_tuple_attr = (3, {"mass": 4.5})
        data2 = pp_module.parse_graph_primitive(vertex_tuple_attr)
        assert data2.vertices[0].label == "3"
        assert data2.vertices[0].attr["mass"] == 4.5
        assert not data2.edges, "there should be no parsed edges"

    def test_parse_graph_primitives_from(self):
        primitives = [1, "s", (2, {"mass": 3.5}), (2, 3), (3, 4, 3.5), (4, 5, {"color": "blue"})]
        data: pp_module.ParsedEdgeAndVertexData = pp_module.parse_graph_primitives_from(primitives)
        assert len(data.vertices) == 3, "there should be 3 parsed vertices"
        assert len(data.edges) == 3, "there should be 3 parsed edges"
        assert set(v.label for v in data.vertices) == {"1", "s", "2"}

    def test_parse_vertex_type(self):
        g = Graph()
        g.add_vertex(1, mass=5.5)
        vertex_data1 = pp_module.parse_vertex_type(g[1])
        assert vertex_data1.label == "1"
        assert vertex_data1.attr["mass"] == 5.5

        mg = MultiGraph([(3, 4), (3, 4)])
        vertex_data2 = pp_module.parse_vertex_type(mg[4])
        assert vertex_data2.label == "4"

        vertex_data3 = pp_module.parse_vertex_type(3)
        assert vertex_data3.label == "3"

        vertex_data4 = pp_module.parse_vertex_type("s")
        assert vertex_data4.label == "s"

        vertex_data5 = pp_module.parse_vertex_type(("t", {"mass": 42}))
        assert vertex_data5.label == "t"
        assert vertex_data5.attr["mass"] == 42.0
