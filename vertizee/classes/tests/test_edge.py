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
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

import pytest

from vertizee import (
    Graph,
    Edge,
    DiGraph,
    DiEdge,
    EdgeView,
    DiEdgeView,
    MultiGraph,
    MultiEdge,
    MultiDiGraph,
    MultiDiEdge
)

from vertizee.classes import edge as edge_module


class TestEdgeModuleFunctions:
    """Tests for functions defined in edge module."""

    def test_create_edge_label(self):
        g = Graph()
        v1 = g.add_vertex(1)
        v2 = g.add_vertex(2)
        assert (
            edge_module.create_edge_label(v1, v1, is_directed=g.is_directed()) == "(1, 1)"
        ), "loop edge label should be (1, 1)"
        assert (
            edge_module.create_edge_label(v1, v2, is_directed=g.is_directed()) == "(1, 2)"
        ), "edge label should be (1, 2)"
        assert (
            edge_module.create_edge_label(v2, v1, is_directed=g.is_directed()) == "(1, 2)"
        ), "edge label should be (1, 2)"
        assert (
            edge_module.create_edge_label(5, 9, is_directed=g.is_directed()) == "(5, 9)"
        ), "edge label should be (5, 9)"
        assert (
            edge_module.create_edge_label("t", "s", is_directed=g.is_directed()) == "(s, t)"
        ), "edge label should be (s, t)"

        g2 = DiGraph()
        v3 = g2.add_vertex(3)
        v4 = g2.add_vertex(4)
        assert (
            edge_module.create_edge_label(v3, v4, is_directed=g2.is_directed()) == "(3, 4)"
        ), "edge label should be (3, 4)"
        assert (
            edge_module.create_edge_label(v4, v3, is_directed=g2.is_directed()) == "(4, 3)"
        ), "edge label should be (4, 3)"
        assert (
            edge_module.create_edge_label("t", "s", is_directed=g2.is_directed()) == "(t, s)"
        ), "edge label should be (t, s)"


class Test_EdgeConnectionData:
    """Tests for _EdgeConnectionData class."""

    def test_edge_connection_data(self):
        ecd = edge_module._EdgeConnectionData(weight=99.9)
        assert ecd._attr is None, "attributes property should be None"
        assert not ecd.has_attributes_dict(), "attr dictionary should not be initialized"
        ecd.attr["color"] = "blue"
        assert (
            ecd.has_attributes_dict()
        ), "attributes dict should be initialized after setting an attribute"
        assert ecd.attr["color"] == "blue", "edge should have color attribute set to 'blue'"


class TestEdgeView:
    """Tests for edge view classes: EdgeViewBase, EdgeView, DiEdgeView."""

    def test_edgeview(self):
        mg = MultiGraph([(1, 1), (1, 1), (3, 2), (2, 3)])
        loops: MultiEdge = mg[1, 1]
        loop_view: EdgeView = next(loops.connections())
        assert loop_view.is_loop(), "edge connection should be a loop"
        assert loop_view.label == "(1, 1)", "label should be '(1, 1)'"
        assert (
            str(loop_view) == "EdgeView(1, 1)"
        ), "__str__() should return '<class name><connection label>'"
        assert loop_view.vertex1 == 1, "vertex1 should have label 1"
        assert loop_view.vertex2 == 1, "vertex2 should have label 1"
        assert loop_view.weight == edge_module.DEFAULT_WEIGHT, "edge should have default weight"
        assert not loop_view.has_attributes_dict(), "edge attr dict should not be instantiated"
        loop_view["color"] = "blue"
        assert loop_view["color"] == "blue", "connection should have 'color' attribute"
        assert loop_view.attr["color"] == "blue", "connection should have 'color' attribute"
        assert (
            loop_view.has_attributes_dict()
        ), "edge attr dictionary should be instantiated after using property accessor"
        with pytest.raises(KeyError):
            _ = loop_view["unknown_key"]

        multiedge = mg[2, 3]
        view = next(multiedge.connections())
        assert view.vertex1 == mg[3], "vertex1 should be vertex 2"
        assert view.vertex2 == mg[2], "vertex2 should be vertex 3"
        assert view.label == "(2, 3)", "label should be '(2, 3)'"

    def test_diedgeview(self):
        mdg = MultiDiGraph([(3, 2), (3, 2)])
        multiedge: MultiDiEdge = mdg[3, 2]
        view: DiEdgeView = next(multiedge.connections())
        assert view.vertex1 == mdg[3], "vertex1 should be vertex 2"
        assert view.vertex2 == mdg[2], "vertex2 should be vertex 3"
        assert view.tail == view.vertex1, "tail should be synonym for vertex1"
        assert view.head == view.vertex2, "head should be synonym for vertex2"
        assert view.label == "(3, 2)", "label should be '(3, 2)'"
        assert (
            str(view) == "DiEdgeView(3, 2)"
        ), "__str__() should return '<class name><connection label>'"
        assert view.weight == edge_module.DEFAULT_WEIGHT, "edge should have default weight"
        assert not view.is_loop(), "edge connection should not be a loop"
        assert not view.has_attributes_dict(), "edge attr dict should not be instantiated"
        view["color"] = "blue"
        assert view["color"] == "blue", "connection should have 'color' attribute"
        assert view.attr["color"] == "blue", "connection should have 'color' attribute"
        assert (
            view.has_attributes_dict()
        ), "edge attr dictionary should be instantiated after using property accessor"


class TestEdgeBase:
    """Tests for EdgeBase features shared by all single-connection edge classes."""

    def test__eq__(self):
        g1 = Graph([(1, 2), (3, 4, 2.5)])
        g2 = Graph([(1, 2), (3, 4, 5.0)])

        assert g1[1, 2] == g2[1, 2], "edges (1, 2) should be equal"
        assert g1[3, 4] != g2[3, 4], "edges (3, 4, 2.5) and (3, 4, 5.0) should not be equal"

    def test_attr__getitem__setitem__(self):
        g = Graph([(1, 2)])
        edge = g[1, 2]
        assert not edge.has_attributes_dict(), "attr dict should not be instantiated"
        _ = edge.attr
        assert (
            edge.has_attributes_dict()
        ), "attr dict should be instantiated after using property accessor"
        edge.attr["mass"] = 1000
        assert edge.attr["mass"] == 1000, "edge should have 'mass' attribute set to 1000"
        assert edge["mass"] == 1000, "edge __getitem__(key) should be alias for edge.attr[key]"
        edge["color"] = "blue"
        assert (
            edge.attr["color"] == "blue"
        ), "__setitem__(key, value) should be alias for attr[key] = value"

        with pytest.raises(KeyError):
            _ = edge["unknown_key"]

    def test_contract(self):
        g = Graph([(1, 2), (1, 3), (2, 3), (2, 4), (2, 2)])
        assert (
            g[1].adj_vertices() == {g[2], g[3]}
        ), "vertex 1 should be adjacent to vertices 2 and 3"

        g[1, 2].contract(remove_loops=False)

        assert (
            g[1].adj_vertices() == {g[1], g[3], g[4]}
        ), "after edge contraction, vertex 1 should be adjacent to vertices 1, 3, and 4"
        assert (
            g[1].loop_edge.label == "(1, 1)"
        ), "vertex 1 should have loop edge (1, 1), due to loop that was on vertex 2"
        assert not g.has_vertex(2), "after edge contraction, vertex 2 should be removed"

        g2 = Graph([("a", "b"), ("b", "c"), ("b", "b")])
        g2["a", "b"].contract(remove_loops=True)
        assert not g2["a"].loop_edge, "loop edge should be removed after edge contraction"

    def test_loop(self):
        g = Graph([(1, 1)])
        assert g[1, 1].is_loop(), "edge (1, 1) should self identify as a loop"
        assert g[1, 1].vertex1 == 1, "loop edge vertex1 should be 1"
        assert g[1, 1].vertex2 == 1, "loop edge vertex2 should be 1"

    def test_remove(self):
        g = Graph([(1, 2), (2, 3)])
        assert g.has_edge(1, 2), "prior to removal, graph should have edge (1, 2)"
        assert not g[1].is_isolated(), "vertex 1 should not be isolated prior to edge removal"

        g[1, 2].remove(remove_isolated_vertices=False)

        assert not g.has_edge(1, 2), "after removal, graph should not have edge (1, 2)"
        assert g.has_vertex(1), "after edge removal, isolated vertex 1 should still be in graph"
        assert g[1].is_isolated(), "vertex 1 should be isolated after edge removal"

        g2 = Graph([(1, 2), (1, 1), (2, 3)])
        g2[1, 2].remove(remove_isolated_vertices=True)
        assert (
            not g2.has_vertex(1)
        ), "after edge removal, isolated vertex 1 should have been removed"

    def test_weight(self):
        g = Graph([(1, 2), (3, 4, 9.5)])
        assert (
            g[1, 2].weight == edge_module.DEFAULT_WEIGHT
        ), "edge without a specified weight should default to edge_module.DEFAULT_WEIGHT"
        assert g[3, 4].weight == 9.5, "weight of edge (3, 4) should be 9.5"
        with pytest.raises(AttributeError):
            g[3, 4].weight = 100

    def test_vertex1_vertex2(self):
        g = Graph()
        g.add_edge(2, 1)
        assert g[1, 2].vertex1 == 2, "vertex1 should be 2"
        assert g[1, 2].vertex2 == 1, "vertex2 should be 1"


class TestMultiEdgeBase:
    """Tests for MultiEdgeBase features shared by all multiconnection classes."""

    def test_add_remove_get_connection(self):
        g = MultiGraph([(1, 2)])
        assert g[1, 2].multiplicity == 1, "edge (1, 2) should have multiplicity 1"

        g[1, 2].add_connection(weight=3.5, key="connection2", color="blue")

        assert (
            g[1, 2].multiplicity == 2
        ), "edge (1, 2) should have multiplicity 2 after adding parallel connection"
        connection = g[1, 2].get_connection("connection2")
        assert connection["color"] == "blue", "new connection should have color attribute 'blue'"
        assert connection.weight == 3.5, "new connection should have weight 3.5"

        g[1, 2].remove_connection("connection2")
        assert (
            g[1, 2].multiplicity == 1
        ), "edge (1, 2) should have multiplicity 1 after removing parallel connection"

    def test_connections_and_connection_items(self):
        g = MultiGraph([(1, 2), (1, 2), (2, 2), (2, 2), (2, 2)])
        c12 = list(g[1, 2].connections())
        assert len(c12) == 2, "edge (1, 2) should have 2 connections"

        keys = set()
        for key, connection in g[2, 2].connection_items():
            assert not connection.has_attributes_dict(), "loop connection should not have attr dict"
            keys.add(key)
        assert len(keys) == 3, "edge (2, 2) should have 3 parallel loop connections"

    def test_contract(self):
        g = MultiGraph([(1, 2), (1, 3), (2, 3), (2, 4), (2, 2), (2, 2)])
        assert (
            g[1].adj_vertices() == {g[2], g[3]}
        ), "vertex 1 should be adjacent to vertices 2 and 3"
        assert (
            g[1, 3].multiplicity == 1
        ), "before edge contraction, edge (1, 3) should have multiplicity 1"
        assert g[2, 2].multiplicity == 2, "vertex 2 should have two loops"

        g[1, 2].contract(remove_loops=False)

        assert (
            g[1].adj_vertices() == {g[1], g[3], g[4]}
        ), "after edge contraction, vertex 1 should be adjacent to vertices 1, 3, and 4"
        assert (
            g[1, 3].multiplicity == 2
        ), "after edge contraction, edge (1, 3) should have multiplicity 2"
        assert (
            g[1].loop_edge
        ), "vertex 1 should have loop edge (1, 1)"
        print(f"\nDEBUG: g[1].loop_edge => {g[1].loop_edge}\n")
        assert g[1].loop_edge.multiplicity == 3, "loop edge should have multiplicity 3"
        assert not g.has_vertex(2), "after edge contraction, vertex 2 should be removed"

        g2 = MultiDiGraph([(1, 2), (2, 4), (2, 2), (2, 2)])

        g2[1, 2].contract(remove_loops=True)

        assert not g2[1].loop_edge, "loop edge should be removed after edge contraction"

    def test_loop(self):
        g = MultiGraph([(1, 1), (1, 1)])
        assert g[1, 1].is_loop(), "edge (1, 1) should self identify as a loop"
        assert g[1, 1].vertex1 == 1, "loop edge vertex1 should be 1"
        assert g[1, 1].vertex2 == 1, "loop edge vertex2 should be 1"
        assert g[1, 1].multiplicity == 2, "there should be two parallel loop edge connections"

    def test_remove(self):
        g = MultiGraph([(1, 2), (1, 2), (2, 3)])
        assert g.has_edge(1, 2), "prior to removal, graph should have edge (1, 2)"
        assert not g[1].is_isolated(), "vertex 1 should not be isolated prior to edge removal"

        g[1, 2].remove(remove_isolated_vertices=False)

        assert not g.has_edge(1, 2), "after removal, graph should not have edge (1, 2)"
        assert g.has_vertex(1), "after edge removal, isolated vertex 1 should still be in graph"
        assert g[1].is_isolated(), "vertex 1 should be isolated after edge removal"

        g2 = MultiGraph([(1, 2), (1, 2), (1, 1), (1, 1), (2, 3)])
        g2[1, 2].remove(remove_isolated_vertices=True)
        assert (
            not g2.has_vertex(1)
        ), "after edge removal, isolated vertex 1 should have been removed"

    def test_weight(self):
        g = MultiGraph([(1, 2), (1, 2), (3, 4, 9.5), (3, 4, 0.5)])
        assert (
            g[1, 2].weight == 2 * edge_module.DEFAULT_WEIGHT
        ), "each parallel connection should default to edge_module.DEFAULT_WEIGHT"
        assert g[3, 4].weight == 10.0, "multiedge weight should be cumulative of all connections"
        assert (
            next(g[1, 2].connections()).weight == edge_module.DEFAULT_WEIGHT
        ), "Individual connections should default to edge_module.DEFAULT_WEIGHT"

    def test_vertex1_vertex2(self):
        g = MultiGraph([(2, 1), (1, 2)])
        assert g[1, 2].multiplicity == 2, "multiedge should have multiplicity 2"
        assert g[1, 2].vertex1 == 2, "vertex1 should be 2"
        assert g[1, 2].vertex2 == 1, "vertex2 should be 1"


class TestEdge:
    """Tests for the Edge class (concrete class _Edge)."""

    def test_issubclass_and_isinstance(self):
        g = Graph()
        edge: Edge = g.add_edge(1, 2)
        assert (
            isinstance(edge, edge_module.Connection)
        ), "edge should be an instance of superclass Connection"
        assert (
            isinstance(edge, edge_module.EdgeBase)
        ), "edge should be an instance of superclass EdgeBase"
        assert isinstance(edge, Edge), "edge should be an Edge instance"
        assert issubclass(Edge, edge_module.Connection), "Edge should be Connection subclass"
        assert issubclass(Edge, edge_module.EdgeBase), "Edge should be EdgeBase subclass"

    def test_equality_operator(self):
        g = Graph([(1, 2), (3, 4, 3.5), (4, 5, 7.5, {"color": "blue"}), (6, 7, 9.5, {"k": "v"})])
        g2 = Graph([(2, 1), (3, 4), (4, 5, 7.5, {"color": "red"}), (6, 7, 9.5, {"k": "v"})])
        assert g[1, 2] == g[1, 2], "edge (1, 2) should equal itself within the same graph"
        assert g[1, 2] == g2[2, 1], "edges (1, 2) and (2, 1) should be the same edge"
        assert g[3, 4] != g2[3, 4], "edges (3, 4) should not be equal due to different weights"
        assert g[4, 5] != g2[4, 5], "edges (4, 5) should not be equal due to different attributes"
        assert g[6, 7] == g2[6, 7], "edges (6, 7) should be equal"

    def test_repr_str_and_label(self):
        g = Graph([(2, 1)])
        assert g[2, 1].label == "(1, 2)", "edge label should be (1, 2)"
        assert g[2, 1].__str__() == g[2, 1].__repr__(), "edge __repr__ should equal __str__"
        assert g[2, 1].__str__() == "(1, 2)", "edge __str__ should be (1, 2)"
        g.add_edge(3, 4, weight=1.5)
        assert g[2, 1].label == "(1, 2)", "edge label should be (1, 2)"
        assert (
            g[2, 1].__str__() == "(1, 2, 1.0)"
        ), "edge __str__ should be (1, 2, 1.0) after adding weighted edge to graph"


class TestDiEdge:
    """Tests for the DiEdge class (concrete class _DiEdge)."""

    def test_issubclass_and_isinstance(self):
        g = DiGraph()
        edge: DiEdge = g.add_edge(1, 2)
        assert (
            isinstance(edge, edge_module.Connection)
        ), "edge should be an instance of superclass Connection"
        assert (
            isinstance(edge, edge_module.EdgeBase)
        ), "edge should be an instance of superclass EdgeBase"
        assert isinstance(edge, DiEdge), "edge should be an DiEdge instance"
        assert issubclass(DiEdge, edge_module.Connection), "DiEdge should be Connection subclass"
        assert issubclass(DiEdge, edge_module.EdgeBase), "DiEdge should be EdgeBase subclass"

    def test_equality_operator(self):
        dg = DiGraph([(1, 2), (3, 4, 3.5), (4, 5, 7.5, {"color": "blue"}), (6, 7, 9.5, {"k": "v"})])
        dg2 = DiGraph([(2, 1), (3, 4), (4, 5, 7.5, {"color": "red"}), (6, 7, 9.5, {"k": "v"})])
        assert dg[1, 2] == dg[1, 2], "edge (1, 2) should equal itself within the same graph"
        assert dg[1, 2] != dg2[2, 1], "edges (1, 2) and (2, 1) should not be equal in digraph"
        assert dg[3, 4] != dg2[3, 4], "edges (3, 4) should not be equal due to different weights"
        assert dg[4, 5] != dg2[4, 5], "edges (4, 5) should not be equal due to different attributes"
        assert dg[6, 7] == dg2[6, 7], "edges (6, 7) should be equal"

    def test_repr_str_and_label(self):
        dg = DiGraph([(2, 1)])
        assert dg[2, 1].label == "(2, 1)", "edge label should be (2, 1)"
        assert dg[2, 1].__str__() == dg[2, 1].__repr__(), "edge __repr__ should equal __str__"
        assert dg[2, 1].__str__() == "(2, 1)", "edge __str__ should be (2, 1)"
        dg.add_edge(3, 4, weight=1.5)
        assert dg[2, 1].label == "(2, 1)", "edge label should be (2, 1)"
        assert (
            dg[2, 1].__str__() == "(2, 1, 1.0)"
        ), "edge __str__ should be (2, 1, 1.0) after adding weighted edge to graph"


class TestMultiEdge:
    """Tests for MultiEdge class (concrete class _MultiEdge)."""

    def test_issubclass_and_isinstance(self):
        g = MultiGraph()
        edge: MultiEdge = g.add_edge(1, 2)
        assert (
            isinstance(edge, edge_module.MultiConnection)
        ), "edge should be an instance of superclass MultiConnection"
        assert (
            isinstance(edge, edge_module.MultiEdgeBase)
        ), "edge should be an instance of superclass MultiEdgeBase"
        assert isinstance(edge, MultiEdge), "edge should be an MultiEdge instance"
        assert (
            issubclass(MultiEdge, edge_module.MultiConnection)
        ), "MultiEdge should be MultiConnection subclass"
        assert (
            issubclass(MultiEdge, edge_module.MultiEdgeBase)
        ), "MultiEdge should be MultiEdgeBase subclass"

    def test_equality_operator(self):
        mg = MultiGraph([(1, 2), (2, 1), (3, 4), (3, 4, 1.5), (6, 7, 9.5, {"k": "v1"})])
        mg2 = MultiGraph([(2, 1), (2, 1), (3, 4), (3, 4, 5.5), (6, 7, 9.5, {"k": "v2"})])
        assert mg[1, 2] == mg2[2, 1], "multiedges (1, 2) should be equal"
        assert (
            mg[3, 4] != mg2[3, 4]
        ), "multiedges (3, 4) should not be equal due to different weights"
        assert (
            mg[6, 7] == mg2[6, 7]
        ), "multiedges (6, 7) should be equal (attributes of parallel connections not checked)"

    def test_repr_str_and_label(self):
        mg = MultiGraph([(2, 1), (1, 2)])
        assert mg[2, 1].label == "(1, 2)", "multiedge label should be (1, 2)"
        assert mg[2, 1].__str__() == mg[2, 1].__repr__(), "edge __repr__ should equal __str__"
        assert mg[2, 1].__str__() == "(1, 2), (1, 2)", "edge __str__ should be '(1, 2), (1, 2)'"
        mg.add_edge(3, 4, weight=1.5)
        assert mg[2, 1].label == "(1, 2)", "multiedge label should be (1, 2)"
        assert (
            mg[2, 1].__str__() == "(1, 2, 1.0), (1, 2, 1.0)"
        ), "edge __str__ should be '(1, 2, 1.0), (1, 2, 1.0)' after adding weighted edge to graph"


class TestMultiDiEdge:
    """Tests for MultiDiEdge class (concrete class _MultiDiEdge)."""

    def test_issubclass_and_isinstance(self):
        g = MultiDiGraph()
        edge: MultiDiEdge = g.add_edge(1, 2)
        assert (
            isinstance(edge, edge_module.MultiConnection)
        ), "edge should be an instance of superclass MultiConnection"
        assert (
            isinstance(edge, edge_module.MultiEdgeBase)
        ), "edge should be an instance of superclass MultiEdgeBase"
        assert isinstance(edge, MultiDiEdge), "edge should be an MultiDiEdge instance"
        assert (
            issubclass(MultiDiEdge, edge_module.MultiConnection)
        ), "MultiDiEdge should be MultiConnection subclass"
        assert (
            issubclass(MultiDiEdge, edge_module.MultiEdgeBase)
        ), "MultiDiEdge should be MultiEdgeBase subclass"

    def test_equality_operator(self):
        mdg = MultiDiGraph([(1, 2), (2, 1), (3, 4), (3, 4, 1.5), (6, 7, 9.5, {"k": "v1"})])
        mdg2 = MultiDiGraph([(2, 1), (2, 1), (3, 4), (3, 4, 5.5), (6, 7, 9.5, {"k": "v2"})])
        assert mdg[1, 2] != mdg2[2, 1], "multiedges (1, 2) and (2, 1) should not be equal"
        assert (
            mdg[2, 1] != mdg2[2, 1]
        ), "multiedges (2, 1) should not be equal due to different multiplicities)"
        assert (
            mdg[3, 4] != mdg2[3, 4]
        ), "multiedges (3, 4) should not be equal due to different weights"
        assert (
            mdg[6, 7] == mdg2[6, 7]
        ), "multiedges (6, 7) should be equal (attributes of parallel connections not checked)"

    def test_repr_str_and_label(self):
        mdg = MultiDiGraph([(2, 1), (2, 1)])
        assert mdg[2, 1].label == "(2, 1)", "directed multiedge label should be (2, 1)"
        assert mdg[2, 1].__str__() == mdg[2, 1].__repr__(), "edge __repr__ should equal __str__"
        assert mdg[2, 1].__str__() == "(2, 1), (2, 1)", "__str__ should be '(2, 1), (2, 1)'"
        mdg.add_edge(3, 4, weight=1.5)
        assert mdg[2, 1].label == "(2, 1)", "directed multiedge label should be (2, 1)"
        assert (
            mdg[2, 1].__str__() == "(2, 1, 1.0), (2, 1, 1.0)"
        ), "edge __str__ should be '(2, 1, 1.0), (2, 1, 1.0)' after adding weighted edge to graph"
