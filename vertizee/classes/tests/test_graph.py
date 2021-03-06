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

"""Tests for graph classes."""
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

import gc

import collections
from typing import Counter, Type, Union

import pytest

from vertizee import exception
from vertizee.classes import edge as edge_module
from vertizee.classes.edge import DiEdge, Edge, MultiDiEdge, MultiEdge
from vertizee.classes.graph import DiGraph, Graph, MultiDiGraph, MultiGraph
from vertizee.classes.vertex import Vertex


def _check_for_memory_leak(
    cls_graph: Type[Union[Graph, DiGraph, MultiGraph, MultiDiGraph]]
) -> None:
    def count_objects_of_type(_type: type) -> int:
        return sum(1 for obj in gc.get_objects() if isinstance(obj, _type))

    # Graph/DiGraph instances will ignore attempt to create parallel edge.
    g = cls_graph([(1, 2, {"color": "blue"}), (1, 2, {"color": "yellow"})])
    g.attr["name"] = "test graph"
    g[1].attr["mass"] = 42

    gc.collect()
    before = count_objects_of_type(cls_graph)
    g.deepcopy()
    gc.collect()
    after = count_objects_of_type(cls_graph)
    assert before == after, "unassigned graph reference should have been garbage collected"

    # Test subclassing.
    class MyGraph(cls_graph):  # type: ignore
        """cls_graph subclass."""

    gc.collect()
    mg = MyGraph([(1, 2, {"color": "blue"}), (1, 2, {"color": "yellow"})])
    mg.attr["name"] = "test graph"
    mg[1].attr["mass"] = 42
    before = count_objects_of_type(MyGraph)
    mg.deepcopy()
    gc.collect()
    after = count_objects_of_type(MyGraph)
    assert before == after, "unassigned graph reference should have been garbage collected"


class TestG:
    """Tests for graph features shared by all graph classes."""

    def test__contains__(self) -> None:
        g = Graph()
        g.add_edge(1, 2)
        assert g[1] in g, "vertex 1 should be in graph"
        assert 1 in g, "vertex specified as int should be in graph"
        assert "1" in g, "vertex specified as str should be in graph"
        assert 3 not in g, "vertex 3 should not be in graph"

        assert g.get_edge(1, 2) in g, "edge (1, 2) should be in graph"
        assert (1, 2) in g, "edge specified as tuple should be in graph"
        assert ("1", "2") in g, "edge specified as tuple should be in graph"
        assert (1, 3) not in g, "edge (1, 3) should not be in graph"

        with pytest.raises(TypeError):
            _ = 4.5 not in g  # type: ignore
        with pytest.raises(TypeError):
            _ = (1, 2, 3, 4) not in g  # type: ignore

    def test__getitem__(self) -> None:
        g = Graph()
        g.add_edge(1, 2)
        assert isinstance(g[1], Vertex), "graph should have vertex 1"
        assert isinstance(g["1"], Vertex), "graph should have vertex 1"
        assert isinstance(g[(1, {})], Vertex), "graph should have vertex 1"
        assert isinstance(g[1, {}], Vertex), "graph should have vertex 1"
        v1 = g[1]
        assert isinstance(g[v1], Vertex), "graph should have vertex 1"
        with pytest.raises(KeyError):
            _ = g.get_edge(1, 3)
        with pytest.raises(KeyError):
            _ = g[3]

        assert isinstance(g.get_edge(1, 2), Edge), "graph should have edge (1, 2)"
        assert isinstance(g.get_edge("1", "2"), Edge), "graph should have edge (1, 2)"
        _ = g.get_edge(1, 2)
        with pytest.raises(KeyError):
            _ = g.get_edge(1.0, 2.0)  # type: ignore
        with pytest.raises(KeyError):
            _ = g.get_edge(1, 3)

    def test__iter__(self) -> None:
        g = Graph([(1, 2), (2, 3), (3, 4)])
        count = sum(1 for _ in g)
        assert count == 4, "graph should iterate over its 4 vertices"
        assert set([g[1], g[2], g[3], g[4]]) == set(g), "graph should iterate over its 4 vertices"

    def test__len__(self) -> None:
        g = Graph([(1, 2), (2, 3), (3, 4)])
        assert len(g) == 4, "graph should contain 4 vertices"

    def test_add_vertex(self) -> None:
        g = Graph()
        g.add_vertex(1)
        g.add_vertex("2")
        g.add_vertex("v3", color="blue", mass=42)
        assert g.has_vertex(1), "graph should have vertex 1"
        assert g.has_vertex(2), "graph should have vertex 2"
        assert g.has_vertex("v3"), "graph should have vertex v3"
        assert g["v3"]["color"] == "blue", "vertex should have 'color' attribute set to 'blue'"
        assert g["v3"]["mass"] == 42, "vertex should have 'mass' attribute set to 42"

    def test_add_vertices_from(self) -> None:
        g = Graph()
        g.add_vertices_from([1, "2", ("v3", {"color": "blue", "mass": 42})])
        assert g.has_vertex(1), "graph should have vertex 1"
        assert g.has_vertex(2), "graph should have vertex 2"
        assert g.has_vertex("v3"), "graph should have vertex v3"
        assert g["v3"]["color"] == "blue", "vertex should have 'color' attribute set to 'blue'"
        assert g["v3"]["mass"] == 42, "vertex should have 'mass' attribute set to 42"

    def test_allows_self_loops(self) -> None:
        g = Graph()
        assert g.allows_self_loops(), "graph should default to allowing self loops"
        g.add_edge(1, 1)  # graph should allow adding self loop

        g2 = Graph(allow_self_loops=False)
        assert not g2.allows_self_loops(), "graph 2 should not allow self loops"
        with pytest.raises(exception.SelfLoopsNotAllowed):
            g2.add_edge(1, 1)

    def test_attr(self) -> None:
        g = Graph([(1, 2), (3, 4)], name="undirected graph", secret=42)

        assert g.attr["name"] == "undirected graph"
        assert g.attr["secret"] == 42, "graph should have 'secret' attribute set to 42"
        with pytest.raises(KeyError):
            _ = g.attr["unknown_key"]

    def test_clear(self) -> None:
        g = Graph([(1, 2), (2, 3), (3, 4)])
        assert g.edge_count > 0, "graph should have edges"
        assert g.vertex_count > 0, "graph should have vertices"
        g.clear()
        assert g.edge_count == 0, "graph should not have edges after clear()"
        assert g.vertex_count == 0, "graph should not have vertices after clear()"

    def test_deepcopy(self) -> None:
        g = Graph([(1, 2, {"color": "blue"}), (3, 4)])
        g.add_vertex(42)
        g_copy = g.deepcopy()

        assert set(g.vertices()) == set(g_copy.vertices()), "graph copy should have same vertices"
        assert (
            g[1] is not g_copy[1]
        ), "graph copy vertex objects should be distinct from original graph"

        assert set(g.edges()) == set(g_copy.edges()), "graph copy should have same edges"
        assert g.get_edge(1, 2) is not g_copy.get_edge(
            1, 2
        ), "graph copy edge objects should be distinct from original graph"
        assert (
            g.get_edge(1, 2)._attr == g_copy.get_edge(1, 2)._attr
        ), "graph copy edge object `_attr` dictionary should contain logically equal contents"
        assert (
            g.get_edge(1, 2)._attr is not g_copy.get_edge(1, 2)._attr
        ), "graph copy edge object `_attr` dictionary should be distinct from original graph"

    def test_has_vertex(self) -> None:
        g = Graph()
        g.add_edge(1, 2)
        assert g.has_vertex(1), "vertex specified as int should be in graph"
        assert g.has_vertex("1"), "vertex specified as str should be in graph"
        v1 = g[1]
        assert g.has_vertex(v1), "vertex specified as object should be in graph"
        assert not g.has_vertex(3), "vertex 3 should not be in the graph"

    def test_is_weighted(self) -> None:
        g = Graph()
        g.add_edge(1, 2)
        assert (
            not g.is_weighted()
        ), "graph without custom edge weights should not identify as weighted"
        g.add_edge(3, 4, weight=9.5)
        assert g.is_weighted(), "graph with custom edge weights should identify as weighted"

    def test_remove_edge(self) -> None:
        g = Graph([(1, 2), (2, 3), (3, 4)])
        assert g.edge_count == 3, "graph should have 3 edges"
        g.remove_edge(1, 2)
        assert g.edge_count == 2, "after edge removal, graph should have 2 edges"
        assert g.has_vertex(1), "isolated vertex 1 should not have been removed"

        g.remove_edge(2, 3, remove_semi_isolated_vertices=True)
        assert g.edge_count == 1, "after edge removal, graph should have 1 edge"
        assert not g.has_vertex(
            2
        ), "with `remove_semi_isolated_vertices` set to True, vertex 2 should have been removed"

        with pytest.raises(exception.EdgeNotFound):
            g.remove_edge(8, 9)

    def test_remove_edges_from(self) -> None:
        g = Graph([(1, 2), (2, 3), (3, 4)])
        assert g.edge_count == 3, "graph should have 3 edges"
        g.remove_edges_from([(1, 2), (2, 3)])
        assert g.edge_count == 1, "graph should have 1 edge after edge removals"
        assert g.has_edge(3, 4)

    def test_remove_isolated_vertices(self) -> None:
        g = Graph([(1, 2), (2, 3), (5, 5)])
        g.remove_edge(1, 2, remove_semi_isolated_vertices=False)
        g.add_vertex(4)
        assert set(g.vertices()) == {g[1], g[2], g[3], g[4], g[5]}

        g.remove_isolated_vertices(ignore_self_loops=False)
        assert set(g.vertices()) == {
            g[2],
            g[3],
            g[5],
        }, "isolated vertices 1 and 4 should have been removed"

        g.remove_isolated_vertices(ignore_self_loops=True)
        assert set(g.vertices()) == {g[2], g[3]}, "semi-isolated vertex 5 should have been removed"

    def test_remove_vertex(self) -> None:
        g = Graph([(1, 2), (3, 3)])
        g.add_vertex(4)

        assert g.has_vertex(4), "graph should have vertex 4"
        g.remove_vertex(4)
        assert not g.has_vertex(4), "graph should not have vertex 4 after removal"

        assert g.has_vertex(3), "graph should have semi-isolated vertex 3"
        g.remove_vertex(3)
        assert not g.has_vertex(3), "graph should not have vertex 3 after removal"

        with pytest.raises(exception.VertizeeException):
            g.remove_vertex(1)

    def test_vertex_count(self) -> None:
        g = Graph([(1, 2), (2, 3), (3, 4)])
        assert g.vertex_count == 4, "graph should have 4 vertices"
        g.add_vertex(5)
        assert g.vertex_count == 5, "graph should have 5 vertices"
        g.remove_vertex(5)
        assert g.vertex_count == 4, "graph should have 4 vertices"

    def test_vertices(self) -> None:
        g = Graph([(1, 2), (2, 3), (3, 4)])
        assert set(g.vertices()) == {
            g[1],
            g[2],
            g[3],
            g[4],
        }, "graph should have vertices 1, 2, 3, 4"


class TestGraph:
    """Tests for the Graph class."""

    def test_memory_leak(self) -> None:
        _check_for_memory_leak(Graph)

    def test__init__from_graph(self) -> None:
        mg = MultiGraph([(1, 2, 5.0, {"color": "red"}), (1, 2, 10.0, {"color": "black"}), (3, 4)])
        assert mg.weight == 16.0, "multigraph should have weight 16.0"

        g = Graph(mg)
        assert g.has_edge(1, 2), "graph should have edge (1, 2) copied from multigraph"
        assert (
            g.get_edge(1, 2)["color"] == "red"
        ), "edge (1, 2) should have 'color' attribute set to red"
        assert g.get_edge(1, 2).weight == 5.0, "edge (1, 2) should have weight 5.0"
        assert g.has_edge(3, 4), "graph should have edge (3, 4) copied from multigraph"
        assert g.edge_count == 2, "graph should have two edges"
        assert g.weight == 6.0, "graph should have weight 6.0"
        assert g.get_edge(1, 2) is not mg.get_edge(  # type: ignore
            1, 2
        ), "graph should have deep copies of edges and vertices"

    def test_add_edge(self) -> None:
        g = Graph()
        edge = g.add_edge(1, 2, weight=4.5, color="blue", mass=42)
        assert isinstance(edge, Edge), "new edge should an Edge object"
        assert g.get_edge(1, 2).weight == 4.5, "edge (1, 2) should have weight 4.5"
        assert edge["color"] == "blue", "edge should have 'color' attribute set to 'blue'"
        assert edge["mass"] == 42, "edge should have 'mass' attribute set to 42"

        edge_dup = g.add_edge(1, 2, weight=1.5, color="red", mass=57)
        assert (
            edge is edge_dup
        ), "adding an edge with same vertices as existing edge should return existing edge"

        g.add_edge(3, 4)
        assert (
            g.get_edge(3, 4).weight == edge_module.DEFAULT_WEIGHT
        ), "edge should have default weight"
        assert not g.get_edge(
            3, 4
        ).has_attributes_dict(), "edge should not have attributes dictionary"

    def test_add_edges_from(self) -> None:
        g = Graph()
        g.add_edges_from(
            [
                (1, 2, 4.5, {"color": "blue", "mass": 42}),
                (4, 3, 9.5),
                (5, 6, {"color": "red", "mass": 99}),
                (8, 7),
                (7, 8),
            ]
        )

        assert g.edge_count == 4, "graph should have 4 edges"
        assert g.get_edge(1, 2).weight == 4.5, "edge (1, 2) should have weight 4.5"
        assert g.get_edge(1, 2)["color"] == "blue", "edge (1, 2) should have 'color' set to 'blue'"
        assert g.get_edge(1, 2)["mass"] == 42, "edge (1, 2) should have 'mass' set to 42"
        assert not g.get_edge(
            3, 4
        ).has_attributes_dict(), "edge (3, 4) should not have attributes dict"
        assert (
            g.get_edge(5, 6).weight == edge_module.DEFAULT_WEIGHT
        ), "edge should have default weight"
        assert g.get_edge(7, 8) is g.get_edge(8, 7), "order of vertices should not matter"

    def test_edges_and_edge_count(self) -> None:
        g = Graph([(1, 2), (2, 1), (2, 3), (3, 4)])
        assert set(g.edges()) == {
            g.get_edge(1, 2),
            g.get_edge(2, 3),
            g.get_edge(3, 4),
        }, "edges should be (1, 2), (2, 3), (3, 4)"
        assert g.edge_count == 3, "graph should have 3 edges"

    def test_get_random_edge(self) -> None:
        g = Graph([(1, 2), (3, 4), (5, 6)])

        cnt: Counter[Edge] = collections.Counter()
        for _ in range(1000):
            rand_edge = g.get_random_edge()
            if rand_edge is None:
                raise Exception("rand_edge returned None")
            cnt[rand_edge] += 1
        assert cnt[g.get_edge(1, 2)] > 270, r"~33% of random samples should be edge (1, 2)"
        assert cnt[g.get_edge(3, 4)] > 270, r"~33% of random samples should be edge (3, 4)"
        assert cnt[g.get_edge(5, 6)] > 270, r"~33% of random samples should be edge (5, 6)"


class TestDiGraph:
    """Tests for the DiGraph class."""

    def test_memory_leak(self) -> None:
        _check_for_memory_leak(DiGraph)

    def test__init__from_graph(self) -> None:
        g = Graph([(2, 1, 5.0, {"color": "red"}), (4, 3)])
        assert g.weight == 6.0, "graph should have weight 6.0"

        dg = DiGraph(g)
        assert dg.has_edge(2, 1), "digraph should have edge (2, 1) copied from graph"
        assert (
            dg.get_edge(2, 1)["color"] == "red"
        ), "edge (2, 1) should have 'color' attribute set to red"
        assert dg.get_edge(2, 1).weight == 5.0, "edge (2, 1) should have weight 5.0"
        assert dg.has_edge(4, 3), "graph should have edge (4, 3) copied from graph"
        assert dg.edge_count == 2, "graph should have two edges"
        assert dg.weight == 6.0, "graph should have weight 6.0"
        assert dg.get_edge(2, 1) is not g.get_edge(  # type: ignore
            2, 1
        ), "digraph should have deep copies of edges and vertices"

    def test_add_edge(self) -> None:
        g = DiGraph()
        edge = g.add_edge(2, 1, weight=4.5, color="blue", mass=42)
        assert isinstance(edge, DiEdge), "new edge should an DiEdge object"
        assert g.get_edge(2, 1).weight == 4.5, "edge (2, 1) should have weight 4.5"
        assert edge["color"] == "blue", "edge should have 'color' attribute set to 'blue'"
        assert edge["mass"] == 42, "edge should have 'mass' attribute set to 42"

        edge_dup = g.add_edge(2, 1, weight=1.5, color="red", mass=57)
        assert (
            edge is edge_dup
        ), "adding an edge with same vertices as existing edge should return existing edge"

        g.add_edge(3, 4)
        assert (
            g.get_edge(3, 4).weight == edge_module.DEFAULT_WEIGHT
        ), "edge should have default weight"
        assert not g.get_edge(
            3, 4
        ).has_attributes_dict(), "edge should not have attributes dictionary"
        assert not g.has_edge(1, 2), "digraph should not have edge (1, 2)"

    def test_add_edges_from(self) -> None:
        g = DiGraph()
        g.add_edges_from(
            [
                (1, 2, 4.5, {"color": "blue", "mass": 42}),
                (4, 3, 9.5),
                (5, 6, {"color": "red", "mass": 99}),
                (8, 7),
                (7, 8),
            ]
        )

        assert g.edge_count == 5, "digraph should have 5 edges"
        assert g.get_edge(1, 2).weight == 4.5, "edge (1, 2) should have weight 4.5"
        assert g.get_edge(1, 2)["color"] == "blue", "edge (1, 2) should have 'color' set to 'blue'"
        assert g.get_edge(1, 2)["mass"] == 42, "edge (1, 2) should have 'mass' set to 42"
        assert not g.get_edge(
            4, 3
        ).has_attributes_dict(), "edge (3, 4) should not have attributes dict"
        assert (
            g.get_edge(5, 6).weight == edge_module.DEFAULT_WEIGHT
        ), "edge should have default weight"
        assert g.get_edge(7, 8) != g.get_edge(
            8, 7
        ), "order of vertices should specify different edges"

    def test_edges_and_edge_count(self) -> None:
        g = DiGraph([(1, 2), (2, 1), (2, 3)])
        assert set(g.edges()) == {
            g.get_edge(1, 2),
            g.get_edge(2, 1),
            g.get_edge(2, 3),
        }, "edges should be (1, 2), (2, 1), (2, 3)"
        assert g.edge_count == 3, "graph should have 3 edges"

    def test_get_random_edge(self) -> None:
        g = DiGraph([(1, 2), (2, 1), (5, 6)])

        cnt: Counter[DiEdge] = collections.Counter()
        for _ in range(1000):
            rand_edge = g.get_random_edge()
            if rand_edge is None:
                raise Exception("rand_edge returned None")
            cnt[rand_edge] += 1
        assert cnt[g.get_edge(1, 2)] > 270, r"~33% of random samples should be edge (1, 2)"
        assert cnt[g.get_edge(2, 1)] > 270, r"~33% of random samples should be edge (2, 1)"
        assert cnt[g.get_edge(5, 6)] > 270, r"~33% of random samples should be edge (5, 6)"


class TestMultiGraph:
    """Tests for the MultiGraph class."""

    def test_memory_leak(self) -> None:
        _check_for_memory_leak(MultiGraph)

    def test__init__from_graph(self) -> None:
        g0 = Graph([(1, 2, 5.0, {"color": "red"}), (3, 4)])

        g = MultiGraph(g0)
        assert g.has_edge(1, 2), "multigraph should have edge (1, 2) copied from graph"
        connection12 = g.get_edge(1, 2).connections()[0]
        assert (
            connection12["color"] == "red"
        ), "edge connection (1, 2) should have 'color' attribute set to red"
        assert g.get_edge(1, 2).weight == 5.0, "edge (1, 2) should have weight 5.0"
        assert g.has_edge(3, 4), "graph should have edge (3, 4) copied from graph"
        assert g.edge_count == 2, "graph should have two edges"
        assert g.weight == 6.0, "graph should have weight 6.0"
        assert g.get_edge(1, 2) is not g0.get_edge(  # type: ignore
            1, 2
        ), "multigraph should have deep copies of edges and vertices"

    def test_add_edge(self) -> None:
        g = MultiGraph()
        edge = g.add_edge(1, 2, weight=4.5, color="blue", mass=42)
        assert isinstance(edge, MultiEdge), "new edge should a MultiEdge object"
        assert g.get_edge(1, 2).weight == 4.5, "edge (1, 2) should have weight 4.5"

        connection12 = g.get_edge(1, 2).connections()[0]
        assert connection12["color"] == "blue", "edge should have 'color' attribute set to 'blue'"
        assert connection12["mass"] == 42, "edge should have 'mass' attribute set to 42"

        edge_dup = g.add_edge(1, 2, weight=1.5, color="red", mass=57)
        assert (
            edge is edge_dup
        ), "adding an edge with same vertices as existing edge should return existing edge"
        assert (
            g.get_edge(1, 2).multiplicity == 2
        ), "after adding parallel connection, edge multiplicity should be 2"

        g.add_edge(3, 4)
        assert (
            g.get_edge(3, 4).weight == edge_module.DEFAULT_WEIGHT
        ), "edge should have default weight"
        connection34 = g.get_edge(3, 4).connections()[0]
        assert not connection34.has_attributes_dict(), "edge should not have attributes dictionary"

    def test_add_edges_from(self) -> None:
        g = MultiGraph()
        g.add_edges_from(
            [
                (1, 2, 4.5, {"color": "blue", "mass": 42}),
                (4, 3, 9.5),
                (5, 6, {"color": "red", "mass": 99}),
                (8, 7),
                (7, 8),
            ]
        )

        connection12 = g.get_edge(1, 2).connections()[0]
        connection34 = g.get_edge(3, 4).connections()[0]
        assert g.edge_count == 5, "graph should have 5 edge connections"
        assert g.get_edge(1, 2).weight == 4.5, "edge (1, 2) should have weight 4.5"
        assert connection12["color"] == "blue", "edge (1, 2) should have 'color' set to 'blue'"
        assert connection12["mass"] == 42, "edge (1, 2) should have 'mass' set to 42"
        assert not connection34.has_attributes_dict(), "edge (3, 4) should not have attributes dict"
        assert (
            g.get_edge(5, 6).weight == edge_module.DEFAULT_WEIGHT
        ), "edge should have default weight"
        assert g.get_edge(7, 8) is g.get_edge(8, 7), "order of vertices should not matter"
        assert (
            g.get_edge(7, 8).multiplicity == 2
        ), "edge (7, 8) should have two parallel connections"

    def test_edges_and_edge_count(self) -> None:
        g = MultiGraph([(1, 2), (2, 1), (2, 3), (3, 4)])
        assert set(g.edges()) == {
            g.get_edge(1, 2),
            g.get_edge(2, 3),
            g.get_edge(3, 4),
        }, "multiedges should be (1, 2), (2, 3), (3, 4)"
        assert g.edge_count == 4, "graph should have 4 edge connections"
        assert (
            g.get_edge(1, 2).multiplicity == 2
        ), "multiedge (1, 2) should have two parallel connections"

    def test_get_random_edge(self) -> None:
        g = MultiGraph([(1, 2), (1, 2), (3, 4)])

        cnt1: Counter[MultiEdge] = collections.Counter()
        for _ in range(1000):
            rand_edge = g.get_random_edge_weighted_by_multiplicity()
            if rand_edge is None:
                raise Exception("rand_edge returned None")
            cnt1[rand_edge] += 1
        assert cnt1[g.get_edge(1, 2)] > 600, r"~66% of random samples should be edge (1, 2)"
        assert cnt1[g.get_edge(3, 4)] < 400, r"~33% of random samples should be edge (3, 4)"

        cnt2: Counter[MultiEdge] = collections.Counter()
        for _ in range(1000):
            rand_edge = g.get_random_edge()
            if rand_edge is None:
                raise Exception("rand_edge returned None")
            cnt2[rand_edge] += 1
        assert cnt2[g.get_edge(1, 2)] > 420, r"~50% of random samples should be edge (1, 2)"
        assert cnt2[g.get_edge(3, 4)] > 420, r"~50% of random samples should be edge (3, 4)"


class TestMultiDiGraph:
    """Tests for the MultiDiGraph class."""

    def test_memory_leak(self) -> None:
        _check_for_memory_leak(MultiDiGraph)

    def test__init__from_graph(self) -> None:
        g0 = MultiGraph([(2, 1, 5.0, {"color": "red"}), (2, 1, 2.5, {"color": "blue"})])

        g = MultiDiGraph(g0)
        assert g.has_edge(2, 1), "multigraph should have edge (2, 1) copied from graph"
        connection21 = g.get_edge(2, 1).connections()[0]
        assert (
            connection21["color"] == "red"
        ), "first connection of edge (2, 1) should have 'color' attribute set to red"
        assert g.get_edge(2, 1).weight == 7.5, "multiedge (2, 1) should have weight 7.5"
        assert g.edge_count == 2, "graph should have two edges"
        assert g.weight == 7.5, "graph should have weight 6.0"
        assert g.get_edge(2, 1) is not g0.get_edge(  # type: ignore
            2, 1
        ), "multigraph should have deep copies of edges and vertices"

    def test_add_edge(self) -> None:
        g = MultiDiGraph()
        edge = g.add_edge(1, 2, weight=4.5, color="blue", mass=42)
        assert isinstance(edge, MultiDiEdge), "new edge should a MultiDiEdge object"
        assert g.get_edge(1, 2).weight == 4.5, "edge (1, 2) should have weight 4.5"

        connection12 = g.get_edge(1, 2).connections()[0]
        assert connection12["color"] == "blue", "edge should have 'color' attribute set to 'blue'"
        assert connection12["mass"] == 42, "edge should have 'mass' attribute set to 42"

        edge_dup = g.add_edge(1, 2, weight=1.5, color="red", mass=57)
        assert (
            edge is edge_dup
        ), "adding an edge with same vertices as existing edge should return existing edge"
        assert (
            g.get_edge(1, 2).multiplicity == 2
        ), "after adding parallel connection, edge multiplicity should be 2"

        g.add_edge(3, 4)
        assert (
            g.get_edge(3, 4).weight == edge_module.DEFAULT_WEIGHT
        ), "edge should have default weight"
        connection34 = g.get_edge(3, 4).connections()[0]
        assert not connection34.has_attributes_dict(), "edge should not have attributes dictionary"

    def test_add_edges_from(self) -> None:
        g = MultiDiGraph()
        g.add_edges_from(
            [
                (1, 2, 4.5, {"color": "blue", "mass": 42}),
                (4, 3, 9.5),
                (5, 6, {"color": "red", "mass": 99}),
                (8, 7),
                (8, 7),
                (7, 8),
            ]
        )

        connection12 = g.get_edge(1, 2).connections()[0]
        connection43 = g.get_edge(4, 3).connections()[0]

        assert g.edge_count == 6, "graph should have 6 edges"
        assert g.get_edge(1, 2).weight == 4.5, "edge (1, 2) should have weight 4.5"
        assert connection12["color"] == "blue", "edge (1, 2) should have 'color' set to 'blue'"
        assert connection12["mass"] == 42, "edge (1, 2) should have 'mass' set to 42"
        assert (
            not connection43.has_attributes_dict()
        ), "edge connection (4, 3) should not have attributes dict"
        assert (
            g.get_edge(5, 6).weight == edge_module.DEFAULT_WEIGHT
        ), "edge should have default weight"
        assert g.get_edge(7, 8) is not g.get_edge(
            8, 7
        ), "order of vertices define different directed edges"
        assert (
            g.get_edge(8, 7).multiplicity == 2
        ), "edge (8, 7) should have two parallel connections"

    def test_edges_and_edge_count(self) -> None:
        g = MultiDiGraph([(1, 2), (2, 1), (2, 3), (3, 4)])
        assert set(g.edges()) == {
            g.get_edge(1, 2),
            g.get_edge(2, 1),
            g.get_edge(2, 3),
            g.get_edge(3, 4),
        }, "multiedges should be (1, 2), (2, 1), (2, 3), (3, 4)"
        assert g.edge_count == 4, "graph should have 4 edge connections"
        assert (
            g.get_edge(1, 2).multiplicity == 1
        ), "multiedge (1, 2) should have one edge connection"

    def test_get_random_edge(self) -> None:
        g = MultiDiGraph([(1, 2), (1, 2), (3, 4)])

        cnt1: Counter[MultiDiEdge] = collections.Counter()
        for _ in range(1000):
            rand_edge = g.get_random_edge_weighted_by_multiplicity()
            if rand_edge is None:
                raise Exception("rand_edge returned None")
            cnt1[rand_edge] += 1
        assert cnt1[g.get_edge(1, 2)] > 600, r"~66% of random samples should be edge (1, 2)"
        assert cnt1[g.get_edge(3, 4)] < 400, r"~33% of random samples should be edge (3, 4)"

        cnt2: Counter[MultiDiEdge] = collections.Counter()
        for _ in range(1000):
            rand_edge = g.get_random_edge()
            if rand_edge is None:
                raise Exception("rand_edge returned None")
            cnt2[rand_edge] += 1
        assert cnt2[g.get_edge(1, 2)] > 400, r"~50% of random samples should be edge (1, 2)"
        assert cnt2[g.get_edge(3, 4)] > 400, r"~50% of random samples should be edge (3, 4)"
