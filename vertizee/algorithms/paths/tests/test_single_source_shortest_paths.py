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

"""Tests for algorithms that solve the single-source-shortest-paths problem."""
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

import timeit
from typing import cast, Optional

import pytest

from vertizee import NegativeWeightCycle
from vertizee.algorithms.algo_utils.path_utils import reconstruct_path, ShortestPath
from vertizee.algorithms.paths.single_source import (
    bellman_ford,
    dijkstra,
    dijkstra_fibonacci,
    shortest_paths,
    breadth_first_search_shortest_paths,
)
from vertizee.classes.data_structures.vertex_dict import VertexDict
from vertizee.classes.edge import Attributes, MultiEdgeBase
from vertizee.classes.graph import DiGraph, MultiDiGraph, MultiGraph
from vertizee.classes.vertex import V


class TestBellmanFord:
    """Tests for Bellman-Ford algorithm."""

    def test_bellman_ford_default_edge_weight(self):
        g = DiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", 2),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 3),
                ("y", "x", 9),
                ("y", "z", 2),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        path_dict: VertexDict[ShortestPath] = bellman_ford(g, "s")

        assert len(path_dict) == 5, "Shortest path_dict dictionary should have length equal to |V|."
        assert path_dict["s"].length == 0, "Length of s path should be 0."
        assert path_dict["t"].length == 8, "Length of path s ~> t should be 8."
        assert path_dict["x"].length == 9, "Length of path s ~> x should be 9."
        assert path_dict["y"].length == 5, "Length of path s ~> y should be 5."
        assert path_dict["z"].length == 7, "Length of path s ~> z should be 7."

    def test_bellman_ford_negative_edge_weights(self):
        g = DiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", -6),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 8),
                ("y", "x", 4),
                ("y", "z", -3),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        path_dict: VertexDict[ShortestPath] = bellman_ford(g, "s")

        assert len(path_dict) == 5, "Shortest path_dict dictionary should have length equal to |V|."
        assert path_dict["s"].length == 0, "Length of s path should be 0."
        assert path_dict["t"].length == 10, "Length of path s ~> t should be 10."
        assert path_dict["x"].length == 7, "Length of path s ~> x should be 7."
        assert path_dict["y"].length == 4, "Length of path s ~> y should be 4."
        assert path_dict["z"].length == 1, "Length of path s ~> z should be 1."

    def test_bellman_ford_path_reconstruction(self):
        g = DiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", -6),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 8),
                ("y", "x", 4),
                ("y", "z", -3),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        path_dict: VertexDict[ShortestPath] = bellman_ford(g, "s", save_paths=True)

        assert path_dict["t"].path() == ["s", "t"], "Path s ~> t should be [s, t]."
        assert path_dict["x"].path() == [
            "s",
            "t",
            "y",
            "z",
            "x",
        ], "Path s ~> x should be [s, t, y, z, x]."
        assert path_dict["z"].path() == ["s", "t", "y", "z"], "Path s ~> z should be [s, t, y, z]."

        path_s_t = reconstruct_path("s", "t", path_dict)
        assert path_s_t == path_dict["t"].path(), "Algorithm path should match reconstructed path."
        path_s_x = reconstruct_path("s", "x", path_dict)
        assert path_s_x == path_dict["x"].path(), "Algorithm path should match reconstructed path."
        path_s_z = reconstruct_path("s", "z", path_dict)
        assert path_s_z == path_dict["z"].path(), "Algorithm path should match reconstructed path."

    def test_bellman_ford_reverse_graph(self):
        g = DiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", -6),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 8),
                ("y", "x", 4),
                ("y", "z", -3),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        path_dict: VertexDict[ShortestPath] = bellman_ford(g, "s", reverse_graph=True)

        assert len(path_dict) == 5, "Shortest path_dict dictionary should have length equal to |V|."
        assert path_dict["s"].length == 0, "Length of s path should be 0."
        assert path_dict["t"].length == -2, "Length of path s ~> t should be -2."
        assert path_dict["x"].length == 11, "Length of path s ~> x should be 11."
        assert path_dict["y"].length == 4, "Length of path s ~> y should be 4."
        assert path_dict["z"].length == 7, "Length of path s ~> z should be 7."

    def test_bellman_ford_negative_weight_cycle(self):
        g = DiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", -6),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 8),
                ("y", "x", 4),
                ("y", "z", -3),
                ("z", "s", -2),
                ("z", "x", 6),
            ]
        )

        with pytest.raises(NegativeWeightCycle):
            bellman_ford(g, "s")

    def test_bellman_ford_undirected_negative_weight_cycle(self):
        g = MultiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", -6),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 8),
                ("y", "x", 4),
                ("y", "z", -3),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        with pytest.raises(NegativeWeightCycle):
            bellman_ford(g, "s")

    def test_bellman_ford_undirected(self):
        g = MultiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", 6),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 8),
                ("y", "x", 4),
                ("y", "z", 3),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        path_dict: VertexDict[ShortestPath] = bellman_ford(g, "s")

        assert len(path_dict) == 5, "Shortest path_dict dictionary should have length equal to |V|."
        assert path_dict["s"].length == 0, "Length of s path should be 0."
        assert path_dict["t"].length == 10, "Length of path s ~> t should be 10."
        assert path_dict["x"].length == 9, "Length of path s ~> x should be 9."
        assert path_dict["y"].length == 5, "Length of path s ~> y should be 5."
        assert path_dict["z"].length == 7, "Length of path s ~> z should be 7."


class TestBreadthFirstSearchShortestPaths:
    """Tests for shortest-paths unweighted using breadth-first search."""

    def test_breadth_first_search_shortest_paths(self):
        g = DiGraph(
            [
                ("s", "t"),
                ("s", "y"),
                ("t", "y"),
                ("t", "x"),
                ("x", "z"),
                ("y", "t"),
                ("y", "x"),
                ("y", "z"),
                ("z", "s"),
                ("z", "x"),
            ]
        )

        path_dict: VertexDict[ShortestPath] = breadth_first_search_shortest_paths(g, "s")

        assert len(path_dict) == 5, "Shortest path_dict dictionary should have length equal to |V|."
        assert path_dict["s"].length == 0, "Length of s path should be 0."
        assert path_dict["t"].length == 1, "Length of path s ~> t should be 1."
        assert path_dict["x"].length == 2, "Length of path s ~> x should be 2."
        assert path_dict["y"].length == 1, "Length of path s ~> y should be 1."
        assert path_dict["z"].length == 2, "Length of path s ~> z should be 2."


class TestDijkstra:
    """Tests for Dijkstra's algorithm."""

    def test_dijkstra_default_edge_weight(self):
        g = DiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", 2),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 3),
                ("y", "x", 9),
                ("y", "z", 2),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        path_dict: VertexDict[ShortestPath] = dijkstra(g, "s")

        assert len(path_dict) == 5, "Shortest path_dict dictionary should have length equal to |V|."
        assert path_dict["s"].length == 0, "Length of s path should be 0."
        assert path_dict["t"].length == 8, "Length of path s ~> t should be 8."
        assert path_dict["x"].length == 9, "Length of path s ~> x should be 9."
        assert path_dict["y"].length == 5, "Length of path s ~> y should be 5."
        assert path_dict["z"].length == 7, "Length of path s ~> z should be 7."

    def test_dijkstra_path_reconstruction(self):
        g = DiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", 2),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 3),
                ("y", "x", 9),
                ("y", "z", 2),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        path_dict: VertexDict[ShortestPath] = dijkstra(g, "s", save_paths=True)

        assert path_dict["t"].path() == ["s", "y", "t"], "Path s ~> t should be [s, y, t]."
        assert path_dict["x"].path() == ["s", "y", "t", "x"], "Path s ~> x should be [s, y, t, x]."
        assert path_dict["z"].path() == ["s", "y", "z"], "Path s ~> z should be [s, y, z]."

        path_s_t = reconstruct_path("s", "t", path_dict)
        assert path_s_t == path_dict["t"].path(), "Algorithm path should match reconstructed path."
        path_s_x = reconstruct_path("s", "x", path_dict)
        assert path_s_x == path_dict["x"].path(), "Algorithm path should match reconstructed path."
        path_s_z = reconstruct_path("s", "z", path_dict)
        assert path_s_z == path_dict["z"].path(), "Algorithm path should match reconstructed path."

    def test_dijkstra_edge_attr_weights(self):
        WEIGHT = "weight_key"
        g = DiGraph(
            [
                ("s", "t"),
                ("s", "y"),
                ("t", "y"),
                ("t", "x"),
                ("x", "z"),
                ("y", "t"),
                ("y", "x"),
                ("y", "z"),
                ("z", "s"),
                ("z", "x"),
            ]
        )
        g.get_edge("s", "t")[WEIGHT] = 10
        g.get_edge("s", "y")[WEIGHT] = 5
        g.get_edge("t", "y")[WEIGHT] = 2
        g.get_edge("t", "x")[WEIGHT] = 1
        g.get_edge("x", "z")[WEIGHT] = 4
        g.get_edge("y", "t")[WEIGHT] = 3
        g.get_edge("y", "x")[WEIGHT] = 9
        g.get_edge("y", "z")[WEIGHT] = 2
        g.get_edge("z", "s")[WEIGHT] = 7
        g.get_edge("z", "x")[WEIGHT] = 6

        path_dict: VertexDict[ShortestPath] = dijkstra(g, "s", weight=WEIGHT)

        assert len(path_dict) == 5, "Shortest path_dict dictionary should have length equal to |V|."
        assert path_dict["s"].length == 0, "Length of s path should be 0."
        assert path_dict["t"].length == 8, "Length of path s ~> t should be 8."
        assert path_dict["x"].length == 9, "Length of path s ~> x should be 9."
        assert path_dict["y"].length == 5, "Length of path s ~> y should be 5."
        assert path_dict["z"].length == 7, "Length of path s ~> z should be 7."

    def test_dijkstra_edge_weight_filter_function(self):
        COLOR = "color_key"

        g = DiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", 2),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 3),
                ("y", "x", 9),
                ("y", "z", 2),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        g.get_edge("s", "t")[COLOR] = "RED"
        g.get_edge("s", "y")[COLOR] = "BLUE"
        g.get_edge("t", "y")[COLOR] = "RED"
        g.get_edge("t", "x")[COLOR] = "RED"
        g.get_edge("x", "z")[COLOR] = "RED"
        g.get_edge("y", "t")[COLOR] = "BLUE"
        g.get_edge("y", "x")[COLOR] = "RED"
        g.get_edge("y", "z")[COLOR] = "BLUE"
        g.get_edge("z", "s")[COLOR] = "BLUE"
        g.get_edge("z", "x")[COLOR] = "BLUE"

        # Exclude blue edges.
        def get_weight(v1: V, v2: V, reverse_graph: bool) -> Optional[float]:
            graph = v1._parent_graph
            if reverse_graph:
                edge = graph.get_edge(v2, v1)
                edge_str = f"({v2.label}, {v1.label})"
            else:
                edge = graph.get_edge(v1, v2)
                edge_str = f"({v1.label}, {v2.label})"
            if edge is None:
                raise ValueError(f"graph does not have edge {edge_str}")

            if graph.is_multigraph():
                assert isinstance(edge, MultiEdgeBase)
                has_color = any(c.attr.get(COLOR, "no color") == "BLUE" for c in edge.connections())
                if has_color:
                    return None
                return min(c.weight for c in edge.connections())

            if cast(Attributes, edge).attr.get(COLOR, "no color attribute") == "BLUE":
                return None
            return edge.weight

        path_dict: VertexDict[ShortestPath] = dijkstra(g, "s", weight=get_weight)
        assert path_dict["s"].length == 0, "Length of s path should be 0."
        assert path_dict["t"].length == 10, "Length of path s ~> t should be 10."
        assert path_dict["x"].length == 11, "Length of path s ~> x should be 11."
        assert path_dict["y"].length == 12, "Length of path s ~> y should be 12."
        assert path_dict["z"].length == 15, "Length of path s ~> z should be 15."

    def test_dijkstra_reverse_graph(self):
        g = MultiDiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", 2),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 3),
                ("y", "x", 9),
                ("y", "z", 2),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        path_dict: VertexDict[ShortestPath] = dijkstra(g, "s", reverse_graph=True)
        assert path_dict["s"].length == 0, "Length of s path should be 0."
        assert path_dict["t"].length == 11, "Length of path s ~> t should be 11."
        assert path_dict["x"].length == 11, "Length of path s ~> x should be 11."
        assert path_dict["y"].length == 9, "Length of path s ~> y should be 9."
        assert path_dict["z"].length == 7, "Length of path s ~> z should be 7."

    def test_dijkstra_undirected_graph(self):
        g = MultiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", 2),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 3),
                ("y", "x", 9),
                ("y", "z", 2),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        path_dict: VertexDict[ShortestPath] = dijkstra(g, "s")
        assert path_dict["s"].length == 0, "Length of s path should be 0."
        assert path_dict["t"].length == 7, "Length of path s ~> t should be 7."
        assert path_dict["x"].length == 8, "Length of path s ~> x should be 8."
        assert path_dict["y"].length == 5, "Length of path s ~> y should be 5."
        assert path_dict["z"].length == 7, "Length of path s ~> z should be 7."

    def test_dijkstra_fibonacci_default_edge_weight(self):
        g = DiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", 2),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 3),
                ("y", "x", 9),
                ("y", "z", 2),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        path_dict: VertexDict[ShortestPath] = dijkstra_fibonacci(g, "s")

        assert len(path_dict) == 5, "Shortest path_dict dictionary should have length equal to |V|."
        assert path_dict["s"].length == 0, "Length of s path should be 0."
        assert path_dict["t"].length == 8, "Length of path s ~> t should be 8."
        assert path_dict["x"].length == 9, "Length of path s ~> x should be 9."
        assert path_dict["y"].length == 5, "Length of path s ~> y should be 5."
        assert path_dict["z"].length == 7, "Length of path s ~> z should be 7."


class TestShortestPaths:
    """Tests for shortest-paths function."""

    g = DiGraph()
    for i in range(100):
        g.add_edge(i, i + 1)
        g.add_edge(i, i + 3)
    for i in range(100):
        g.add_edge(i + 6, i)

    # shortest_paths(g, source=0)
    bfs_time = timeit.timeit(
        "shortest_paths(g, source=0)", globals={"shortest_paths": shortest_paths, "g": g}, number=5
    )

    for index, edge in enumerate(g.edges()):
        if index % 3 == 0:
            edge._weight = -2
    g._is_weighted_graph = True

    bellman_ford_time = timeit.timeit(
        "shortest_paths(g, source=0)", globals={"shortest_paths": shortest_paths, "g": g}, number=5
    )
    # TODO(cpeisert): BFS is running slower than Bellman Ford on small graphs. Look into
    # simplifying BFS default implementation to reduce constant-factor overhead.

    # assert bfs_time < bellman_ford_time
