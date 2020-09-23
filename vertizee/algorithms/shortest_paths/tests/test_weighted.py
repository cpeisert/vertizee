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

"""Tests for finding the shortest paths in weighted graphs."""

import pytest

from vertizee import NegativeWeightCycle
from vertizee.algorithms.shortest_paths.weighted import (
    all_pairs_shortest_paths_floyd_warshall,
    all_pairs_shortest_paths_johnson,
    all_pairs_shortest_paths_johnson_fibonacci,
    shortest_paths_bellman_ford,
    shortest_paths_dijkstra,
    shortest_paths_dijkstra_fibonacci,
)
from vertizee.classes.collections.vertex_dict import VertexDict
from vertizee.classes.digraph import DiGraph, MultiDiGraph
from vertizee.classes.edge import EdgeType
from vertizee.classes.graph import MultiGraph
from vertizee.classes.shortest_path import reconstruct_path, ShortestPath
from vertizee.classes.vertex import Vertex

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip."
)

INFINITY = float("inf")


class TestWeightedAllPairsShortestPaths:
    def test_floyd_warshall_positive_edge_weights(self):
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

        paths: VertexDict[VertexDict[ShortestPath]] = all_pairs_shortest_paths_floyd_warshall(g)

        assert len(paths) == 5, "Shortest paths dictionary should have length equal to |V|."
        assert len(paths["s"]) == 5, (
            "Each source in the shortest paths dictionary should have "
            "a destinations dictionary of length equal to |V|."
        )
        assert paths["s"]["s"].length == 0, "Length of s ~> s path should be 0."
        assert paths["s"]["t"].length == 8, "Length of path s ~> t should be 8."
        assert paths["s"]["x"].length == 9, "Length of path s ~> x should be 9."
        assert paths["s"]["y"].length == 5, "Length of path s ~> y should be 5."
        assert paths["s"]["z"].length == 7, "Length of path s ~> z should be 7."

        assert paths["y"]["t"].length == 3, "Length of path y ~> t should be 3."
        assert paths["y"]["x"].length == 4, "Length of path y ~> x should be 4."
        assert paths["y"]["s"].length == 9, "Length of path y ~> s should be 9."
        assert paths["y"]["z"].length == 2, "Length of path y ~> z should be 2."

    def test_floyd_warshall_negative_edge_weights(self):
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
                ("a", "s", -100),
            ]
        )

        paths: VertexDict[VertexDict[ShortestPath]] = all_pairs_shortest_paths_floyd_warshall(g)

        assert len(paths) == 6, "Shortest paths dictionary should have length equal to |V|."
        assert len(paths["s"]) == 6, (
            "Each source in the shortest paths dictionary should have "
            "a destinations dictionary of length equal to |V|."
        )

        assert paths["a"]["s"].length == -100, "Length of path a ~> s should be -100."
        assert paths["a"]["t"].length == -90, "Length of path a ~> t should be -90."
        assert paths["s"]["a"].length == INFINITY, "Length of path s ~> a should be infinity."
        assert paths["z"]["a"].length == INFINITY, "Length of path z ~> a should be infinity."
        assert not paths["s"][
            "a"
        ].is_destination_reachable(), "'a' should not be reachable from 's'."

        assert paths["s"]["s"].length == 0, "Length of s path should be 0."
        assert paths["s"]["t"].length == 10, "Length of path s ~> t should be 10."
        assert paths["s"]["x"].length == 7, "Length of path s ~> x should be 7."
        assert paths["s"]["y"].length == 4, "Length of path s ~> y should be 4."
        assert paths["s"]["z"].length == 1, "Length of path s ~> z should be 1."

        assert paths["z"]["s"].length == 7, "Length of path z ~> s should be 7."
        assert paths["z"]["t"].length == 17, "Length of path z ~> t should be 17."
        assert paths["z"]["x"].length == 6, "Length of path z ~> x should be 6."
        assert paths["z"]["y"].length == 11, "Length of path z ~> y should be 11."
        assert paths["z"]["z"].length == 0, "Length of path z ~> z should be 0."

    def test_floyd_warshall_find_path_lengths_only(self):
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

        paths: VertexDict[VertexDict[ShortestPath]] = all_pairs_shortest_paths_floyd_warshall(
            g, find_path_lengths_only=True
        )

        assert paths["s"]["z"].path == [], "With find_path_lengths_only == True, all paths empty."
        assert paths["y"]["s"].path == [], "With find_path_lengths_only == True, all paths empty."
        assert paths["y"]["t"].path == [], "With find_path_lengths_only == True, all paths empty."
        assert paths["y"]["x"].path == [], "With find_path_lengths_only == True, all paths empty."
        assert paths["y"]["y"].path == [], "With find_path_lengths_only == True, all paths empty."
        assert paths["y"]["z"].path == [], "With find_path_lengths_only == True, all paths empty."

    def test_floyd_warshall_path_reconstruction(self):
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
                ("a", "s", -100),
            ]
        )

        paths: VertexDict[VertexDict[ShortestPath]] = all_pairs_shortest_paths_floyd_warshall(
            g, find_path_lengths_only=False
        )

        assert paths["s"]["z"].path == ["s", "y", "z"], "Path s ~> z should be [s, y, z]."
        assert paths["y"]["s"].path == ["y", "z", "s"], "Path y ~> s should be [y, z, s]."
        assert paths["y"]["t"].path == ["y", "t"], "Path y ~> t should be [y, t]."
        assert paths["y"]["x"].path == ["y", "t", "x"], "Path y ~> x should be [y, t, x]."
        assert paths["y"]["y"].path == ["y"], "Path y ~> y should be [y]."
        assert paths["y"]["z"].path == ["y", "z"], "Path y ~> z should be [y, z]."
        assert paths["y"]["a"].path == [], "Path y ~> a should be []."

        path_s_z = reconstruct_path("s", "z", paths)
        assert path_s_z == paths["s"]["z"].path, "Algorithm path should match reconstructed path."
        path_y_s = reconstruct_path("y", "s", paths)
        assert path_y_s == paths["y"]["s"].path, "Algorithm path should match reconstructed path."
        path_y_t = reconstruct_path("y", "t", paths)
        assert path_y_t == paths["y"]["t"].path, "Algorithm path should match reconstructed path."
        path_y_x = reconstruct_path("y", "x", paths)
        assert path_y_x == paths["y"]["x"].path, "Algorithm path should match reconstructed path."
        path_y_y = reconstruct_path("y", "y", paths)
        assert path_y_y == paths["y"]["y"].path, "Algorithm path should match reconstructed path."
        path_y_z = reconstruct_path("y", "z", paths)
        assert path_y_z == paths["y"]["z"].path, "Algorithm path should match reconstructed path."
        path_y_a = reconstruct_path("y", "a", paths)
        assert path_y_a == paths["y"]["a"].path, "Algorithm path should match reconstructed path."

    def test_floyd_warshall_negative_weight_cycle(self):
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
            all_pairs_shortest_paths_floyd_warshall(g)

    def test_floyd_warshall_undirected_negative_weight_cycle(self):
        g = MultiGraph(
            [
                ("s", "t", 10),
                ("s", "y", 5),
                ("t", "y", -6),
                ("t", "x", 1),
                ("x", "z", 4),
                ("y", "t", 8),
                ("y", "x", 4),
                ("y", "z", 3),
                ("z", "s", 7),
                ("z", "x", 6),
            ]
        )

        with pytest.raises(NegativeWeightCycle):
            all_pairs_shortest_paths_floyd_warshall(g)

    def test_floyd_warshall_undirected(self):
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

        paths: VertexDict[VertexDict[ShortestPath]] = all_pairs_shortest_paths_floyd_warshall(g)

        assert len(paths) == 5, "Shortest paths dictionary should have length equal to |V|."
        assert paths["s"]["s"].length == 0, "Length of path s ~> s should be 0."
        assert paths["s"]["t"].length == 10, "Length of path s ~> t should be 10."
        assert paths["s"]["x"].length == 9, "Length of path s ~> x should be 9."
        assert paths["s"]["y"].length == 5, "Length of path s ~> y should be 5."
        assert paths["s"]["z"].length == 7, "Length of path s ~> z should be 7."

        assert paths["x"]["s"].length == 9, "Length of path x ~> s should be 9."
        assert paths["x"]["t"].length == 1, "Length of path x ~> t should be 1."
        assert paths["x"]["x"].length == 0, "Length of path x ~> x should be 0."
        assert paths["x"]["y"].length == 4, "Length of path x ~> y should be 5."
        assert paths["x"]["z"].length == 4, "Length of path x ~> z should be 7."

    def test_johnson_positive_edge_weights(self):
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

        paths: VertexDict[VertexDict[ShortestPath]] = all_pairs_shortest_paths_johnson(g)

        assert len(paths) == 5, "Shortest paths dictionary should have length equal to |V|."
        assert len(paths["s"]) == 5, (
            "Each source in the shortest paths dictionary should have "
            "a destinations dictionary of length equal to |V|."
        )
        assert paths["s"]["s"].length == 0, "Length of s ~> s path should be 0."
        assert paths["s"]["t"].length == 8, "Length of path s ~> t should be 8."
        assert paths["s"]["x"].length == 9, "Length of path s ~> x should be 9."
        assert paths["s"]["y"].length == 5, "Length of path s ~> y should be 5."
        assert paths["s"]["z"].length == 7, "Length of path s ~> z should be 7."

        assert paths["y"]["t"].length == 3, "Length of path y ~> t should be 3."
        assert paths["y"]["x"].length == 4, "Length of path y ~> x should be 4."
        assert paths["y"]["s"].length == 9, "Length of path y ~> s should be 9."
        assert paths["y"]["z"].length == 2, "Length of path y ~> z should be 2."

    def test_johnson_negative_edge_weights(self):
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
                ("a", "s", -100),
            ]
        )

        paths: VertexDict[VertexDict[ShortestPath]] = all_pairs_shortest_paths_johnson(g)

        assert len(paths) == 6, "Shortest paths dictionary should have length equal to |V|."
        assert len(paths["s"]) == 6, (
            "Each source in the shortest paths dictionary should have "
            "a destinations dictionary of length equal to |V|."
        )

        assert paths["a"]["s"].length == -100, "Length of path a ~> s should be -100."
        assert paths["a"]["t"].length == -90, "Length of path a ~> t should be -90."
        assert paths["s"]["a"].length == INFINITY, "Length of path s ~> a should be infinity."
        assert paths["z"]["a"].length == INFINITY, "Length of path z ~> a should be infinity."
        assert not paths["s"][
            "a"
        ].is_destination_reachable(), "'a' should not be reachable from 's'."

        assert paths["s"]["s"].length == 0, "Length of s path should be 0."
        assert paths["s"]["t"].length == 10, "Length of path s ~> t should be 10."
        assert paths["s"]["x"].length == 7, "Length of path s ~> x should be 7."
        assert paths["s"]["y"].length == 4, "Length of path s ~> y should be 4."
        assert paths["s"]["z"].length == 1, "Length of path s ~> z should be 1."

        assert paths["z"]["s"].length == 7, "Length of path z ~> s should be 7."
        assert paths["z"]["t"].length == 17, "Length of path z ~> t should be 17."
        assert paths["z"]["x"].length == 6, "Length of path z ~> x should be 6."
        assert paths["z"]["y"].length == 11, "Length of path z ~> y should be 11."
        assert paths["z"]["z"].length == 0, "Length of path z ~> z should be 0."

    def test_johnson_path_reconstruction(self):
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
                ("a", "s", -100),
            ]
        )

        paths: VertexDict[VertexDict[ShortestPath]] = all_pairs_shortest_paths_johnson(
            g, find_path_lengths_only=False
        )

        assert paths["s"]["z"].path == ["s", "y", "z"], "Path s ~> z should be [s, y, z]."
        assert paths["y"]["y"].path == ["y"], "Path y ~> y should be [y]."
        assert paths["y"]["z"].path == ["y", "z"], "Path y ~> z should be [y, z]."
        assert paths["y"]["a"].path == [], "Path y ~> a should be []."

        path_s_z = reconstruct_path("s", "z", paths)
        assert path_s_z == paths["s"]["z"].path, "Algorithm path should match reconstructed path."
        path_y_y = reconstruct_path("y", "y", paths)
        assert path_y_y == paths["y"]["y"].path, "Algorithm path should match reconstructed path."
        path_y_z = reconstruct_path("y", "z", paths)
        assert path_y_z == paths["y"]["z"].path, "Algorithm path should match reconstructed path."
        path_y_a = reconstruct_path("y", "a", paths)
        assert path_y_a == paths["y"]["a"].path, "Algorithm path should match reconstructed path."

    def test_johnson_negative_weight_cycle(self):
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
            all_pairs_shortest_paths_johnson(g)

    def test_johnson_fibonacci_negative_edge_weights(self):
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
                ("a", "s", -100),
            ]
        )

        paths: VertexDict[VertexDict[ShortestPath]] = all_pairs_shortest_paths_johnson_fibonacci(g)

        assert len(paths) == 6, "Shortest paths dictionary should have length equal to |V|."
        assert len(paths["s"]) == 6, (
            "Each source in the shortest paths dictionary should have "
            "a destinations dictionary of length equal to |V|."
        )

        assert paths["a"]["s"].length == -100, "Length of path a ~> s should be -100."
        assert paths["a"]["t"].length == -90, "Length of path a ~> t should be -90."
        assert paths["s"]["a"].length == INFINITY, "Length of path s ~> a should be infinity."
        assert paths["z"]["a"].length == INFINITY, "Length of path z ~> a should be infinity."
        assert not paths["s"][
            "a"
        ].is_destination_reachable(), "'a' should not be reachable from 's'."

        assert paths["s"]["s"].length == 0, "Length of s path should be 0."
        assert paths["s"]["t"].length == 10, "Length of path s ~> t should be 10."
        assert paths["s"]["x"].length == 7, "Length of path s ~> x should be 7."
        assert paths["s"]["y"].length == 4, "Length of path s ~> y should be 4."
        assert paths["s"]["z"].length == 1, "Length of path s ~> z should be 1."

        assert paths["z"]["s"].length == 7, "Length of path z ~> s should be 7."
        assert paths["z"]["t"].length == 17, "Length of path z ~> t should be 17."
        assert paths["z"]["x"].length == 6, "Length of path z ~> x should be 6."
        assert paths["z"]["y"].length == 11, "Length of path z ~> y should be 11."
        assert paths["z"]["z"].length == 0, "Length of path z ~> z should be 0."


class TestWeightedSingleSourceShortestPaths:
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

        paths: VertexDict[ShortestPath] = shortest_paths_bellman_ford(g, "s")

        assert len(paths) == 5, "Shortest paths dictionary should have length equal to |V|."
        assert paths["s"].length == 0, "Length of s path should be 0."
        assert paths["t"].length == 8, "Length of path s ~> t should be 8."
        assert paths["x"].length == 9, "Length of path s ~> x should be 9."
        assert paths["y"].length == 5, "Length of path s ~> y should be 5."
        assert paths["z"].length == 7, "Length of path s ~> z should be 7."

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

        paths: VertexDict[ShortestPath] = shortest_paths_bellman_ford(g, "s")

        assert len(paths) == 5, "Shortest paths dictionary should have length equal to |V|."
        assert paths["s"].length == 0, "Length of s path should be 0."
        assert paths["t"].length == 10, "Length of path s ~> t should be 10."
        assert paths["x"].length == 7, "Length of path s ~> x should be 7."
        assert paths["y"].length == 4, "Length of path s ~> y should be 4."
        assert paths["z"].length == 1, "Length of path s ~> z should be 1."

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

        paths: VertexDict[ShortestPath] = shortest_paths_bellman_ford(
            g, "s", find_path_lengths_only=False
        )

        assert paths["t"].path == ["s", "t"], "Path s ~> t should be [s, t]."
        assert paths["x"].path == [
            "s",
            "t",
            "y",
            "z",
            "x",
        ], "Path s ~> x should be [s, t, y, z, x]."
        assert paths["z"].path == ["s", "t", "y", "z"], "Path s ~> z should be [s, t, y, z]."

        path_s_t = reconstruct_path("s", "t", paths)
        assert path_s_t == paths["t"].path, "Algorithm path should match reconstructed path."
        path_s_x = reconstruct_path("s", "x", paths)
        assert path_s_x == paths["x"].path, "Algorithm path should match reconstructed path."
        path_s_z = reconstruct_path("s", "z", paths)
        assert path_s_z == paths["z"].path, "Algorithm path should match reconstructed path."

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

        paths: VertexDict[ShortestPath] = shortest_paths_bellman_ford(g, "s", reverse_graph=True)

        assert len(paths) == 5, "Shortest paths dictionary should have length equal to |V|."
        assert paths["s"].length == 0, "Length of s path should be 0."
        assert paths["t"].length == -2, "Length of path s ~> t should be -2."
        assert paths["x"].length == 11, "Length of path s ~> x should be 11."
        assert paths["y"].length == 4, "Length of path s ~> y should be 4."
        assert paths["z"].length == 7, "Length of path s ~> z should be 7."

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
            shortest_paths_bellman_ford(g, "s")

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
            shortest_paths_bellman_ford(g, "s")

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

        paths: VertexDict[ShortestPath] = shortest_paths_bellman_ford(g, "s")

        assert len(paths) == 5, "Shortest paths dictionary should have length equal to |V|."
        assert paths["s"].length == 0, "Length of s path should be 0."
        assert paths["t"].length == 10, "Length of path s ~> t should be 10."
        assert paths["x"].length == 9, "Length of path s ~> x should be 9."
        assert paths["y"].length == 5, "Length of path s ~> y should be 5."
        assert paths["z"].length == 7, "Length of path s ~> z should be 7."

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

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(g, "s")

        assert len(paths) == 5, "Shortest paths dictionary should have length equal to |V|."
        assert paths["s"].length == 0, "Length of s path should be 0."
        assert paths["t"].length == 8, "Length of path s ~> t should be 8."
        assert paths["x"].length == 9, "Length of path s ~> x should be 9."
        assert paths["y"].length == 5, "Length of path s ~> y should be 5."
        assert paths["z"].length == 7, "Length of path s ~> z should be 7."

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

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(
            g, "s", find_path_lengths_only=False
        )

        assert paths["t"].path == ["s", "y", "t"], "Path s ~> t should be [s, y, t]."
        assert paths["x"].path == ["s", "y", "t", "x"], "Path s ~> x should be [s, y, t, x]."
        assert paths["z"].path == ["s", "y", "z"], "Path s ~> z should be [s, y, z]."

        path_s_t = reconstruct_path("s", "t", paths)
        assert path_s_t == paths["t"].path, "Algorithm path should match reconstructed path."
        path_s_x = reconstruct_path("s", "x", paths)
        assert path_s_x == paths["x"].path, "Algorithm path should match reconstructed path."
        path_s_z = reconstruct_path("s", "z", paths)
        assert path_s_z == paths["z"].path, "Algorithm path should match reconstructed path."

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
        g["s"]["t"].attr[WEIGHT] = 10
        g["s"]["y"].attr[WEIGHT] = 5
        g["t"]["y"].attr[WEIGHT] = 2
        g["t"]["x"].attr[WEIGHT] = 1
        g["x"]["z"].attr[WEIGHT] = 4
        g["y"]["t"].attr[WEIGHT] = 3
        g["y"]["x"].attr[WEIGHT] = 9
        g["y"]["z"].attr[WEIGHT] = 2
        g["z"]["s"].attr[WEIGHT] = 7
        g["z"]["x"].attr[WEIGHT] = 6

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(g, "s", weight=WEIGHT)

        assert len(paths) == 5, "Shortest paths dictionary should have length equal to |V|."
        assert paths["s"].length == 0, "Length of s path should be 0."
        assert paths["t"].length == 8, "Length of path s ~> t should be 8."
        assert paths["x"].length == 9, "Length of path s ~> x should be 9."
        assert paths["y"].length == 5, "Length of path s ~> y should be 5."
        assert paths["z"].length == 7, "Length of path s ~> z should be 7."

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

        g["s"]["t"].attr[COLOR] = "RED"
        g["s"]["y"].attr[COLOR] = "BLUE"
        g["t"]["y"].attr[COLOR] = "RED"
        g["t"]["x"].attr[COLOR] = "RED"
        g["x"]["z"].attr[COLOR] = "RED"
        g["y"]["t"].attr[COLOR] = "BLUE"
        g["y"]["x"].attr[COLOR] = "RED"
        g["y"]["z"].attr[COLOR] = "BLUE"
        g["z"]["s"].attr[COLOR] = "BLUE"
        g["z"]["x"].attr[COLOR] = "BLUE"

        # Exclude blue edges.
        def get_min_weight(v1: Vertex, v2: Vertex, reverse_graph: bool) -> float:
            graph = v1._parent_graph
            if reverse_graph:
                edge: EdgeType = graph[v2][v1]
                edge_str = f"({v2.label}, {v1.label})"
            else:
                edge: EdgeType = graph[v1][v2]
                edge_str = f"({v1.label}, {v2.label})"
            if edge is None:
                raise ValueError(f"graph does not have edge {edge_str}")
            if edge.attr.get(COLOR, "RED") == "BLUE":
                return None
            return edge.weight

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(g, "s", weight=get_min_weight)
        assert paths["s"].length == 0, "Length of s path should be 0."
        assert paths["t"].length == 10, "Length of path s ~> t should be 10."
        assert paths["x"].length == 11, "Length of path s ~> x should be 11."
        assert paths["y"].length == 12, "Length of path s ~> y should be 12."
        assert paths["z"].length == 15, "Length of path s ~> z should be 15."

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

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(g, "s", reverse_graph=True)
        assert paths["s"].length == 0, "Length of s path should be 0."
        assert paths["t"].length == 11, "Length of path s ~> t should be 11."
        assert paths["x"].length == 11, "Length of path s ~> x should be 11."
        assert paths["y"].length == 9, "Length of path s ~> y should be 9."
        assert paths["z"].length == 7, "Length of path s ~> z should be 7."

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

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(g, "s")
        assert paths["s"].length == 0, "Length of s path should be 0."
        assert paths["t"].length == 7, "Length of path s ~> t should be 7."
        assert paths["x"].length == 8, "Length of path s ~> x should be 8."
        assert paths["y"].length == 5, "Length of path s ~> y should be 5."
        assert paths["z"].length == 7, "Length of path s ~> z should be 7."

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

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra_fibonacci(g, "s")

        assert len(paths) == 5, "Shortest paths dictionary should have length equal to |V|."
        assert paths["s"].length == 0, "Length of s path should be 0."
        assert paths["t"].length == 8, "Length of path s ~> t should be 8."
        assert paths["x"].length == 9, "Length of path s ~> x should be 9."
        assert paths["y"].length == 5, "Length of path s ~> y should be 5."
        assert paths["z"].length == 7, "Length of path s ~> z should be 7."
