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

"""Tests for algorithms that solve the all-pairs-shortest-paths problem."""
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

from typing import Final

import pytest

from vertizee import NegativeWeightCycle
from vertizee.algorithms.algo_utils.path_utils import reconstruct_path, ShortestPath
from vertizee.algorithms.paths.all_pairs_shortest_paths import (
    floyd_warshall,
    johnson,
    johnson_fibonacci
)
from vertizee.classes.data_structures.vertex_dict import VertexDict
from vertizee.classes.graph import DiGraph, MultiGraph

INFINITY: Final = float("inf")


class TestFloydWarshall:
    """Tests for the Floyd-Warshall algorithm."""
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

        all_paths_dict: VertexDict[VertexDict[ShortestPath]] = floyd_warshall(g)

        assert (
            len(all_paths_dict) == 5
        ), "all_paths_dict dictionary should have length equal to |V|"
        assert len(all_paths_dict["s"]) == 5, (
            "Each source in the shortest all_paths_dict dictionary should have "
            "a destinations dictionary of length equal to |V|."
        )
        assert all_paths_dict["s"]["s"].length == 0, "length of s ~> s path should be 0"
        assert all_paths_dict["s"]["t"].length == 8, "length of path s ~> t should be 8"
        assert all_paths_dict["s"]["x"].length == 9, "length of path s ~> x should be 9"
        assert all_paths_dict["s"]["y"].length == 5, "length of path s ~> y should be 5"
        assert all_paths_dict["s"]["z"].length == 7, "length of path s ~> z should be 7"

        assert all_paths_dict["y"]["t"].length == 3, "length of path y ~> t should be 3"
        assert all_paths_dict["y"]["x"].length == 4, "length of path y ~> x should be 4"
        assert all_paths_dict["y"]["s"].length == 9, "length of path y ~> s should be 9"
        assert all_paths_dict["y"]["z"].length == 2, "length of path y ~> z should be 2"

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

        all_paths_dict: VertexDict[VertexDict[ShortestPath]] = floyd_warshall(g)

        assert (
            len(all_paths_dict) == 6
        ), "all_paths_dict dictionary should have length equal to |V|."
        assert len(all_paths_dict["s"]) == 6, (
            "Each source in the shortest all_paths_dict dictionary should have "
            "a destinations dictionary of length equal to |V|."
        )

        assert all_paths_dict["a"]["s"].length == -100, "Length of path a ~> s should be -100."
        assert all_paths_dict["a"]["t"].length == -90, "Length of path a ~> t should be -90."
        assert (
            all_paths_dict["s"]["a"].length == INFINITY
        ), "Length of path s ~> a should be infinity."
        assert (
            all_paths_dict["z"]["a"].length == INFINITY
        ), "Length of path z ~> a should be infinity."
        assert not all_paths_dict["s"][
            "a"
        ].is_destination_reachable(), "'a' should not be reachable from 's'."

        assert all_paths_dict["s"]["s"].length == 0, "length of s path should be 0"
        assert all_paths_dict["s"]["t"].length == 10, "length of path s ~> t should be 10"
        assert all_paths_dict["s"]["x"].length == 7, "length of path s ~> x should be 7"
        assert all_paths_dict["s"]["y"].length == 4, "length of path s ~> y should be 4"
        assert all_paths_dict["s"]["z"].length == 1, "length of path s ~> z should be 1"

        assert all_paths_dict["z"]["s"].length == 7, "length of path z ~> s should be 7"
        assert all_paths_dict["z"]["t"].length == 17, "length of path z ~> t should be 17"
        assert all_paths_dict["z"]["x"].length == 6, "length of path z ~> x should be 6"
        assert all_paths_dict["z"]["y"].length == 11, "length of path z ~> y should be 11"
        assert all_paths_dict["z"]["z"].length == 0, "length of path z ~> z should be 0"

    def test_floyd_warshall_save_paths(self):
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

        all_paths_dict: VertexDict[VertexDict[ShortestPath]] = floyd_warshall(
            g, save_paths=False
        )

        assert all_paths_dict["s"]["z"].path() == []
        assert all_paths_dict["y"]["s"].path() == []
        assert all_paths_dict["y"]["t"].path() == []
        assert all_paths_dict["y"]["x"].path() == []
        assert all_paths_dict["y"]["y"].path() == []
        assert all_paths_dict["y"]["z"].path() == []

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

        all_paths_dict: VertexDict[VertexDict[ShortestPath]] = floyd_warshall(
            g, save_paths=True
        )

        assert all_paths_dict["s"]["z"].path() == ["s", "y", "z"]
        assert all_paths_dict["y"]["s"].path() == ["y", "z", "s"]
        assert all_paths_dict["y"]["t"].path() == ["y", "t"], "path y ~> t should be [y, t]"
        assert all_paths_dict["y"]["x"].path() == ["y", "t", "x"]
        assert all_paths_dict["y"]["y"].path() == ["y"], "path y ~> y should be [y]"
        assert all_paths_dict["y"]["z"].path() == ["y", "z"], "path y ~> z should be [y, z]"
        assert all_paths_dict["y"]["a"].path() == [], "path y ~> a should be []"

        path_s_z = reconstruct_path("s", "z", all_paths_dict)
        assert path_s_z == all_paths_dict["s"]["z"].path()
        path_y_s = reconstruct_path("y", "s", all_paths_dict)
        assert path_y_s == all_paths_dict["y"]["s"].path()
        path_y_t = reconstruct_path("y", "t", all_paths_dict)
        assert path_y_t == all_paths_dict["y"]["t"].path()
        path_y_x = reconstruct_path("y", "x", all_paths_dict)
        assert path_y_x == all_paths_dict["y"]["x"].path()
        path_y_y = reconstruct_path("y", "y", all_paths_dict)
        assert path_y_y == all_paths_dict["y"]["y"].path()
        path_y_z = reconstruct_path("y", "z", all_paths_dict)
        assert path_y_z == all_paths_dict["y"]["z"].path()
        path_y_a = reconstruct_path("y", "a", all_paths_dict)
        assert path_y_a == all_paths_dict["y"]["a"].path()

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
            floyd_warshall(g)

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
            floyd_warshall(g)

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

        all_paths_dict: VertexDict[VertexDict[ShortestPath]] = floyd_warshall(g)

        assert (
            len(all_paths_dict) == 5
        ), "all_paths_dict dictionary should have length equal to |V|."
        assert all_paths_dict["s"]["s"].length == 0, "length of path s ~> s should be 0"
        assert all_paths_dict["s"]["t"].length == 10, "length of path s ~> t should be 10"
        assert all_paths_dict["s"]["x"].length == 9, "length of path s ~> x should be 9"
        assert all_paths_dict["s"]["y"].length == 5, "length of path s ~> y should be 5"
        assert all_paths_dict["s"]["z"].length == 7, "length of path s ~> z should be 7"

        assert all_paths_dict["x"]["s"].length == 9, "length of path x ~> s should be 9"
        assert all_paths_dict["x"]["t"].length == 1, "length of path x ~> t should be 1"
        assert all_paths_dict["x"]["x"].length == 0, "length of path x ~> x should be 0"
        assert all_paths_dict["x"]["y"].length == 4, "length of path x ~> y should be 5"
        assert all_paths_dict["x"]["z"].length == 4, "length of path x ~> z should be 7"


class TestJohnson:
    """Tests for Johnson's algorithm."""
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

        all_paths_dict: VertexDict[VertexDict[ShortestPath]] = johnson(g)

        assert len(all_paths_dict) == 5, "all_paths_dict dictionary should have length equal to |V|"
        assert len(all_paths_dict["s"]) == 5, (
            "each source in the shortest all_paths_dict dictionary should have "
            "a destinations dictionary of length equal to |V|"
        )
        assert all_paths_dict["s"]["s"].length == 0, "length of s ~> s path should be 0"
        assert all_paths_dict["s"]["t"].length == 8, "length of path s ~> t should be 8"
        assert all_paths_dict["s"]["x"].length == 9, "length of path s ~> x should be 9"
        assert all_paths_dict["s"]["y"].length == 5, "length of path s ~> y should be 5"
        assert all_paths_dict["s"]["z"].length == 7, "length of path s ~> z should be 7"

        assert all_paths_dict["y"]["t"].length == 3, "length of path y ~> t should be 3"
        assert all_paths_dict["y"]["x"].length == 4, "length of path y ~> x should be 4"
        assert all_paths_dict["y"]["s"].length == 9, "length of path y ~> s should be 9"
        assert all_paths_dict["y"]["z"].length == 2, "length of path y ~> z should be 2"

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

        all_paths_dict: VertexDict[VertexDict[ShortestPath]] = johnson(g)

        assert len(all_paths_dict) == 6, "all_paths_dict dictionary should have length equal to |V|"
        assert len(all_paths_dict["s"]) == 6, (
            "each source in the shortest all_paths_dict dictionary should have "
            "a destinations dictionary of length equal to |V|"
        )

        assert all_paths_dict["a"]["s"].length == -100, "length of path a ~> s should be -100"
        assert all_paths_dict["a"]["t"].length == -90, "length of path a ~> t should be -90"
        assert all_paths_dict["s"]["a"].length == INFINITY
        assert all_paths_dict["z"]["a"].length == INFINITY
        assert not all_paths_dict["s"][
            "a"
        ].is_destination_reachable(), "'a' should not be reachable from 's'"

        assert all_paths_dict["s"]["s"].length == 0, "length of s path should be 0."
        assert all_paths_dict["s"]["t"].length == 10, "length of path s ~> t should be 10"
        assert all_paths_dict["s"]["x"].length == 7, "length of path s ~> x should be 7"
        assert all_paths_dict["s"]["y"].length == 4, "length of path s ~> y should be 4"
        assert all_paths_dict["s"]["z"].length == 1, "length of path s ~> z should be 1"

        assert all_paths_dict["z"]["s"].length == 7, "length of path z ~> s should be 7"
        assert all_paths_dict["z"]["t"].length == 17, "length of path z ~> t should be 17"
        assert all_paths_dict["z"]["x"].length == 6, "length of path z ~> x should be 6"
        assert all_paths_dict["z"]["y"].length == 11, "length of path z ~> y should be 11"
        assert all_paths_dict["z"]["z"].length == 0, "length of path z ~> z should be 0"

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

        all_paths_dict: VertexDict[VertexDict[ShortestPath]] = johnson(g, save_paths=True)

        assert all_paths_dict["s"]["z"].path() == ["s", "y", "z"], "path s ~> z should be [s, y, z]"
        assert all_paths_dict["y"]["y"].path() == ["y"], "path y ~> y should be [y]"
        assert all_paths_dict["y"]["z"].path() == ["y", "z"], "path y ~> z should be [y, z]"
        assert all_paths_dict["y"]["a"].path() == [], "path y ~> a should be []"

        path_s_z = reconstruct_path("s", "z", all_paths_dict)
        assert path_s_z == all_paths_dict["s"]["z"].path()
        path_y_y = reconstruct_path("y", "y", all_paths_dict)
        assert path_y_y == all_paths_dict["y"]["y"].path()
        path_y_z = reconstruct_path("y", "z", all_paths_dict)
        assert path_y_z == all_paths_dict["y"]["z"].path()
        path_y_a = reconstruct_path("y", "a", all_paths_dict)
        assert path_y_a == all_paths_dict["y"]["a"].path()

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
            johnson(g)

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

        all_paths_dict: VertexDict[VertexDict[ShortestPath]] = johnson_fibonacci(g)

        assert len(all_paths_dict) == 6, "all_paths_dict dictionary should have length equal to |V|"
        assert len(all_paths_dict["s"]) == 6, (
            "each source in the shortest all_paths_dict dictionary should have "
            "a destinations dictionary of length equal to |V|"
        )

        assert all_paths_dict["a"]["s"].length == -100, "length of path a ~> s should be -100"
        assert all_paths_dict["a"]["t"].length == -90, "length of path a ~> t should be -90"
        assert all_paths_dict["s"]["a"].length == INFINITY
        assert all_paths_dict["z"]["a"].length == INFINITY
        assert not all_paths_dict["s"][
            "a"
        ].is_destination_reachable(), "'a' should not be reachable from 's'"

        assert all_paths_dict["s"]["s"].length == 0, "length of s path should be 0."
        assert all_paths_dict["s"]["t"].length == 10, "length of path s ~> t should be 10."
        assert all_paths_dict["s"]["x"].length == 7, "length of path s ~> x should be 7"
        assert all_paths_dict["s"]["y"].length == 4, "length of path s ~> y should be 4"
        assert all_paths_dict["s"]["z"].length == 1, "length of path s ~> z should be 1"

        assert all_paths_dict["z"]["s"].length == 7, "length of path z ~> s should be 7"
        assert all_paths_dict["z"]["t"].length == 17, "length of path z ~> t should be 17"
        assert all_paths_dict["z"]["x"].length == 6, "length of path z ~> x should be 6"
        assert all_paths_dict["z"]["y"].length == 11, "length of path z ~> y should be 11"
        assert all_paths_dict["z"]["z"].length == 0, "length of path z ~> z should be 0"
