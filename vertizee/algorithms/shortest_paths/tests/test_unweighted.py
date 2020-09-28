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

"""Tests for finding the shortest paths in unweighted graphs."""

import pytest

from vertizee.classes.collections.vertex_dict import VertexDict
from vertizee.classes.graph import Graph
from vertizee.algorithms.shortest_paths.unweighted import (
    shortest_paths_breadth_first_search,
    ShortestPath,
)

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip."
)

INFINITY = float("inf")


@pytest.mark.usefixtures()
class TestUnweighted:
    def test_shortest_paths(self):
        g = Graph()
        g.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (3, 5), (6, 7)])
        paths: VertexDict[ShortestPath] = shortest_paths_breadth_first_search(g, 0)

        assert (
            len(paths) == 8
        ), "Shortest paths dictionary should have length equal to number of vertices."
        assert not paths[
            6
        ].is_destination_reachable(), "Vertex 6 should be unreachable from vertex 0."
        assert paths[6].length == INFINITY, "Unreachable vertex should have path length infinity."
        assert paths[4].length == 3, "Length of shortest path from 0 -> 4 should be 3"
        assert paths[3].length == 2, "Length of shortest path from 0 -> 3 should be 2"

        # with open('DEBUG.txt', mode='w') as f:
        #     f.write(f'{paths[4].path}')
