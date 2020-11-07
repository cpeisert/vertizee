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

"""Tests for PriorityQueue container."""
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

import pytest

import vertizee as vz
from vertizee import Vertex
from vertizee.classes.data_structures.priority_queue import PriorityQueue


def get_priority_function(vertex_to_priority: vz.VertexDict):
    """Returns a function that retrieves the priority of a vertex."""

    def priority_function(vertex: Vertex) -> float:
        """Returns the priority of the vertex."""
        return vertex_to_priority[vertex]
    return priority_function


class TestPriorityQueue:
    """Tests for PriorityQueue data structure."""

    def test_basic_operations(self):
        g = vz.Graph([(1, 2), (2, 3), (3, 4), (4, 5)])
        vertex_priority = vz.VertexDict()
        vertex_priority[1] = 100
        vertex_priority[2] = 90
        vertex_priority[3] = 80
        vertex_priority[4] = 70
        vertex_priority[5] = 70

        priority_function = get_priority_function(vertex_priority)

        vpq: PriorityQueue[Vertex] = PriorityQueue(priority_function)
        vpq.add_or_update(g[5])
        vpq.add_or_update(g[4])
        vpq.add_or_update(g[1])
        vpq.add_or_update(g[2])
        vpq.add_or_update(g[3])

        assert len(vpq) == 5, "Priority queue should contain 5 vertices."
        next_v = vpq.pop()
        assert next_v == g[5], "First lowest priority vertex should be vertex 5."

        next_v = vpq.pop()
        assert next_v == g[4], (
            "Second lowest priority vertex should be vertex 4, since 4 " " was inserted after 5."
        )
        vertex_priority[1] = 0
        vpq.add_or_update(g[1])
        assert len(vpq) == 3, (
            "Priority queue should contain 3 vertices after popping and " " updating."
        )
        next_v = vpq.pop()
        assert next_v == g[1], "Lowest priority vertex should be 1 after setting priority to 0."

        next_v = vpq.pop()
        assert next_v == g[3], "Lowest priority vertex should be 3."

        g.add_vertex(10)
        vertex_priority[10] = 200
        vpq.add_or_update(g[10])

        next_v = vpq.pop()
        assert next_v == g[2], "Lowest priority vertex should be 2."

        assert len(vpq) == 1, "Priority queue should contain 1 vertex."
        next_v = vpq.pop()
        assert next_v == g[10], "Lowest priority vertex should be 10."

        assert len(vpq) == 0, "Priority queue should be empty."

        # Attempting to pop an item from an empty priority queue should raise KeyError.
        with pytest.raises(KeyError):
            vpq.pop()
