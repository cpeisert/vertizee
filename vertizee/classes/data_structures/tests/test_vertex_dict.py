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

"""Tests for VertexDict container."""
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

from vertizee.classes.graph import Graph
from vertizee.classes.vertex import Vertex
from vertizee.classes.data_structures.vertex_dict import VertexDict


class TestVertexDict:
    """Tests for VertexDict container."""

    def test_init(self):
        d1 = VertexDict()
        d1[1] = "one"
        assert len(d1) == 1, "Dict d1 should have 1 item"

        d2 = VertexDict(d1)
        assert len(d2) == 1, "Dict d2 should have 1 item after initializing from d1"

        pairs = [(1, "one"), (2, "two")]
        d3 = VertexDict(pairs)
        assert len(d3) == 2, "Dict d3 should have 2 items"

    def test_contains(self):
        d = VertexDict(**{"1":"one", "2":"two"})
        assert 1 in d
        assert "1" in d
        assert d.__contains__(2)

        g = Graph([(0, "six")])
        d2 = VertexDict()
        d2[g[0]] = "zero"
        d2[g["six"]] = "six"
        assert 0 in d2
        assert g[0] in d2
        assert "six" in d2
        assert g["six"] in d2

    def test__getitem__setitem(self):
        g = Graph()
        v1: Vertex = g.add_vertex(1)
        pairs = [("1", "one"), (2, "two")]
        d1 = VertexDict(pairs)

        assert d1[v1] == "one", "Dict d1 getitem should work with Vertex object key"
        assert d1[1] == "one", "Dict d1 getitem should work with int key"
        assert d1["1"] == "one", "Dict d1 getitem should work with int key"
        assert d1["2"] == "two", "Dict d1 should have item associated with key 2"
        d1[2] = "new value"
        assert d1["2"] == "new value", 'Dict d1 should have "new value" for key 2'

    def test_update(self):
        pairs = [("1", "one"), (2, "two")]
        d1 = VertexDict(special="special value")

        assert d1["special"] == "special value", "Dict d1 should have special value"
        assert len(d1) == 1, "Dict d1 should have length 1"

        d1.update(pairs)
        assert len(d1) == 3, "Dict d1 should have length 3 after update"
        assert d1["2"] == "two", "Dict d1 should have value associated with key 2"
