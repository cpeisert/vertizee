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

"""Tests for algorithms that find components."""
# pylint: disable=no-self-use
# pylint: disable=missing-function-docstring

from typing import List

import pytest

from vertizee import exception
from vertizee.algorithms.connectivity import components
from vertizee.algorithms.connectivity.components import Component
from vertizee.classes.edge import Edge, MultiDiEdge
from vertizee.classes.graph import Graph, MultiDiGraph
from vertizee.classes.vertex import MultiDiVertex, Vertex


class TestConnectedComponents:
    """Tests for algorithms that find the connected components in graphs."""

    def test_component(self) -> None:
        g = Graph([(1, 2), (2, 3), (4, 5), (7, 7)])
        component_list: List[Component[Vertex, Edge]] = list(components.connected_components(g))
        c_123: Component[Vertex, Edge] = [c for c in component_list if 1 in c][0]
        assert not c_123._edge_set, "without accessing edges, `_edge_set` should not be initialized"

        c_123.edges()
        assert c_123._edge_set, "after accessing edges, `_edge_set` should be initialized"

        c_45 = None
        c_77 = None
        for component in component_list:
            if (4, 5) in component:
                c_45 = component
            elif (7, 7) in component:
                c_77 = component

        assert c_45 is not None
        assert c_77 is not None
        assert (
            c_45._edge_set is not None
        ), "calling __contains__ should result in _edge_set initialization"
        assert (
            c_77._edge_set is not None
        ), "calling __contains__ should result in _edge_set initialization"
        assert g[7] in c_77, "vertex 7 should be in component containing edge (7, 7)"

    def test_connected_components(self) -> None:
        g = Graph([(1, 2), (2, 3), (4, 5), (7, 7)])
        g.add_vertex(8)
        component_list: List[Component[Vertex, Edge]] = list(components.connected_components(g))
        assert len(component_list) == 4, "graph should have 4 components"
        edge_count = sum(len(list(component.edges())) for component in component_list)
        assert edge_count == 4, "components should contain a grand total of 4 edges"

        mg: MultiDiGraph = get_example_multidigraph()
        scc_list: List[Component[MultiDiVertex, MultiDiEdge]] = list(
            components.connected_components(mg)
        )
        assert len(scc_list) == 4, "multidigraph should have 4 strongly-connected components"

    def test_exceptions(self) -> None:
        empty_g = Graph()
        with pytest.raises(exception.Unfeasible):
            components.connected_components(empty_g)

        g = Graph([(1, 2)])
        with pytest.raises(exception.GraphTypeNotSupported):
            components.strongly_connected_components(g)

    def test_kosaraju_topological_ordering(self) -> None:
        """Test that the strongly-connected components are output in topological order, meaning
        that for components Ci and Cj, if output_index[Ci] < output_index[Cj], then there exists
        and edge from Ci to Cj."""
        g: MultiDiGraph = get_example_multidigraph()

        sccs: List[Component[MultiDiVertex, MultiDiEdge]] = list(
            components.strongly_connected_components(g)
        )

        assert len(sccs) == 4, "Graph should have 4 strong-connected components."

        # Test that results of Kosaraju implementation are returned in reverse topological order.
        assert g["a"] in sccs[3].vertices()
        assert g["c"] in sccs[2].vertices()
        assert g["f"] in sccs[1].vertices() or g["h"] in sccs[1].vertices()
        assert g["f"] in sccs[0].vertices() or g["h"] in sccs[0].vertices()

    def test_strongly_connected_components(self) -> None:
        g: MultiDiGraph = get_example_multidigraph()
        scc_list = list(components.strongly_connected_components(g))

        assert len(scc_list) == 4, "should be 4 strongly-connected components"
        assert max(len(list(scc.edges())) for scc in scc_list) == 3

        scc_abe = scc_cd = scc_fg = scc_h = None
        for scc in scc_list:
            if g["a"] in scc:
                scc_abe = scc
            elif g["c"] in scc:
                scc_cd = scc
            elif g["f"] in scc:
                scc_fg = scc
            else:
                scc_h = scc
        assert scc_abe is not None
        assert scc_cd is not None
        assert scc_fg is not None
        assert scc_h is not None
        assert len(scc_abe) == 3, "SCC 'abe' should have 3 vertices."
        assert len(scc_cd) == 2, "SCC 'cd' should have 2 vertices."
        assert len(scc_fg) == 2, "SCC 'fg' should have 2 vertices."
        assert len(scc_h) == 1, "SCC 'h' should have 1 vertex."

    def test_weakly_connected_components(self) -> None:
        g: MultiDiGraph = get_example_multidigraph()
        scc_list = list(components.weakly_connected_components(g))
        assert len(scc_list) == 1, "graph should have one weakly-connected component"

        g2 = MultiDiGraph([(1, 2), (3, 2), (4, 5)])
        assert len(list(components.weakly_connected_components(g2))) == 2
        assert len(list(components.strongly_connected_components(g2))) == 5


def get_example_multidigraph() -> MultiDiGraph:
    """This graph is from "Introduction to Algorithms: Third Edition", page 616. It contains
    four strongly-connected components."""
    g = MultiDiGraph()
    g.add_edges_from(
        [
            ("a", "b"),
            ("b", "e"),
            ("e", "a"),
            ("e", "f"),
            ("b", "f"),
            ("b", "c"),
            ("c", "d"),
            ("d", "c"),
            ("c", "g"),
            ("d", "h"),
            ("h", "h"),
            ("f", "g"),
            ("g", "f"),
        ]
    )
    return g
