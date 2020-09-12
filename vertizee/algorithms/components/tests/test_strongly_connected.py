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

"""Tests algorithms for strongly-connected components of directed graphs."""

from typing import List

import pytest

from vertizee.algorithms.components.strongly_connected import kosaraju_strongly_connected_components
from vertizee.algorithms.search.depth_first_search import (
    DepthFirstSearchTree,
    dfs_postorder_traversal,
)
from vertizee.classes.digraph import MultiDiGraph

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip."
)


@pytest.mark.usefixtures()
class TestStronglyConnectedComponents:
    def test_kosaraju_scc(self):
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

        sccs: List[DepthFirstSearchTree] = kosaraju_strongly_connected_components(g)

        assert len(sccs) == 4, "Graph should have 4 strong-connected components."
        for scc in sccs:
            if g["a"] in scc.vertices:
                scc_abe = scc
            elif g["c"] in scc.vertices:
                scc_cd = scc
            elif g["f"] in scc.vertices:
                scc_fg = scc
            else:
                scc_h = scc
        assert len(scc_abe.vertices) == 3, "SCC abe should have 3 vertices."
        assert len(scc_cd.vertices) == 2, "SCC cd should have 2 vertices."
        assert len(scc_fg.vertices) == 2, "SCC fg should have 2 vertices."
        assert len(scc_h.vertices) == 1, "SCC h should have 1 vertex."

        assert len(scc_abe.edges_in_discovery_order) == 2, "SCC abe should have 2 tree edges."
        assert len(scc_cd.edges_in_discovery_order) == 1, "SCC cd should have 1 tree edges."
        assert len(scc_fg.edges_in_discovery_order) == 1, "SCC fg should have 1 tree edges."
        assert len(scc_h.edges_in_discovery_order) == 0, "SCC h should have 0 tree edges."

    def test_kosaraju_topological_order(self):
        """Test that the strongly-connected components are output in topological order, meaning
        that for components Ci and Cj, if output_index[Ci] < output_index[Cj], then there exists
        and edge from Ci to Cj."""
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

        sccs: List[DepthFirstSearchTree] = kosaraju_strongly_connected_components(g)

        assert len(sccs) == 4, "Graph should have 4 strong-connected components."
        for scc in sccs:
            if g["a"] in scc.vertices:
                scc_abe = scc
            elif g["c"] in scc.vertices:
                scc_cd = scc
            elif g["f"] in scc.vertices:
                scc_fg = scc
            else:
                scc_h = scc

        # Create condensation subgraph, where each scc becomes a vertex.
        condensation = MultiDiGraph()

        for edge in g.edges:
            if edge.tail in scc_abe.vertices:
                tail = "abe"
            elif edge.tail in scc_cd.vertices:
                tail = "cd"
            elif edge.tail in scc_fg.vertices:
                tail = "fg"
            else:
                tail = "h"

            if edge.head in scc_abe.vertices:
                head = "abe"
            elif edge.head in scc_cd.vertices:
                head = "cd"
            elif edge.head in scc_fg.vertices:
                head = "fg"
            else:
                head = "h"

            if tail != head:
                condensation.add_edge(tail, head)

        postorder = list(dfs_postorder_traversal(condensation))

        # Topological sort of condensation: abe, cd, fg, h (reversed: h, fg, cd, abe)
        assert postorder[3] == "abe"
        assert postorder[2] == "cd"
        assert postorder[1] == "fg" or postorder[1] == "h"
        assert postorder[0] == "fg" or postorder[0] == "h"

        # Test that results of Kosaraju implementation are returned in reverse topological order.
        assert g["a"] in sccs[3].vertices
        assert g["c"] in sccs[2].vertices
        assert g["f"] in sccs[1].vertices or g["h"] in sccs[1].vertices
        assert g["f"] in sccs[0].vertices or g["h"] in sccs[0].vertices
