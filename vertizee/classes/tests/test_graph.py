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

"""Tests for undirected graphs: Graph, MultiGraph, SimpleGraph."""

from collections import Counter
from typing import List

import pytest

from vertizee.classes.edge import DEFAULT_WEIGHT
from vertizee.classes.graph import Graph, MultiGraph, SimpleGraph
from vertizee.classes.vertex import Vertex

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip."
)


@pytest.mark.usefixtures()
class TestUndirectedGraphs:
    def test_vertex(self):
        g = Graph()
        v0 = g.add_vertex("0")
        assert v0.label == "0", f"Vertex v0 should have label '0', but had label '{v0.label}'"
        assert v0.degree == 0, f"Vertex v0 should have degree 0, but had degree {v0.degree}"
        assert len(v0.edges) == 0, "Vertex v0 should have no incoming edges."

        v1 = g.add_vertex("1")
        assert v1.label == "1", f"Vertex v1 should have label '1', but had label '{v1.label}'"
        assert v1.degree == 0, f"Vertex v1 should have degree 0, but had degree {v1.degree}"

        g.add_edge(0, 0)
        assert g[0].degree == 2, "Vertex 0 should have degree '2', since loops count twice."
        assert next(iter(g[0].adjacent_vertices)) == g[0], "Vertex 0 should be adjacent to itself."
        assert g[0].loops == {g[0, 0]}, "Loop edge on vertex should equal the same edge in graph."
        assert g[0].edges == {g[0, 0]}, "Vertex 0 adjacent edges should include the self loop."

    def test_edge(self):
        g = MultiGraph()
        v0 = g.add_vertex("0")
        v1 = g.add_vertex("1")
        e1 = g.add_edge(v0, v1)
        e_loop = g.add_edge(v0, v0)  # Create self-loop.

        e1_str = e1.__str__()
        assert e1_str.find("0") < e1_str.find("1"), 'Edge e1 __str__() should have "0" before "1".'
        assert (
            e1.weight == DEFAULT_WEIGHT
        ), f"Edge e1 should have weight {DEFAULT_WEIGHT} (default), but weight was {e1.weight}"
        assert e1.parallel_edge_count == 0, "Edge e1 should have zero parallel edges."
        assert (
            e1.vertex1 == v0
        ), f"Edge e1 should have vertex1 ({v0.label}), but it was ({e1.vertex1.label})"
        assert (
            e1.vertex2 == v1
        ), f"Edge e1 should have vertex2 ({v1.label}), but it was ({e1.vertex2.label})"
        assert e_loop.is_loop(), "Edge e_loop should be a loop."
        assert v0.degree == 3, "Vertex v0 should have degree 3."
        assert len(v0.non_loop_edges) == 1, "Vertex v0 should have 1 non-loop edge."
        assert len(v0.adjacent_vertices) == 2, "Vertex v0 should have 2 adjacent vertices."
        assert v0.adjacent_vertices == {
            g[0],
            g[1],
        }, "Vertex v0 should be adjacent to itself and vertex 1."

        v2 = g.add_vertex("2")
        e2 = g.add_edge(v1, v2, weight=1.5, parallel_edge_count=1, parallel_edge_weights=[3])
        assert e2.weight == 1.5, "Edge e2 should have weight 1.5."
        assert (
            e2.weight_with_parallel_edges == 4.5
        ), "Edge e2 should have total weight 4.5 including parallel edges."
        assert e2.parallel_edge_count == 1, "Edge e2 should have 1 parallel edge."
        e3 = g.add_edge(3, 4, 10.5)
        assert e3.__str__() == (
            "(3, 4, 10.5)"
        ), "Weighted edges should include weight in string representation."

        g2 = Graph([(1, 2)])
        assert g2[1, 2].__str__() == "(1, 2)", "Unweighted graphs should show edges with weights."

    def test_graph_initialization_and_parallel_edges(self):
        g = MultiGraph()
        v0 = g.add_vertex(0)
        v1 = g.add_vertex(1)
        v2 = g.add_vertex(2)
        v3 = g.add_vertex("3")
        v4 = g.add_vertex("4")

        _build_parallel_weighted_graph(g, [v0, v1, v2, v3, v4])

        assert (
            g.vertex_count == 5
        ), f"Graph should have 5 vertices, but had {g.vertex_count} vertices."
        assert g.edge_count == 115, f"Graph should have 115 edges, but had {g.edge_count} edges."
        assert g.edge_count_ignoring_parallel_edges() == 5, (
            f"Graph should have 5 edges (ignoring parallel edges), but had "
            f"{g.edge_count_ignoring_parallel_edges()}."
        )
        assert not g.current_state_is_simple_graph(), "Graph should not be a simple graph."

        deep_g = g.deepcopy()
        assert (
            deep_g.vertex_count == 5
        ), f"Copy of graph should have 5 vertices, but had {deep_g.vertex_count} vertices."
        assert (
            deep_g.edge_count == 115
        ), f"Copy of graph should have 115 edges, but had {deep_g.edge_count} edges."
        assert deep_g.edge_count_ignoring_parallel_edges() == 5, (
            f"Copy of graph should have 5 edges (ignoring parallel edges), but had "
            f"{deep_g.edge_count_ignoring_parallel_edges()}."
        )
        assert not deep_g.current_state_is_simple_graph(), "Graph should not be a simple graph."

        assert v0.degree == 119, f"Vertex v0 should have degree 119, but had degree {v0.degree}."
        assert v2.degree == 3, f"Vertex v2 should have degree 3, but had degree {v2.degree}."
        assert v4.degree == 0, f"Vertex v4 should have degree 0, but had degree {v4.degree}."
        assert v4 in g, "Graph should contain isolated vertex v4."
        assert v0 in g, "Graph should contain vertex with label 0."
        assert g.has_vertex(v0), "Graph should contain vertex with label 0."

        edge00 = g[0, 0]
        assert edge00.is_loop(), "Edge (0, 0) should be a loop."

        assert g.graph_weight == 4971.5, "Graph should have weight of 4971.5."
        assert (
            edge00.weight_with_parallel_edges == 10.5
        ), "Edge (0, 0) should have total weight 10.5, including parallel edges"

        g.remove_edge_from(edge00)
        assert (
            edge00.weight_with_parallel_edges == 6.5
        ), "Edge (0, 0) should have total weight 6.5, after deleting a parallel loop."
        assert (
            len(edge00.parallel_edge_weights) == 4
        ), "Edge (0, 0) should have 4 parallel edge weights."
        assert (
            edge00.parallel_edge_count == 4
        ), "Edge (0, 0) should have 4 parallel edges after deleting a parallel loop."

        assert v0.delete_loops() == 5, "Vertex v0 should have deleted 5 loops."

        edge01 = g.get_edge(v0, v1)
        # Test accessor notation.
        edge10 = g[1, 0]
        edge12 = g.get_edge(v1, v2)
        edge20 = g[2, 0]
        edge32 = g.get_edge(v3, v2)

        assert not g.has_edge(edge00), "Graph should not contain edge (0, 0)."
        assert g.has_edge(edge01), "Graph should not contain edge (0, 1)."
        assert g.has_edge(edge10), "Graph should not contain edge (1, 0)."
        assert edge01 == edge10, "Undirected graph edges (0, 1) and (1, 0) should be equal."

        assert g.has_edge(edge12), "Graph should not contain edge (1, 2)."
        assert g.has_edge(edge20), "Graph should not contain edge (2, 0)."
        assert g.has_edge(edge32), "Graph should not contain edge (3, 2)."

        assert g.graph_weight == 4961, "Graph should have weight 4961 after deleting loops on v0."
        assert g.edge_count == 109, "Graph should have 109 edges after deleting loops on v0."

        g.remove_all_edges_from(edge01)
        assert g.edge_count == 3, (
            "Graph should have 3 edges (including parallel edges) after deleting edges (0, 0) "
            " and (0, 1)."
        )

    def test_isolated_vertex_removal(self):
        g = MultiGraph([(0, 0), (0, 0), (1, 2), (3, 4)])
        g.add_vertex(5)
        count = g.remove_isolated_vertices()
        assert count == 2, "Should have removed two vertices."
        assert g[0] is None, "Vertex 0 should have been removed."
        assert g[5] is None, "Vertex 5 should have been removed."
        assert (
            g[1] is not None and g[2] is not None and g[3] is not None and g[4] is not None
        ), "Vertices with adjacent edges should not have been removed."
        count = g.remove_isolated_vertices()
        assert count == 0, "Should not have removed any vertices, since no vertices were isolated."
        g.add_vertex(10)
        count = g.remove_isolated_vertices()
        assert count == 1, "Should have removed isolated vertex 10."

    def test_iterators(self):
        g = MultiGraph()
        v0 = g.add_vertex(0)
        v1 = g.add_vertex(1)
        v2 = g.add_vertex(2)
        v3 = g.add_vertex(3)
        v4 = g.add_vertex(4)

        _build_parallel_weighted_graph(g, [v0, v1, v2, v3, v4])
        total_weight = 0
        edge_count = 0
        for incident_edge in v0:
            edge_count = edge_count + 1 + incident_edge.parallel_edge_count
            total_weight += incident_edge.weight_with_parallel_edges
        assert (
            total_weight == 4968.1
        ), "Iterating over v0 incident edges and summing weights should total 4968.1."
        assert edge_count == 113, "Iterating over v0 incident edges should include 113 edges."

        total_degree = 0
        vertex_count = 0
        for vertex in g:
            vertex_count += 1
            total_degree += vertex.degree
        assert vertex_count == 5, "Iterating over vertices in graph should include 5 vertices."
        assert total_degree == (
            119 + 107 + 3 + 1 + 0
        ), "Iterating over vertices should yield total degree of 230."

    def test_vertex_merging(self):
        g = MultiGraph()
        v0 = g.add_vertex(0)
        v1 = g.add_vertex(1)
        v2 = g.add_vertex(2)
        v3 = g.add_vertex(3)
        v4 = g.add_vertex(4)

        _build_parallel_weighted_graph(g, [v0, v1, v2, v3, v4])
        g.remove_vertex(v4)  # Remove isolated vertex.

        pre_merge_degree = v0.degree + v1.degree
        assert (
            pre_merge_degree == 226
        ), "The total degree of v0 + v1 prior to merge should be 226 (119 + 107)."

        v1_old_edges = v1.edges

        g.merge_vertices(v0, v1)
        """
        POST CONDITIONS
        - Pre-merge total degree of deg(v0) + deg(v1) must equal new merged deg(v0).
        - incident edges of v1 updated so that v1 is replaced by v0
        - v1 is deleted from the graph
        """
        # Pre-merge total degree of deg(v0) + deg(v1) must equal new merged deg(v0).
        assert pre_merge_degree == v0.degree, (
            f"After merge, degree(v0) => {v0.degree} must equal pre-merge "
            f"value of degree(v0) + degree(v1) => {pre_merge_degree}."
        )

        # v1's incident edges updated so that v1 replaced by v0.
        for edge in v1_old_edges:
            if edge.vertex1 == v1:
                other_edge = edge.vertex2
            else:
                other_edge = edge.vertex1
            new_edge = v0.get_edge(v0, other_edge)
            assert new_edge is not None, (
                f"After merging v1 into v0, old edge (1, {other_edge}) should become "
                f" (0, {other_edge})"
            )

        # v1 is deleted from the graph
        assert v1 not in g, "After merging v1 into v0, v1 should not be in the graph."

        edge00 = v0.get_edge(v0, v0)
        assert edge00.parallel_edge_count == 111, (
            "After merging v1 into v0, there should be 112 loops on v0"
            " (i.e. 111 parallel edge loops plus the initial loop)."
        )

    def test_simple_graph_functionality(self):
        g = SimpleGraph()
        g.add_vertices_from("s0", "s1")
        g.add_edges_from([("s0", "s1"), ("s0", "s2"), ("s0", "s3"), ("s3", "s1")])

        # Attempting to add a loop to a simple graph should raise ValueError.
        with pytest.raises(ValueError):
            g.add_edge("s0", "s0")
        # Attempting to add a parallel edge to a simple graph should raise ValueError.
        with pytest.raises(ValueError):
            g.add_edge("s1", "s0")

        # MultiGraph - test simple graph state
        g = MultiGraph()
        g.add_edges_from([("s0", "s1"), ("s0", "s2"), ("s0", "s3"), ("s3", "s1")])
        vs0 = g["s0"]
        vs1 = g["s1"]
        vs3 = g["s3"]
        assert (
            g.current_state_is_simple_graph()
        ), "Graph should be a simple graph with no loops and no parallel edges."

        # Adding a loop to a simple graph should change it to a non-simple graph.
        g.add_edge(vs0, vs0)
        assert not g.current_state_is_simple_graph(), (
            "Graph should not be a simple graph after " " adding a loop."
        )
        g.convert_to_simple_graph()
        assert (
            g.current_state_is_simple_graph()
        ), "Graph should be a simple graph with no loops and no parallel edges."
        assert g.edge_count == 4, "Graph should have four edges after conversion to simple graph."
        assert len(g.edges) == 4, "Graph should have 4 edges after conversion to simple graph."

        # Adding a parallel edge to a simple graph should change it to a non-simple graph.
        g.add_edge(vs1, vs3)
        assert g.edge_count == 5, "Graph should have 5 edges after adding parallel edge."
        assert (
            not g.current_state_is_simple_graph()
        ), "Graph should not be a simple graph after adding a parallel edge."
        g.convert_to_simple_graph()
        assert (
            g.current_state_is_simple_graph()
        ), "Graph should be a simple graph with no parallel edges after conversion."
        assert g.edge_count == 4, "Graph should have 4 edges after conversion to simple graph."
        assert len(g.edges) == 4, "Graph should have 4 edges after conversion to simple graph."

    def test_graph_random_edge_sampling(self):
        g = MultiGraph()
        v0 = g.add_vertex(0)
        v1 = g.add_vertex(1)
        v2 = g.add_vertex(2)
        v3 = g.add_vertex(3)
        v4 = g.add_vertex(4)

        _build_parallel_weighted_graph(g, [v0, v1, v2, v3, v4])

        # Since there are 115 edges in the graph, and edge (0, 1) has 106 parallel edges,
        # edge (0, 1) should appear with frequency 106/115 (92.1%) on average when randomly
        # sampled.
        cnt = Counter()
        for _ in range(2000):
            rand_edge = g.get_random_edge()
            cnt[rand_edge] += 1
        total_samples = sum(cnt.values())
        edge01 = v0.get_edge(v0, v1)

        ###
        # edge00 = v0.get_edge(v0, v0)
        # edge12 = v1.get_edge(v1, v2)
        # edge32 = v3.get_edge(v3, v2)
        # print('\n\nRANDOM SAMPLE RESULTS')
        # print(f'(0,0) => {cnt[edge00] / total_samples}')
        # print(f'(0,1) => {cnt[edge01] / total_samples}')
        # print(f'(1,2) => {cnt[edge12] / total_samples}')
        # print(f'(3,2) => {cnt[edge32] / total_samples}')
        ###

        frequency = cnt[edge01] / total_samples
        assert frequency > 0.9, (
            "Random edge sampling should yield edge (0, 1) 92% of time due to 106 "
            f"parallel edges in a graph of 115 total edges. Actual frequency was {frequency}."
        )


def _build_parallel_weighted_graph(g: MultiGraph, v: List[Vertex]) -> MultiGraph:
    # (0, 0, 0.5)  sum(parallel_edge_weights) => 10
    g.add_edge(v[0], v[0], weight=0.5, parallel_edge_count=5, parallel_edge_weights=list(range(5)))
    # (0, 1, 0.5)  sum(parallel_edge_weights) => 4950
    g.add_edge(
        v[0], v[1], weight=0.5, parallel_edge_count=100, parallel_edge_weights=list(range(100))
    )
    # Add more parallel edges between v0 and v1.
    # (0, 1, 1.0)  sum(parallel_edge_weights) => 4957
    # Total parallel edges => 106
    g.add_edge(v[1], v[0], weight=1.0, parallel_edge_count=4, parallel_edge_weights=list(range(4)))
    # (1, 2, 1.5)
    g.add_edge(v[1], v[2], weight=1.5)
    # (2, 0, 0.1)
    g.add_edge(v[2], v[0], weight=0.1)
    # (3, 2, 1.9)
    g.add_edge(v[3], v[2], weight=1.9)
    # Isolated vertex
    g.add_vertex(v[4].label)

    return g
