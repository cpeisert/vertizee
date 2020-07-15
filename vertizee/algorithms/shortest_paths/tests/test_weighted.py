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
from vertizee.algorithms.shortest_paths.weighted \
    import shortest_paths_bellman_ford, shortest_paths_dijkstra, shortest_paths_dijkstra_fibonacci
from vertizee.classes.collections.vertex_dict import VertexDict
from vertizee.classes.digraph import DiGraph, MultiDiGraph
from vertizee.classes.edge import EdgeType
from vertizee.classes.graph import MultiGraph
from vertizee.classes.shortest_path import ShortestPath
from vertizee.classes.vertex import Vertex

pytestmark = pytest.mark.skipif(
    False, reason="Set first param to False to run tests, or True to skip.")


@pytest.mark.usefixtures()
class TestWeighted:

    def test_bellman_ford_default_edge_weight(self):
        g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])

        paths: VertexDict[ShortestPath] = shortest_paths_bellman_ford(g, 's')

        assert len(paths) == 5, 'Shortest paths dictionary should have length equal to |V|.'
        assert paths['s'].length == 0, 'Length of s path should be 0.'
        assert paths['t'].length == 8, 'Length of path s ~> t should be 8.'
        assert paths['x'].length == 9, 'Length of path s ~> x should be 9.'
        assert paths['y'].length == 5, 'Length of path s ~> x should be 5.'
        assert paths['z'].length == 7, 'Length of path s ~> z should be 7.'

    def test_bellman_ford_negative_edge_weights(self):
        g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', -6), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 8), ('y', 'x', 4), ('y', 'z', -3),
            ('z', 's', 7), ('z', 'x', 6)
        ])

        paths: VertexDict[ShortestPath] = shortest_paths_bellman_ford(g, 's')

        assert len(paths) == 5, 'Shortest paths dictionary should have length equal to |V|.'
        assert paths['s'].length == 0, 'Length of s path should be 0.'
        assert paths['t'].length == 10, 'Length of path s ~> t should be 10.'
        assert paths['x'].length == 7, 'Length of path s ~> x should be 7.'
        assert paths['y'].length == 4, 'Length of path s ~> x should be 4.'
        assert paths['z'].length == 1, 'Length of path s ~> z should be 1.'

    def test_bellman_ford_reverse_graph(self):
        g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', -6), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 8), ('y', 'x', 4), ('y', 'z', -3),
            ('z', 's', 7), ('z', 'x', 6)
        ])

        paths: VertexDict[ShortestPath] = shortest_paths_bellman_ford(g, 's', reverse_graph=True)

        assert len(paths) == 5, 'Shortest paths dictionary should have length equal to |V|.'
        assert paths['s'].length == 0, 'Length of s path should be 0.'
        assert paths['t'].length == -2, 'Length of path s ~> t should be -2.'
        assert paths['x'].length == 11, 'Length of path s ~> x should be 11.'
        assert paths['y'].length == 4, 'Length of path s ~> x should be 4.'
        assert paths['z'].length == 7, 'Length of path s ~> z should be 7.'

    def test_bellman_ford_negative_weight_cycle(self):
        g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', -6), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 8), ('y', 'x', 4), ('y', 'z', -3),
            ('z', 's', -2), ('z', 'x', 6)
        ])

        with pytest.raises(NegativeWeightCycle):
            shortest_paths_bellman_ford(g, 's')

    def test_bellman_ford_undirected_negative_weight_cycle(self):
        g = MultiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', -6), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 8), ('y', 'x', 4), ('y', 'z', -3),
            ('z', 's', 7), ('z', 'x', 6)
        ])

        with pytest.raises(NegativeWeightCycle):
            shortest_paths_bellman_ford(g, 's')

    def test_bellman_ford_undirected(self):
        g = MultiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 6), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 8), ('y', 'x', 4), ('y', 'z', 3),
            ('z', 's', 7), ('z', 'x', 6)
        ])

        paths: VertexDict[ShortestPath] = shortest_paths_bellman_ford(g, 's')

        assert len(paths) == 5, 'Shortest paths dictionary should have length equal to |V|.'
        assert paths['s'].length == 0, 'Length of s path should be 0.'
        assert paths['t'].length == 10, 'Length of path s ~> t should be 10.'
        assert paths['x'].length == 9, 'Length of path s ~> x should be 9.'
        assert paths['y'].length == 5, 'Length of path s ~> x should be 5.'
        assert paths['z'].length == 7, 'Length of path s ~> z should be 7.'

    def test_dijkstra_default_edge_weight(self):
        g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(g, 's')

        assert len(paths) == 5, 'Shortest paths dictionary should have length equal to |V|.'
        assert paths['s'].length == 0, 'Length of s path should be 0.'
        assert paths['t'].length == 8, 'Length of path s ~> t should be 8.'
        assert paths['x'].length == 9, 'Length of path s ~> x should be 9.'
        assert paths['y'].length == 5, 'Length of path s ~> x should be 5.'
        assert paths['z'].length == 7, 'Length of path s ~> z should be 7.'

    def test_dijkstra_edge_attr_weights(self):
        WEIGHT = 'weight_key'
        g = DiGraph([
            ('s', 't'), ('s', 'y'),
            ('t', 'y'), ('t', 'x'),
            ('x', 'z'),
            ('y', 't'), ('y', 'x'), ('y', 'z'),
            ('z', 's'), ('z', 'x')
        ])
        g['s']['t'].attr[WEIGHT] = 10
        g['s']['y'].attr[WEIGHT] = 5
        g['t']['y'].attr[WEIGHT] = 2
        g['t']['x'].attr[WEIGHT] = 1
        g['x']['z'].attr[WEIGHT] = 4
        g['y']['t'].attr[WEIGHT] = 3
        g['y']['x'].attr[WEIGHT] = 9
        g['y']['z'].attr[WEIGHT] = 2
        g['z']['s'].attr[WEIGHT] = 7
        g['z']['x'].attr[WEIGHT] = 6

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(g, 's', weight=WEIGHT)

        assert len(paths) == 5, 'Shortest paths dictionary should have length equal to |V|.'
        assert paths['s'].length == 0, 'Length of s path should be 0.'
        assert paths['t'].length == 8, 'Length of path s ~> t should be 8.'
        assert paths['x'].length == 9, 'Length of path s ~> x should be 9.'
        assert paths['y'].length == 5, 'Length of path s ~> x should be 5.'
        assert paths['z'].length == 7, 'Length of path s ~> z should be 7.'

    def test_dijkstra_edge_weight_filter_function(self):
        COLOR = 'color_key'

        g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])

        g['s']['t'].attr[COLOR] = 'RED'
        g['s']['y'].attr[COLOR] = 'BLUE'
        g['t']['y'].attr[COLOR] = 'RED'
        g['t']['x'].attr[COLOR] = 'RED'
        g['x']['z'].attr[COLOR] = 'RED'
        g['y']['t'].attr[COLOR] = 'BLUE'
        g['y']['x'].attr[COLOR] = 'RED'
        g['y']['z'].attr[COLOR] = 'BLUE'
        g['z']['s'].attr[COLOR] = 'BLUE'
        g['z']['x'].attr[COLOR] = 'BLUE'

        # Exclude blue edges.
        def get_min_weight(v1: Vertex, v2: Vertex, reverse_graph: bool) -> float:
            graph = v1._parent_graph
            if reverse_graph:
                edge: EdgeType = graph[v2][v1]
                edge_str = f'({v2.key}, {v1.key})'
            else:
                edge: EdgeType = graph[v1][v2]
                edge_str = f'({v1.key}, {v2.key})'
            if edge is None:
                raise ValueError(f'graph does not have edge {edge_str}')
            if edge.attr.get(COLOR, 'RED') == 'BLUE':
                return None
            return edge.weight

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(g, 's', weight=get_min_weight)
        assert paths['s'].length == 0, 'Length of s path should be 0.'
        assert paths['t'].length == 10, 'Length of path s ~> t should be 10.'
        assert paths['x'].length == 11, 'Length of path s ~> x should be 11.'
        assert paths['y'].length == 12, 'Length of path s ~> x should be 12.'
        assert paths['z'].length == 15, 'Length of path s ~> z should be 15.'

    def test_dijkstra_reverse_graph(self):
        g = MultiDiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(g, 's', reverse_graph=True)
        assert paths['s'].length == 0, 'Length of s path should be 0.'
        assert paths['t'].length == 11, 'Length of path s ~> t should be 11.'
        assert paths['x'].length == 11, 'Length of path s ~> x should be 11.'
        assert paths['y'].length == 9, 'Length of path s ~> x should be 9.'
        assert paths['z'].length == 7, 'Length of path s ~> z should be 7.'

    def test_dijkstra_undirected_graph(self):
        g = MultiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra(g, 's')
        assert paths['s'].length == 0, 'Length of s path should be 0.'
        assert paths['t'].length == 7, 'Length of path s ~> t should be 7.'
        assert paths['x'].length == 8, 'Length of path s ~> x should be 8.'
        assert paths['y'].length == 5, 'Length of path s ~> x should be 5.'
        assert paths['z'].length == 7, 'Length of path s ~> z should be 7.'

    def test_dijkstra_fibonacci_default_edge_weight(self):
        g = DiGraph([
            ('s', 't', 10), ('s', 'y', 5),
            ('t', 'y', 2), ('t', 'x', 1),
            ('x', 'z', 4),
            ('y', 't', 3), ('y', 'x', 9), ('y', 'z', 2),
            ('z', 's', 7), ('z', 'x', 6)
        ])

        paths: VertexDict[ShortestPath] = shortest_paths_dijkstra_fibonacci(g, 's')

        assert len(paths) == 5, 'Shortest paths dictionary should have length equal to |V|.'
        assert paths['s'].length == 0, 'Length of s path should be 0.'
        assert paths['t'].length == 8, 'Length of path s ~> t should be 8.'
        assert paths['x'].length == 9, 'Length of path s ~> x should be 9.'
        assert paths['y'].length == 5, 'Length of path s ~> x should be 5.'
        assert paths['z'].length == 7, 'Length of path s ~> z should be 7.'
