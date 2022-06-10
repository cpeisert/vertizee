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

"""
==========
Exceptions
==========

Vertizee exceptions.
"""


class VertizeeException(Exception):
    """Base class for exceptions in Vertizee."""


class VertizeeError(VertizeeException):
    """Exception for a serious error in Vertizee."""


class AlgorithmError(VertizeeException):
    """Exception for unexpected termination of algorithms."""


class EdgeNotFound(VertizeeException):
    """Exception raised if indicated edge is not present in the graph"""


class Unfeasible(AlgorithmError):
    """Exception raised by algorithms trying to solve a problem instance that has no feasible
    solution."""


class GraphTypeNotSupported(Unfeasible):
    """Exception raised if an algorithm does not support (or is undefined) for a particular type
    of graph (for example, :term:`digraph`, :term:`graph`, :term:`multigraph`,
    :term:`multidigraph`)."""


class NegativeWeightCycle(Unfeasible):
    """Exception raised if a negative weight :term:`cycle` is found that precludes a solution.
    For example, see the :func:`Bellman-Ford algorithm
    <vertizee.algorithms.paths.single_source.bellman_ford>` for solving the single-source
    shortest-paths problem."""


class NoPath(Unfeasible):
    """Exception for algorithms that should return a :term:`path` when running on
    :term:`graphs <graph>` where such a path does not exist."""


class ParallelEdgesNotAllowed(VertizeeException):
    """Exception raised when attempting to add a :term:`parallel edge` to a :term:`graph` that is
    not a :term:`multigraph`."""


class SelfLoopsNotAllowed(VertizeeException):
    """Exception raised when attempting to add a :term:`self loop <loop>` to a :term:`graph`
    instantiated with ``allow_self_loops`` set to False."""


class VertexNotFound(VertizeeException):
    """Exception raised if indicated :term:`vertex` is not present in the :term:`graph`."""
