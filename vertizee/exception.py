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

"""Vertizee exceptions."""


class VertizeeException(Exception):
    """Base class for exceptions in Vertizee."""


class VertizeeError(VertizeeException):
    """Exception for a serious error in Vertizee."""


class AlgorithmError(VertizeeException):
    """Exception for unexpected termination of algorithms."""


class Unfeasible(VertizeeException):
    """Exception raised by algorithms trying to solve a problem
    instance that has no feasible solution."""


class GraphTypeNotSupported(Unfeasible):
    """Exception raised if an algorithm does not support (or is undefined) for a particular type
    of graph (e.g. DiGraph, Graph, MultiGraph, MultiDiGraph)."""


class NegativeWeightCycle(Unfeasible):
    """Exception raised if a negative weight cycle is found that precludes a solution. For example,
    see the Bellman-Ford algorithm for solving the single-source shortest-paths problem."""


class NoPath(Unfeasible):
    """Exception for algorithms that should return a path when running
    on graphs where such a path does not exist."""


class VertexNotFound(VertizeeException):
    """Exception raised if requested vertex is not present in the graph"""
