# This source code is from the scikit-learn project.
# Original source: https://github.com/scikit-learn/scikit-learn/blob/7782a91b5a23dd5e45bebc18739ce3a2c7b7d4d1/fibonacci.pxd
#
#
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
The definition of the Fibonacci heap data structure, which provide a fast
find-minimum operation needed for a number of algorithms such as Dijkstra's
algorithm for shortest graph path searches.
"""

cimport numpy as np
ctypedef np.float64_t DTYPE_t


######################################################################
# FibonacciNode structure
#  This structure and the operations on it are the nodes of the
#  Fibonacci heap.
#

cdef struct FibonacciNode:
    unsigned int index, rank, state
    DTYPE_t val
    FibonacciNode *parent
    FibonacciNode *left_sibling
    FibonacciNode *right_sibling
    FibonacciNode *children

ctypedef FibonacciNode* pFibonacciNode

cdef void initialize_node(FibonacciNode* node,
                          unsigned int index,
                          DTYPE_t val=*)

cdef FibonacciNode* rightmost_sibling(FibonacciNode* node)

cdef FibonacciNode* leftmost_sibling(FibonacciNode* node)

cdef void add_child(FibonacciNode* node, FibonacciNode* new_child)

cdef void add_sibling(FibonacciNode* node, FibonacciNode* new_sibling)

cdef void remove(FibonacciNode* node)


######################################################################
# FibonacciHeap structure
#  This structure and operations on it use the FibonacciNode
#  routines to implement a Fibonacci heap

cdef struct FibonacciHeap:
    FibonacciNode* min_node
    pFibonacciNode[100] roots_by_rank  # maximum number of nodes is ~2^100.

cdef void insert_node(FibonacciHeap* heap,
                      FibonacciNode* node)

cdef void decrease_val(FibonacciHeap* heap,
                       FibonacciNode* node,
                       DTYPE_t newval)

cdef void link(FibonacciHeap* heap, FibonacciNode* node)

cdef FibonacciNode* remove_min(FibonacciHeap* heap)
