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

"""Burkhard-Keller tree data structure."""

from __future__ import annotations
from typing import Callable, Dict, Generic, List, TypeVar, Union

#:Type variable for values in a generic BKTree data structure.
T = TypeVar("T")

GC_DEFAULT_THRESHOLD = 0.3  # % of deleted nodes relative to tree size prior to garbage collection
GC_MIN_TREE_SIZE = 1000  # minimum tree size required before performing garbage collection


# TODO(cpeisert): redesign so that internal nodes are not exposed to the end user.
class BKNode(Generic[T]):
    """A BK-tree node.

    Args:
        key_value (:class:`T`): the value associated with the node used for calculating distances
            between other nodes in the metric space.

    Attributes:
        key_value (:class:`T`): The value associated with the node used for calculating distances between other
            nodes in the metric space.
        children: Dictionary where the keys are the non-negative integer distances between this
            node and child nodes and the values are the child nodes.
    """

    def __init__(self, key_value: T):
        self._deleted = False
        self.key_value: T = key_value
        self.children: Dict[int, BKNode] = {}

    def __contains__(self, distance: int) -> bool:
        return distance in self.children

    def __eq__(self, other: "BKNode"):
        if not isinstance(other, BKNode):
            return False
        return self.key_value == other.key_value

    def __getitem__(self, distance: int) -> "BKNode":
        """Support index accessor notation to retrieve child node based on its distance from this
        node."""
        return self.children[distance]

    def __hash__(self):
        return hash(self.key_value)

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)

    def __repr__(self):
        return f"BKNode({self.key_value})"

    def __setitem__(self, distance: int, child_node: "BKNode"):
        self.children[distance] = child_node

    def __str__(self):
        return f"BKNode({self.key_value})"


class BKNodeLabeled(BKNode[T]):
    """A labeled BK tree node, where each node has a unique string label as well as a value used
    to calculate the distance between other nodes in the metric space.

    Attributes:
        key_value (:class:`T`): The value associated with the node used for calculating distances
            between other nodes in the metric space.
        key_label: A string representing the node name (e.g. the name of a vertex in a graph).
    """

    def __init__(self, key_value: T, key_label: str):
        if key_label is None:
            raise KeyError("key_label was None")
        self.key_label: str = str(key_label)
        super().__init__(key_value)

    def __eq__(self, other: "BKNodeLabeled"):
        if not isinstance(other, BKNodeLabeled):
            return False
        return self.key_label == other.key_label

    def __getitem__(self, distance: int) -> "BKNodeLabeled":
        """Support index accessor notation to retrieve child node based on its distance from this
        node."""
        return self.children[distance]

    def __hash__(self):
        return hash(self.key_label)

    def __repr__(self):
        return f"BKNodeLabeled(key_label={self.key_label}, key_value={self.key_value})"

    def __setitem__(self, distance: int, child_node: "BKNodeLabeled"):
        self.children[distance] = child_node

    def __str__(self):
        return f"BKNodeLabeled(key_label={self.key_label}, key_value={self.key_value})"


class BKTree(Generic[T]):
    """A Burkhard-Keller tree data structure.

    A BK-tree is designed to perform efficient key queries that determine all keys in the tree that
    are closest to the query key as defined by some distance function :math:`d`. The distance
    function is the "metric" that defines the metric space over the set of possible keys :math:`K`.

    The distance function (metric) must satisfy the following properties:

        - :math:`\\forall{x,\\ y} \\in K,\\ d(x,\\ y) = 0 \\Longleftrightarrow x = y`
          |emsp| [identity of indiscernibles]
        - :math:`\\forall{x,\\ y} \\in K,\\ d(x,\\ y) =\\ d(y,\\ x)` |emsp| [symmetry]
        - :math:`\\forall{x,\\ y,\\ z} \\in K,\\ d(x,\\ z) \\leq d(x,\\ y) + d(y,\\ z)`
          |emsp| [triangle inequality]

    This implementation supports deletion through marking nodes deleted and then performing
    garbage collection once a minimum threshold is reached.

    Note:
        This data structure is based on the original paper
        :download:`"Some approaches to best-match file searching."
        </references/Burkhard-Keller_BK-Trees.pdf>` by Burkhard and Keller. [BK1973]_

    Args:
        distance_function: The function to calculate the distance between
            any two keys in the metric space.
        labeled_nodes: Optional; If True, use labeled nodes (BKNodeLabeled objects) that
            use both unique string labels (key_label) for each node as well as node values
            (key_value). If False (default), use unlabeled nodes (BKNode objects). Defaults to
            False.
        garbage_collection_threshold: Optional; Percentage of all nodes that must be
            marked deleted before removing them from the tree by rebuilding the tree from scratch
            with the non-deleted nodes. Defaults to 30%.

    Attributes:
        root: The root node of the BK tree.

    References:
     .. [BK1973] Burkhard, W.; Keller, R.M.
                 :download:`"Some approaches to best-match file searching."
                 </references/Burkhard-Keller_BK-Trees.pdf>`, CACM, 1973.
    """

    def __init__(
        self,
        distance_function: Callable[[T, T], int],
        labeled_nodes: bool = False,
        garbage_collection_threshold: float = GC_DEFAULT_THRESHOLD,
    ):
        self.root: Union[BKNode, BKNodeLabeled] = None
        self._dist_func: Callable[[T, T], int] = distance_function
        self._deleted_item_count = 0
        self._labeled_nodes = labeled_nodes
        self._length = 0
        if 0 < garbage_collection_threshold < 1.0:
            self._gc_threshold = garbage_collection_threshold
        else:
            self._gc_threshold = GC_DEFAULT_THRESHOLD
        self._gc_min_tree_size = GC_MIN_TREE_SIZE

    def __len__(self):
        return self._length

    def delete_node(self, node: Union["BKNode", "BKNodeLabeled"]):
        if not node._deleted:
            node._deleted = True
            self._deleted_item_count += 1
            self._length -= 1
            if self._length < 0:
                self._length = 0

    def insert(self, key_value: T, key_label: str = None):
        """Insert key value, and in the case of labeled nodes, a unique key label, into the tree.

        Args:
            key_value: The key value used to calculate distances between other keys.
            key_label: Optional; A unique key label for the new node. Defaults to None.
        """
        if key_label is not None:
            key_label = str(key_label)
        self._length += 1

        if self.root is None:
            self.root = self._create_new_node(key_value, key_label)
            return

        current_node = self.root
        while True:
            distance = self._dist_func(current_node.key_value, key_value)

            if distance in current_node:
                current_node = current_node[distance]
            else:
                new_node = self._create_new_node(key_value, key_label)
                current_node[distance] = new_node
                break

    def search(
        self, key_value: int, radius: int, key_label: str = None
    ) -> Union[List["BKNode"], List["BKNodeLabeled"]]:
        if self._length == 0:
            return []
        if key_label is not None:
            key_label = str(key_label)
        remaining: Union[List[BKNode], List[BKNodeLabeled]] = [self.root]
        results: Union[List[BKNode], List[BKNodeLabeled]] = []
        while remaining:
            node = remaining.pop()
            node_qualifies = False
            distance = self._dist_func(node.key_value, key_value)
            if distance <= radius:
                if self._labeled_nodes:
                    labeled_node: BKNodeLabeled = node
                    if key_label is not None and labeled_node.key_label != key_label:
                        node_qualifies = True
                elif not self._labeled_nodes:
                    node_qualifies = True
                if node_qualifies and not node._deleted:
                    results.append(node)

            min_dist = distance - radius
            max_dist = distance + radius
            candidates = [
                node for dist, node in node.children.items() if min_dist <= dist <= max_dist
            ]
            remaining += candidates

        # Is it time to run garbage collector?
        total_length = self._length + self._deleted_item_count
        threshold = max(self._gc_threshold * total_length, self._gc_min_tree_size)
        if self._deleted_item_count > threshold:
            self._collect_garbage()

        return results

    def _collect_garbage(self):
        # Collect nodes that have not been deleted.
        active_nodes: Union[List[BKNode], List[BKNodeLabeled]] = []

        remaining: Union[List[BKNode], List[BKNodeLabeled]] = [self.root]
        while remaining:
            node = remaining.pop()
            if not node._deleted:
                active_nodes.append(node)
            remaining += list(node.children.values())

        # Rebuild the BK Tree from scratch.
        self.root = None
        self._deleted_item_count = 0
        self._length = 0

        while active_nodes:
            node = active_nodes.pop()
            if self._labeled_nodes:
                self.insert(key_value=node.key_value, key_label=node.key_label)
            else:
                self.insert(key_value=node.key_value)

    def _create_new_node(self, key_value: T, key_label: str) -> Union[BKNode, BKNodeLabeled]:
        if self._labeled_nodes:
            return BKNodeLabeled(key_value, key_label)
        else:
            return BKNode(key_value)
