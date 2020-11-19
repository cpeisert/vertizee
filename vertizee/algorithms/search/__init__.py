"""Make certain functions available to the user as direct imports from the
``vertizee.algorithms.search`` namespace.
"""

from vertizee.algorithms.search.breadth_first_search import (
    breadth_first_search,
    bfs_preorder_traversal,
    bfs_labeled_edge_traversal
)

from vertizee.algorithms.search.depth_first_search import (
    depth_first_search,
    dfs_labeled_edge_traversal,
    dfs_preorder_traversal,
    dfs_postorder_traversal
)

from vertizee.algorithms.algo_utils.search_utils import (
    Direction,
    Label,
    SearchResults,
    SearchTree
)
