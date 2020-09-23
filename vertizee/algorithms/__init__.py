"""Make certain functions available to the user as direct imports from the ``vertizee.algorithms``
namespace."""

from vertizee.algorithms.components.strongly_connected import kosaraju_strongly_connected_components
from vertizee.algorithms.search.depth_first_search import (
    depth_first_search,
    dfs_preorder_traversal,
    dfs_postorder_traversal,
    dfs_labeled_edge_traversal,
)
from vertizee.algorithms.shortest_paths.unweighted import breadth_first_search_shortest_paths
from vertizee.algorithms.shortest_paths.weighted import (
    shortest_paths_bellman_ford,
    shortest_paths_dijkstra,
)
from vertizee.algorithms.tree.spanning import (
    spanning_tree_kruskal,
    spanning_tree_prim,
    spanning_tree_prim_fibonacci,
)
