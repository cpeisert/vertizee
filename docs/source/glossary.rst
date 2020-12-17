===============================
Glossary
===============================

.. glossary::

  arc
    A synonym for :term:`edge`.

  acyclic
    A :term:`graph` with no :term:`cycles <cycle>` is called *acyclic* or *cycle-free*.

  adjacency matrix
    An *adjacency matrix* is a matrix representing a :term:`graph` :math:`G = (V, E)`, where the
    rows and columns are indexed by the :term:`vertices <vertex>`. A one at row :math:`i` and
    column :math:`j` indicates that vertices :math:`i` and :math:`j` are :term:`adjacent`. Formally,
    let the vertices be arbitrarily numbered :math:`1, 2,...,|V|`. Then the adjacent matrix
    :math:`A = (a_{ij})` is a :math:`|V| \\times |V|` matrix such that: :math:`a_{ij} = 1` if
    math:`(i, j) \\in E` and :math:`0` otherwise. :cite:`2009:clrs`

  adjacent
    Two :term:`vertices <vertex>` that are connected by an :term:`edge` are called *adjacent*, and
    a vertex that is an :term:`endpoint` of a :term:`loop` is said to be *adjacent to itself*. Two
    edges incident on the same vertex are also called *adjacent*. :cite:`2010:epp`

  anti-arborescence
    An *anti-arborescence* (also called an *in-tree*) is a :term:`directed rooted tree` in which
    all :term:`edges <edge>` point toward the root. The path from any vertex to the root is unique.
    :cite:`1974:deo`

  arborescence
    An *arborescence* (also called an *out-tree*) is a :term:`directed rooted tree` in which all
    :term:`edges <edge>` point away from the root. Each edge is directed toward a different vertex,
    that is, there is no vertex with more than a single incoming edge. The path from the root to
    any other vertex is unique. :cite:`1967:edmonds` :cite:`1974:deo`
    See :func:`optimum_directed_forest <vertizee.algorithms.trees.spanning.optimum_directed_forest>`

  binary relation
    A *binary relation* :math:`R` from a set :math:`A` to a set :math:`B` is a subset of
    :math:`A \\times B`. Given an ordered pair :math:`(x, y)` in :math:`A \\times B`, :math:`x` is
    related to :math:`y` by :math:`R`, written :math:`x R y`, if and only if, :math:`(x, y)` is in
    :math:`R`. :cite:`2010:epp`

  bipartite graph
    A *bipartite graph* is an :term:`undirected graph` :math:`G(V, E)` in which :math:`V` can be
    partitioned into :math:`V_1` and :math:`V_2` such that :math:`(u, v) \\in E` implies either
    :math:`u \\in V_1` and `v \\in V_2` or :math:`u \\in V_2` and :math:`v \\in V_1`.
    :cite:`2009:clrs`

  branching
    A *branching* is a :term:`forest` of :term:`arborescences <arborescence>`. A branching is also
    called a :term:`directed forest`. :cite:`1967:edmonds`
    See :func:`optimum_directed_forest <vertizee.algorithms.trees.spanning.optimum_directed_forest>`

  circuit
    A synonym for :term:`closed path`.

  clique
    A *clique* is a set of :term:`vertices <vertex>` that are mutually :term:`adjacent`. A clique
    within a graph :math:`G` is a complete :term:`subgraph` of :math:`G`. :cite:`2012:chartrand`

  clique number
    The *clique number* of a graph is the size (that is, the number of :term:`vertices <vertex>`)
    of its largest :term:`clique`. :cite:`2012:chartrand`

  closed path
    A *closed path* is a :term:`path` that begins and ends at the same :term:`vertex`.
    :cite:`2010:epp`

  complete graph
    A *complete graph* is an undirected :term:`simple graph` on :math:`n` :term:`vertices <vertex>`,
    denoted :math:`K_{n}`, whose :term:`edge` set contains exactly one edge for each pair of
    distinct vertices. :cite:`2010:epp`

  complete subgraph
    A synonym for :term:`clique`.

  component
    See :term:`connected component`.

  condensation
    A *condensation* of a :term:`digraph` :math:`G` is an acyclic digraph :math:`G_{c}` where the
    :term:`vertices <vertex>` of :math:`G_{c}` are the :term:`strongly connected` components of
    :math:`G`, and each edge of :math:`G_{c}` is formed by combining the directed edges of
    :math:`G` that connect one strongly connected component to another. Note that the condensation
    of a strongly connected digraph is simply a vertex. :cite:`1974:deo`

  connected
    An :term:`undirected graph` is *connected* if every :term:`vertex` is :term:`reachable` from
    all other vertices.

  connected component
    A *connected component* within an :term:`undirected graph` :math:`G(V, E)` is a
    :term:`connected` :term:`subgraph` :math:`G'(V', E')` such that :math:`\\forall v \\in V'` and
    :math:`\\forall u \\in (V - V')`, there is no :math:`v \\leadsto u` :term:`path`. Another way
    to define connected components is by recognizing that "is :term:`reachable` from" is an
    :term:`equivalence relation`. The connected components of an undirected graph are the
    equivalence classes of vertices under the "is reachable from" relation. :cite:`2009:clrs`

  connected graph
    See :term:`connected`.

  connection
    A *connection* is an :term:`edge` that has exactly one connection between its endpoints. By
    contrast, a :term:`multiconnection` is an edge that may have :term:`parallel edges
    <parallel edge>`. Note that these definitions of *connection* and *multiconnection* are
    specific to the Vertizee library. See :class:`Connection <vertizee.classes.edge.Connection>`.

  contraction
    An *edge contraction* is the operation of removing an edge and merging its endpoints into a new
    vertex. Given a graph :math:`G(V, E)` and an edge :math:`e = (u, v)`, the contraction of
    :math:`e` is written :math:`G/e`. Contracting edge :math:`e` results in a graph
    :math:`G' = (V', E')`, where :math:`V' = (V - {u, v}) \\cup {x}`, where :math:`x` is a new
    vertex. The set of edges :math:`E'` is formed from :math:`E` by deleting edge :math:`(u, v)`
    and, for each vertex :math:`w` adjacent to :math:`u` or :math:`v`, deleting whichever of
    :math:`(u, w)` and :math:`(v, w)` is in :math:`E` and adding the new edge :math:`(x, w)`.
    In effect, :math:`u` and :math:`v` are "contracted" into a single vertex :math:`x`.
    :cite:`2009:clrs`
    See :meth:`EdgeBase.contract <vertizee.classes.edge.EdgeBase.contract>`.

  converse
    A synonym for :term:`reverse`.

  cut
    A *cut* of an :term:`undirected graph` :math:`G = (V, E)`, is a partition of :math:`V` into
    :math:`(S, V - S)`. An edge *crosses* the cut if one endpoint is in :math:`S` and the other
    endpoint is in :math:`V - S`. A cut *respects* a set of edges if none of the edges in the set
    crosses the cut. See also :term:`net flow`. :cite:`2009:clrs`

  cycle
    A *cycle* (also called a *simple circuit*) is a :term:`closed path` that does not have any other
    repeated :term:`vertex` except the first and the last. Thus, a cycle is a path of the form:
    :math:`v_0 e_1 v_1 e_2...v_{n - 1} e_n v_n` where all of the :math:`e_i` are distinct and all
    the :math:`v_j` are distinct except that :math:`v_0 = v_n'. The minimum cycle is a :term:`loop`.
    In an :term:`undirected graph`, and the second smallest cycle is two vertices connected by
    :term:`parallel edges <parallel edge>`. In a :term:`digraph`, the second smallest cycle is two
    vertices connected by two edges facing opposite directions. :cite:`2010:epp`

  dag
    A *dag* is a *directed acyclic graph*.

  degree
    The *degree* of a vertex (also called its :term:`valence`) is the count of its incident
    :term:`edges <edge>`, where self-loops are counted twice. :cite:`2010:epp`
    See :attr:`VertexBase.degree <vertizee.classes.vertex.VertexBase.degree>`.

  dense
    A *dense* :term:`graph` is a graph in which the number of :term:`edges <edge>` is close to the
    maximum possible. For a graph :math:`G(V, E)` with :math:`m = |E|` (the number of edges) and
    :math:`n = |V|` (the number of :term:`vertices <vertex>`), :math:`m = O(n^2)`. If :math:`G`
    is connected, :math:`m = \\Omega (n)`. Graphs with :math:`m = \\Theta (n^2)` are called *dense*
    and graphs with :math:`m = \\Theta (n)` are called *sparse*. :cite:`2013:jungnickel`

  density
    The *density* of a :term:`graph` with :math:`n` vertices is the ratio of its :term:`edge` count
    to the number of edges in a :term:`complete graph` with :math:`n` vertices. See :term:`dense`.

  dictionary
    A *dictionary* is data structure comprised of key-value pairs, where each key appears at most
    once. Dictionaries provide efficient key lookup and are one of the Python standard library
    built-in types.

  diedge
    A *diedge* (pronounced "di-edge") is a synonym for a :term:`directed edge`. Note that *diedge*
    is not a standard graph theory term and is specific to the Vertizee library.
    See :class:`DiEdge <vertizee.classes.edge.DiEdge>` and :class:`MultiDiEdge
    <vertizee.classes.edge.MultiDiEdge>`.

  digraph
    A *digraph* is a :term:`graph` comprised of :term:`directed edges <directed edge>`. See
    :class:`DiGraph <vertizee.classes.graph.DiGraph>` and :class:`MultiDiGraph
    <vertizee.classes.graph.MultiDiGraph>`.

  directed edge
    A *directed edge* is an :term:`edge` defined by an ordered pair of :term:`endpoints <endpoint>`.
    A directed edge :math:`(u, v)` travels from :math:`u` (the :term:`head`) to :math:`v` (the
    :term:`tail`). :cite:`2010:epp` :cite:`2018:roughgarden`
    See :class:`DiEdge <vertizee.classes.edge.DiEdge>` and :class:`MultiDiEdge
    <vertizee.classes.edge.MultiDiEdge>`.

  directed forest
    A *directed forest* (also called a :term:`branching`) is a :term:`forest` of
    :term:`arborescences <arborescence>`. :cite:`1967:edmonds`
    See :func:`optimum_directed_forest <vertizee.algorithms.trees.spanning.optimum_directed_forest>`

  directed graph
    A synonym for :term:`digraph`.

  directed rooted tree
    A *directed rooted tree* is a :term:`rooted tree` that has been assigned an orientation, with
    :term:`edges <edge>` that are either directed *away from* the root (see :term:`arborescence`)
    or *towards* the root (see :term:`anti-arborescence`). :cite:`1967:edmonds` :cite:`1974:deo`
    See :class:`Tree <vertex.classes.data_structures.tree.Tree>`.

  directed spanning forest
    A *directed spanning forest* is a :term:`directed forest` (or :term:`branching`) that contains
    all the :term:`vertices <vertex>` of a :term:`digraph`.
    See :func:`optimum_directed_forest <vertizee.algorithms.trees.spanning.optimum_directed_forest>`

  disconnected
    A graph :math:`G(V, E)` is *disconnected* if and only if its :term:`vertex` set :math:`V` can be
    partitioned into two nonempty, disjoint subsets :math:`V_1` and :math:`V_2` such that there
    exists no :term:`edge` in :math:`G` with one :term:`endpoint` in :math:`V_1` and one endpoint
    in :math:`V_2`. :cite:`1974:deo`

  divertex
    A *divertex* (pronounced "di-vertex") is a :term:`vertex` in a :term:`digraph` that may be
    connected to other vertices via :term:`diedges <diedge>`. Note that *divertex* is not a
    standard graph theory term and is specific to the Vertizee library.
    See :class:`DiVertex <vertizee.classes.vertex.DiVertex>`.

  edge
    An *edge* is a connection between either one or two :term:`vertices <vertex>` called its
    endpoints. An edge with just one endpoint is called a :term:`loop`. Two vertices that are
    connected by an edge are called :term:`adjacent`, and a vertex that is an endpoint of a loop
    is said to be adjacent to itself. :cite:`2010:epp`
    See :class:`DiEdge <vertizee.classes.edge.DiEdge>`,
    class:`Edge <vertizee.classes.edge.Edge>`, :class:`MultiDiEdge
    <vertizee.classes.edge.MultiDiEdge>`, and :class:`MultiEdge <vertizee.classes.edge.MultiEdge>`.

  edge contraction
    See :term:`contraction`.

  empty graph
    A :term:`graph` with no vertices and no edges.

  endpoint
    An *endpoint* is a :term:`vertex` that has one or more :term:`incident edges <incident>`.
    :cite:`2010:epp`

  equivalence relation
    An *equivalence relation* is a :term:`binary relation` that is reflexive, symmetric, and
    transitive. :cite:`2010:epp`

  Fibonacci heap
    A *Fibonacci heap* (also called an *F-heap*) is a data structure that provides
    :term:`priority queue` operations. F-heaps maintain a collection of heap-ordered
    :term:`rooted trees <rooted tree>`. The name comes from the Fibonacci numbers, which are used in
    the F-heap runtime analysis. See also :term:`heap`. :cite:`1987:fredman`

  forest
    A disjoint union of :term:`trees <tree>`. :cite:`1974:deo`

  free tree
    A *free tree* :math:`T` is an undirected graph that is :term:`connected` and :term:`acyclic`. A
    free tree of :math:`n` vertices contains :math:`n - 1` edges. :cite:`1983:tarjan`
    See :class:`Tree <vertex.classes.data_structures.tree.Tree>`.

  graph
    A *graph* :math:`G = (V, E)` consists of a set of :term:`vertices <vertex>` :math:`V` and a set
    of :term:`edges <edge>` :math:`E`, where each edge is associated with either one or two
    vertices called its :term:`endpoints <endpoint>`. An edge with just one endpoint is called a
    :term:`loop`. :cite:`2010:epp`
    See :class:`G <vertizee.classes.graph.G>`, :class:`DiGraph <vertizee.classes.graph.DiGraph>`,
    :class:`Graph <vertizee.classes.graph.Graph>`,
    :class:`MultiGraph <vertizee.classes.graph.MultiGraph>`, and
    :class:`MultiDiGraph <vertizee.classes.graph.MultiDiGraph>`.

  head
    The *head* is the second :term:`vertex` of a :term:`directed edge` :math:`(u, v)` traveling
    from :math:`u` (the :term:`tail`) to :math:`v` (the *head*). In a :term:`queue` data structure,
    the head is the end from which elements are removed. :cite:`2018:roughgarden`

  heap
    A *heap* is a :term:`rooted tree` data structure where each tree node contains one item, with
    the items arranged in *heap order*: if :math:`x` and :math:`p(x)` are a node and its parent,
    then the key of the item in :math:`p(x)` is no greater than the key of the item in :math:`x`.
    Thus the root of the tree contains an item of minimum key and the operation of finding the item
    of minimum key can be carried out in :math:`O(1)` time by accessing the root. Such a heap is
    called a *min heap*. A *max heap* has the opposite heap order where the key of the item in
    :math:`p(x)` is no less than the key of the item in :math:`x`. :cite:`1983:tarjan`
    See :class:`PriorityQueue <vertex.classes.data_structures.priority_queue.PriorityQueue>`.

  in-degree
    The *in-degree* of a :term:`vertex` in a :term:`digraph` is the count of its :term:`incoming
    edges <incoming edge>`. :term:`Self-loops <self-loop>` are counted once (the same as other
    incoming edges). :cite:`1983:tarjan`

  incidence matrix
    An *incidence matrix* is a matrix representing a :term:`graph`, where the rows are indexed by
    the :term:`vertices <vertex>` and the columns are indexed by the :term:`edges <edge>`. A one at
    row :math:`i` and column :math:`j` indicates that edge :math:`j` is :term:`incident` on vertex
    :math:`i`. A zero indicates that they are not incident. :cite:`2009:clrs`

  incident
    :term:`Edges <edge>` that connect to a :term:`vertex` are said to be *incident on* the vertex.
    Two edges that are incident on the same vertex are said to be :term:`adjacent`. :cite:`2010:epp`

  incoming edge
    An *incoming edge* is a :term:`directed edge` pointing to a vertex, that is, where the vertex
    is the :term:`head` of the edge.

  induced subgraph
    An *induced subgraph* is a :term:`subgraph` formed from a subset of the vertices, that includes
    all edges that connect pairs of vertices in the subset. :cite:`clrs`

  in-tree
    A synonym for :term:`anti-arborescence`.

  isolated
    A :term:`vertex` is said to be *isolated* if it has :term:`degree` zero, that is, no
    :term:`incident edges <incident>`. :cite:`1974:deo` See also :term:`semi-isolated`.

  isomorphic
    A graph :math:`G` is isomorphic to graph :math:`G'` if the vertices of :math:`G'` can be
    relabeled to match the vertices of :math:`G`, and if after relabeling, every edge in :math:`G`
    is also in :math:`G'` and vice versa. :cite:`2009:clrs`

  leaf
    A *leaf* vertex (also called an *external node*) is a :term:`tree` vertex with :term:`degree`
    one. :cite:`clrs` See also :term:`pendant`.

  loop
    A *loop* is an :term:`edge` with just one :term:`endpoint`. A loop is also called a *self-loop*,
    since a :term:`vertex` with an :term:`incident` loop has an edge that leaves the vertex and
    loops back around to itself. :cite:`2010:epp`

  multiconnection
    A *multiconnection* is an edge that may have :term:`parallel edges <parallel edge>` between its
    :term:`endpoints <endpoint>`. By contrast, a :term:`connection` is an :term:`edge` that has
    exactly one connection between its endpoints. Hence, a *multiconnection* may be described as a
    collection of parallel *connections*. Note that these definitions of *multiconnection* and
    *connection* are specific to the Vertizee library.
    See :class:`MultiConnection <vertizee.classes.edge.MultiConnection>`.

  multidiedge
    A *multidiedge* (pronounced "multi-di-edge") is a directed :term:`multiconnection`. Note that
    *multidiedge* is not a standard graph theory term and is specific to the Vertizee library.
    See :class:`MultiDiEdge <vertizee.classes.edge.MultiDiEdge>`.

  multidigraph
    A *multidigraph* (pronounced "multi-di-graph") is a directed :term:`multigraph`.
    See :class:`MultiDiGraph <vertizee.classes.graph.MultiDiGraph>`.

  multidivertex
    A *multidivertex* (pronounced "multi-di-vertex") is a :term:`vertex` in a :term:`multidigraph`
    that may be connected to other vertices via :term:`multidiedges <multidiedge>`. Note that
    *multidivertex* is not a standard graph theory term and is specific to the Vertizee library.
    See :class:`MultiDiVertex <vertizee.classes.vertex.MultiDiVertex>`.

  multiedge
    A *multiedge* (pronounced "multi-edge") is an undirected :term:`multiconnection`.
    See :class:`MultiEdge <vertizee.classes.edge.MultiEdge>`.

  multigraph
    A *multigraph* is a :term:`graph` whose edges are :term:`multiconnections <multiconnection>`.
    The :class:`MultiGraph <vertizee.classes.graph.MultiGraph>` class supports undirected
    multigraphs and the :class:`MultiDiGraph <vertizee.classes.graph.MultiDiGraph>` class supports
    directed multigraphs.

  multiplicity
    The *multiplicity* of a :term:`multiconnection` is the count of its parallel
    :term:`connections` (also called :term:`parallel edges <parallel edge>`). The *multiplicity* of
    a :term:`graph` is equal to the largest multiplicity of any of its multiconnections.

  multivertex
    A *multivertex* (pronounced "multi-vertex") is a :term:`vertex` in a :term:`multigraph` that
    may be connected to other vertices via undirected :term:`multiconnections <multiconnection>`.
    Note that *multivertex* is not a standard graph theory term and is specific to the Vertizee
    library. See :class:`MultiVertex <vertizee.classes.vertex.MultiVertex>`.

  neighbor
  neighbour
    A *neighbor* of a :term:`vertex` :math:`u` is any vertex that is :term:`adjacent` to :math:`u`.
    In a :term:`directed graph` :math:`G(V, E)`, :math:`v` is a neighbor of :math:`u` if
    :math:`u \\ne v` and either :math:`(u, v) \\in E` or :math:`(v, u) \\in E`. :cite:`2009:clrs`

  neighborhood
  neighbourhood
    The *neighborhood* of a :term:`vertex` :math:`v` is the :term:`subgraph`
    :term:`induced <induced subgraph>` by its :term:`neighbors <neighbor>`. This is also called the
    *open neighborhood* of :math:`v`. The *closed neighborhood* of :math:`v` includes :math:`v`
    itself in addition to its neighbors.

  net flow
    The *net flow* across a :term:`cut` :math:`(S, T)` with a *flow* :math:`f`, is defined as
    :math:`f(S, T) = \\[ \\sum_{u \\in S} \\sum_{v \\in T} f(u, v) - \\sum_{u \\in S} \\sum_{v \\in T} f(v, u) \\]`
    :cite:`2009:clrs`

  network
    A synonym for :term:`graph`. Sometimes a *network* is defined as a graph where attributes (for
    example, names or labels) are assigned to vertices and/or edges.

  node
    A synonym for :term:`vertex`.

  null graph
    A synonym for :term:`empty graph`.

  optimum spanning arborescence
    An *optimum spanning arborescence* is a :term:`spanning arborescence` that has either maximum
    or minimum total :term:`weight`. For a :term:`digraph` :math:`G(V, E)`, let :math:`c_j` be the
    cost (or :term:`weight`) of :term:`edge` :math:`e_j \\in E`. The maximum spanning arborescence
    can be found as an :term:`optimum spanning branching` where the :term:`edges <edge>` carry new
    weights: :math:`c'_j = c_j + h, h \\gt \\sum |c_j|, e_j \\in G`. Constant :math:`h` is larger
    than the difference in total weights (relative to weights :math:`c_j, e_j \\in G) of any two
    branchings in :math:`G`. A minimum spanning arborescence is the same as a maximum spanning
    arborescence, except that it is relative to weights :math:`c'_j = -c_j`. :cite:`1967:edmonds`
    See :func:`optimum_directed_forest
    <vertizee.algorithms.trees.spanning.optimum_directed_forest>`.

  optimum spanning branching
    An *optimum spanning branching* is equivalent to an *optimum spanning forest*, except that
    the edges are directed, and instead of being comprised of :term:`trees <tree>`, a
    :term:`branching` is comprised of :term:`arborescences <arborescence>`. :cite:`1967:edmonds`

  optimum spanning forest
    An *optimum spanning forest* is a :term:`spanning forest` that has either maximum or minimum
    total :term:`weight`. Every graph :math:`G(V, E)` has a *trivial spanning forest*
    :math:`G'(V', E')` where :math:`V' = V` and :math:`E' = \\emptyset`, since a single vertex
    defines a *trivial tree*. A trivial spanning forest always has weight zero. Hence, a *minimum
    spanning forest* does not contain any positive edges and a *maximum spanning forest* does not
    contain any negative edges. Note that a spanning forest can never have more than :math:`n - 1`
    edges, where :math:`n = |V|`. :cite:`1967:edmonds`
    See :func:`optimum_forest <vertizee.algorithms.trees.spanning.optimum_forest>`.

  oriented graph
    A synonym for :term:`digraph`.

  out-degree
    The *out-degree* of a :term:`vertex` in a :term:`digraph` is the count of its :term:`outgoing
    edges <outgoing edge>`. :term:`Self-loops <self-loop>` are counted once (the same as other
    outgoing edges). :cite:`1983:tarjan`

  out-tree
    A synonym for :term:`arborescence`.

  outgoing edge
    An *outgoing edge* is a :term:`directed edge` pointing away from a vertex, that is, where the
    vertex is the :term:`tail` of the edge.

  parallel edge
    In an :term:`undirected graph`, an :term:`edge` is *parallel* to another edge if the edges are
    :term:`incident` on the same two :term:`endpoints <endpoint>`. In a :term:`digraph`, an edge is
    *parallel* to another :term:`directed edge` if the edges have the same :term:`tail` vertex and
    the same :term:`head` vertex.

  path
    A *path* (sometimes called a *trail*) is a :term:`walk` that does not contain a repeated
    :term:`edge`. A path from a vertex :math:`u` to a vertex :math:`v` is written
    :math:`u \\leadsto v`. See also :term:`simple path`. :cite:`2010:epp`
    See :class:`ShortestPath <vertizee.algorithms.algo_utils.path_utils.ShortestPath>`.

  pendant
    A *pendant* :term:`vertex` is a vertex whose :term:`neighborhood` contains exactly one vertex.
    :cite:`1974:deo` See also :term:`leaf`.

  planar graph
    A *planar graph* is a graph that can be drawn in the plane such that no edges cross each other.
    :cite:`1974:deo`

  postorder
    A *postorder* traversal refers to traversing a :term:`rooted search tree <rooted tree>`
    in the order that each vertex is finished being visited during a search (e.g. breadth-first
    search or depth-first search).

  preorder
    A *preorder* traversal refers to traversing a :term:`rooted search tree <rooted tree>` starting
    with the root vertex and proceeding in the order of vertex discovery.

  priority queue
    A *priority queue* is a data structure similar to a :term:`queue` or :term:`stack` where each
    element has an associated priority. A minimum priority queue always serves the lowest priority
    element first, whereas a maximum priority queue always serves the highest priority element
    first. Priority queues are most often implemented using a :term:`heap` data structure.
    See :class:`PriorityQueue <vertex.classes.data_structures.priority_queue.PriorityQueue>`.

  queue
    A *queue* is a collection that maintains elements in a sequence. The end of the queue to which
    elements are added is called the :term:`tail` and the end from which elements are removed is
    called the :term:`head`. The operation for adding an element is known as *enqueue* and the
    operation for removing an element is know as *dequeue*. A queue is a first-in-first-out (FIFO)
    data structure. :cite:`clrs`

  reachable
    If there is a path :math:`p` from a vertex :math:`u` to a vertex :math:`v`, we say that
    :math:`v` is *reachable* from :math:`u` via :math:`p`. In a :term:`directed graph` this is
    sometimes written :math:`u \\leadsto v`. :cite:`clrs`

  reverse
    The *reverse* of a directed graph is the graph formed by reversing the directions of the edges.
    The *reverse* of a graph is also called the *transpose* or the *converse*.

  rooted tree
    A *rooted tree* is a :term:`free tree` :math:`T` with a distinguished vertex :math:`r`, called
    the *root*. If :math:`u` and :math:`v` are vertices such that :math:`u` is on the path from
    :math:`r` to :math:`v`, :math:`u` is an *ancestor* of :math:`v` and :math:`v` is a *descendent*
    of :math:`u`. If in addition :math:`u \\ne v` and :math:`u` and :math:`v` are :term:`adjacent`,
    then :math:`u` is the *parent* of :math:`v` and :math:`v` is a *child* of :math:`u`. Every
    vertex :math:`u` except the root has a unique parent, generally denoted :math:`p(u)`, and zero
    or more children. The root has no parent and zero or more children. A vertex with no children
    is a :term:`leaf`. :cite:`1983:tarjan`
    See :class:`Tree <vertex.classes.data_structures.tree.Tree>`.

  self-loop
    A synonym for :term:`loop`.

  semi-isolated
    A :term:`vertex` is said to be *semi-isolated* if it has no :term:`incident edges <incident>`
    except for term:`self-loops <self-loop>`. See also :term:`isolated`.

  simple circuit
    A synonym for :term:`cycle`.

  simple graph
    A *simple graph* is a :term:`graph` that does not have loops or parallel edges. A directed
    graph that does not have loops or parallel edges is called a *simple directed graph*.
    :cite:`2010:epp`

  simple path
    A *simple path* is a :term:`path` that does not contain a repeated :term:`vertex`.
    :cite:`2010:epp`

  sink
    A *sink* in a :term:`digraph` is a vertex with no :term:`outgoing edges <outgoing edge>`, that
    is, with :term:`out-degree` zero.

  source
    A *source* in a :term:`digraph` is a vertex with no :term:`incoming edges <incoming edge>`, that
    is, with :term:`in-degree` zero.

  spanning arborescence
    The *spanning arborescence* of a :term:`digraph` :math:`G(V, E)` is an :term:`arborescence`
    that contains :math:`|V| - 1` :term:`edges <edge>`. There are :term:`paths <path>` from the
    arborescence root :term:`vertex` :math:`r` to every other vertex :math:`v \\in V`.
    :cite:`1967:edmonds` See :func:`optimum_directed_forest
    <vertizee.algorithms.trees.spanning.optimum_directed_forest>`

  spanning forest
    A :term:`forest` that contains all the :term:`vertices <vertex>` of a :term:`graph`.
    :cite:`1967:edmonds`
    See :func:`optimum_forest <vertizee.algorithms.trees.spanning.optimum_forest>`

  spanning subgraph
    A *spanning subgraph* of a :term:`graph` :math:`G`, is a :term:`subgraph` that contains all of
    the vertices of :math:`G`. :cite:`1967:edmonds`

  spanning tree
    A :term:`tree` that contains math:`|V| - 1` :term:`edges <edge>` and includes all the
    :term:`vertices <vertex>` of a :term:`graph`. :cite:`2009:clrs`
    See :func:`spanning_tree <vertizee.algorithms.trees.spanning.spanning_tree>`

  sparse
    A *sparse* :term:`graph` is a graph in which the number of :term:`edges <edge>` is small
    relative to the maximum possible. For a graph :math:`G(V, E)` with :math:`m = |E|` (the number
    of edges) and :math:`n = |V|` (the number of :term:`vertices <vertex>`), :math:`m = O(n^2)`. If
    :math:`G` is connected, :math:`m = \\Omega (n)`. Graphs with :math:`m = \\Theta (n)` are called
    *sparse*, and graphs with :math:`m = \\Theta (n^2)` are called *dense*. :cite:`2013:jungnickel`

  stack
    A *stack* is a data structure that implements two main operations named *push*, which adds an
    element, and *pop*, which removes the element that was most recently added. A stack is a
    last-in-first-out (LIFO) data structure. :cite:`2009:clrs`

  strongly connected
    A :term:`digraph` is *strongly connected* if every pair of vertices are :term:`reachable` from
    each other. A :term:`component` in a digraph is strongly connected if every pair of vertices in
    the component are mutually reachable.

  subgraph
    For a given :term:`graph` :math:`G(V, E)`, :math:`G'(V', E')` is a *subgraph* of :math:`G`
    if :math:`V' \\subseteq V` and :math:`E' \\subseteq E`. :cite:`2009:clrs`

  supergraph
    For a given :term:`graph` :math:`G(V, E)`, :math:`G'(V', E')` is a *supergraph* of :math:`G`
    if :math:`V \\subseteq V'` and :math:`E \\subseteq E'`.

  tail
    The *tail* is the first :term:`vertex` of a :term:`directed edge` :math:`(u, v)` traveling
    from :math:`u` (the :term:`tail`) to :math:`v` (the *head*). In a :term:`queue` data structure,
    the tail is the end to which elements are added. :cite:`2018:roughgarden`

  topological ordering
    A *topological ordering* of a :term:`dag` is a linear ordering of its :term:`vertices <vertex>`
    such that for each :term:`edge` :math:`(u, v)`, :math:`u` precedes :math:`v` in the ordering.
    :cite:`2009:clrs`
    See :class:`SearchResults <vertizee.algorithms.algo_utils.search_utils.SearchResults>` and
    :func:`dfs <vertizee.algorithms.search.depth_first_search.dfs>`.

  topological sorting
    *Topological sorting* is the algorithmic process of finding a :term:`topological ordering` of
    a :term:`dag`. :cite:`2009:clrs` See :class:`SearchResults
    <vertizee.algorithms.algo_utils.search_utils.SearchResults>` and :func:`dfs
    <vertizee.algorithms.search.depth_first_search.dfs>`.

  trail
    A synonym for :term:`path`. See also :term:`simple path`.

  transpose
    A synonym for :term:`reverse`.

  tree
    A synonym for :term:`free tree`. See :class:`Tree <vertex.classes.data_structures.tree.Tree>`.

  undirected edge
    An *undirected edge* is an :term:`edge` defined by an unordered pair of
    :term:`vertices <vertex>`. Undirected edges are bidirectional, meaning that an undirected edge
    :math:`(u, v)` is the same edge as :math:`(v, u)`. :cite:`2010:epp` See :class:`Edge
    <vertizee.classes.edge.Edge>` and :class:`MultiEdge <vertizee.classes.edge.MultiDiEdge>`.

  undirected graph
    An *undirected graph* is a :term:`graph` comprised of :term:`undirected edges <directed edge>`.
    See :class:`Graph <vertizee.classes.graph.Graph>` and :class:`MultiGraph
    <vertizee.classes.graph.MultiGraph>`.

  union find
    The *union find* data structure (also known as a *disjoint set data structure*) maintains a
    collection of disjoint sets. The two main operations are 1) merging two sets together (that is,
    creating the *union* of two sets) and 2) *finding* the unique set that contains a given element.
    :cite:`2009:clrs` See :class:`UnionFind <vertex.classes.data_structures.union_find.UnionFind>`.

  valence
    See :term:`degree`.

  valency:
    See :term:`degree`.

  vertex
    A *vertex* is a point in a :term:`graph`. A vertex that is :term:`incident` on an :term:`edge`
    is said to be an :term:`endpoint` of the edge. :cite:`2010:epp`
    See :class:`DiVertex <vertizee.classes.vertex.DiVertex>`,
    :class:`MultiDiVertex <vertizee.classes.vertex.MultiDiVertex>`,
    :class:`MultiVertex <vertizee.classes.vertex.MultiVertex>`, and
    :class:`Vertex <vertizee.classes.vertex.Vertex>`.

  vertex cut
    A *vertex cut* is a set of vertices, that when removed (along with their :term:`incident`
    edges), results in more :term:`connected components <connected component>` than there were
    previously.

  vertex dictionary
    A *vertex dictionary* is a :term:`dictionary` mapping :term:`vertices <vertex>` to values,
    where the vertex keys may be specified as vertex literals (i.e. strings or integers) or as
    vertex objects, and the values may be any arbitrary type. See :class:`VertexDict
    <vertizee.classes.data_structures.vertex_dict.VertexDict>`.

  walk
    A *walk* is an alternating sequence of :term:`vertices <vertex>` and
    :term:`edges <edge>`, beginning and ending with vertices, such that each edge is
    :term:`incident` on the vertices preceding and following it. A *closed walk* is a walk that
    starts and ends at the same vertex. An *open walk* starts and ends at different vertices. A
    walk may have repeated edges and vertices. :cite:`2010:epp` :cite:`1974:deo`

  weakly connected
    A :term:`digraph` is *weakly connected* if replacing all of its
    :term:`directed edges <directed edge>` with :term:`undirected edges <undirected edge>` produces
    a :term:`connected` :term:`undirected graph`. A :term:`component` in a digraph is *weakly
    connected* if replacing its directed edges with undirected edges produces a
    :term:`connected component`.

  weight
    A *weight* is a numerical value assigned to a :term:`vertex` or :term:`edge`. Given a weight
    function :math:`w`, the weight of an :term:`edge-weighted graph <weighted>` :math:`G(V, E)` is
    :math:`\\sum_{e \\in E} w(e)`; and the weight of a :term:`vertex-weighted graph <weighted>` is
    :math:`\\sum_{v \\in V} w(v)`.

  weighted
    A *weighted* :term:`edge` or :term:`vertex` is an edge or vertex that has been assigned a
    :term:`weight`. A *weighted graph* is a :term:`graph` that contains weighted edges and/or
    vertices. An *edge-weighted graph* is a graph with weighted edges and a *vertex-weighted graph*
    is a graph with weighted vertices. All graphs in Vertizee have edges with a default weight of
    one. Graphs in Vertizee in which all edges have the default edge weight of one are classified as
    *unweighted*. If a weight other than the default (one) is assigned to an edge, then the graph
    is classified as *weighted*. See :meth:`G.is_weighted <vertizee.classes.graph.G.is_weighted>`.
