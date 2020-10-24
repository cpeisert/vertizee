"""Make certain functions available to the user as direct imports from the ``vertizee.typing``
namespace."""

from vertizee.classes.edge import (
    DiEdge,
    Edge,
    EdgeClass,
    EdgeTuple,
    EdgeTupleAttr,
    EdgeTupleWeighted,
    EdgeTupleWeightedAttr,
    EdgeType,
    MultiDiEdge,
    MultiEdge
)
from vertizee.classes.graph import GraphType
from vertizee.classes.primitives_parsing import GraphPrimitive
from vertizee.classes.vertex import (
    Vertex,
    VertexClass,
    VertexLabel,
    VertexTupleAttr,
    VertexType
)
