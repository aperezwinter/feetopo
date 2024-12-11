import numpy as np
from .node import Node
from ..geom.point import Point
from ..geom.tetra import Tetrahedron

class Tetra4n(Tetrahedron):
    elemType = 4
    name = "Tetrahedron 4"
    dim = 3
    order = 1
    numNodes = 4
    localNodeCoord = np.array(
        [
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]
    )
    numPrimaryNodes = 4

    def __init__(self, tag: int, nodes: list):
        points = [node.r for node in nodes]
        super(Tetra4n, self).__init__(points)
        self.tag = tag
        self.nodes = nodes
        self.nodeTags = [node.tag for node in nodes]
        self.nodesByTag = {node.tag: node for node in nodes}

    def __str__(self):
        return f"4-node Tetrahedron({self.tag}): Nodes={self.nodeTags}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Tetra4n):
            return (
                all(
                    [
                        any([node_i == node_j for node_j in other._nodes])
                        for node_i in self._nodes
                    ]
                )
                and self._tag == other.tag
            )
        else:
            return False

    def __hash__(self) -> int:
        return tuple([self.tag] + sorted(self.nodeTags))

    def __getitem__(self, subscript):
        return self._nodes[subscript]

    def getCommonNodes(self, other):
        nodes = set(self.nodes)
        commonNodes = nodes.intersection(other.nodes)
        return list(commonNodes)

    def getCommonNodeTags(self, other):
        nodeTags = set(self.nodeTags)
        commonNodeTags = nodeTags.intersection(other.nodeTags)
        return list(commonNodeTags)

    def isCommonNode(self, commonNode: Node, other) -> bool:
        if (commonNode in self.nodes) and (commonNode in other.nodes):
            return True
        else:
            return False

    def compare(self, other) -> int:
        """Perform comparison between two 4-node Tetrahedrons (Tetra4n).
        Compute a special comparison between the current and another Tetrahedron.
        This function return:
        a)  1	if self.tag == other.tag & self.nodes == other.nodes
        b)  0	if self.tag != other.tag & compare(self.nodes, other.nodes) in [1,0]
        c) -1	if self.nodes != other.nodes
        """

        if isinstance(other, Tetra4n):
            if self.__eq__(other):
                return 1
            elif self.tag != other.tag:
                if all(
                    [
                        any(
                            [node_i.compare(node_j) in [1, 0] for node_j in other.nodes]
                        )
                        for node_i in self.nodes
                    ]
                ):
                    return 0
                else:
                    return -1
            else:
                return -1
        else:
            return -1

    @classmethod
    def refVol(cls) -> float:
        """Reference coordinates:
            r1=[0,0,0], r2=[1,0,0]
            r3=[0,1,0], r4=[0,0,1]
        Volume = 1/6
        """
        return 1 / 6