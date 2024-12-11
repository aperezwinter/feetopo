import numpy as np
from ..geom.quad import Quadrilateral

class Quad4n(Quadrilateral):
    elemType = 3
    name = "Quadrilateral 4"
    dim = 2
    order = 1
    numNodes = 4
    localNodeCoord = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    numPrimaryNodes = 4

    def __init__(self, tag: int, nodes: list):
        points = [node.r for node in nodes]
        super(Quad4n, self).__init__(points)
        self.tag = tag
        self.nodes = nodes
        self.nodeTags = [node.tag for node in nodes]
        self.nodesByTag = {node.tag: node for node in nodes}

    def __str__(self):
        return f"4-node Quadrilateral({self.tag}): Nodes={self.nodeTags}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Quad4n):
            return (
                all(
                    [
                        any([node_i == node_j for node_j in other.nodes])
                        for node_i in self.nodes
                    ]
                )
                and self.tag == other.tag
            )
        else:
            return False

    def __hash__(self) -> int:
        return tuple([self.tag] + sorted(self.nodeTags))

    def __getitem__(self, subscript):
        return self.nodes[subscript]

    def getCommonNodes(self, other):
        nodes = set(self.nodes)
        commonNodes = nodes.intersection(other.nodes)
        return list(commonNodes)

    def getCommonNodeTags(self, other):
        nodeTags = set(self.nodeTags)
        commonNodeTags = nodeTags.intersection(other.nodeTags)
        return list(commonNodeTags)

    def compare(self, other) -> int:
        if isinstance(other, Quad4n):
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
    def materialArea(cls) -> float:
        """Reference coordinates:
        (u,v) = [ [0,0], [1,0], [0,1] ]
        Area = 1/2
        """
        return 0.5
