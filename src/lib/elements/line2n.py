import numpy as np
from .node import Node
from ..geom.point import Point
from ..geom.line import Line

class Line2n(Line):
    elemType = 1
    name = "Line 2"
    dim = 1
    order = 1
    numNodes = 2
    localNodeCoord = np.array([-1., 1.])
    numPrimaryNodes = 2

    def __init__(self, tag: int, nodes: list):
        points = [node.r for node in nodes]
        super(Line2n, self).__init__(points)
        self.tag = tag
        self.nodeTags = [node.tag for node in nodes]
        self.nodes = nodes
        self.nodesByTag = {node.tag: node for node in nodes}
        self.alpha = np.arccos(np.inner(Point.ex, self.tang())) # angle (e_x, t)
        self.u = np.cross(Point.ex, self.tang()) # ortogonal to (e_x, t)
        self.u /= np.linalg.norm(self.u) if np.linalg.norm(self.u) != 0 else 1

    def __str__(self):
        return f"2-node Line({self.tag}): Nodes={self.nodeTags}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Line2n):
            return (self.tag == other.tag) and \
                (sorted(self.nodeTags) == sorted(other.nodeTags))
        else:
            raise TypeError(f"Mismatch type in '==' operator. A comparison \
                            with a {type(other).__name__} is not implemented.")

    def __hash__(self) -> int:
        return tuple([self.tag] + sorted(self.nodeTags))

    def __getitem__(self, subscript):
        return self.nodes[subscript]

    def __setitem__(self, subscript, item):
        self.nodes[subscript] = item
        self.nodeTags = [node.tag for node in self.nodes]
        self.nodesByTag = {node.tag: node for node in self.nodes}

    def getCommonNodes(self, other):
        nodes = set(self.nodes)
        commonNodes = nodes.intersection(other.nodes)
        return list(commonNodes)

    def getCommonNodeTags(self, other):
        nodeTags = set(self.nodeTags)
        commonNodeTags = nodeTags.intersection(other.nodeTags)
        return list(commonNodeTags)

    def isCommonNode(self, node: Node, other) -> bool:
        return (node in self.nodes) and (node in other.nodes)

    def compare(self, other) -> int:
        """Perform comparison between two 2-node Lines (Line2n).
        Compute a special comparison between the current and another line.
        This function return:
        a)  1	if self.tag == other.tag & self.nodes == other.nodes
        b)  0	if self.tag != other.tag & compare(self.nodes, other.nodes) in [1,0]
        c) -1	if self.nodes != other.nodes
        REVISAR EN FUNCIÓN DE LO QUE NECESITO MÁS A FUTURO
        """
        if isinstance(other, Line2n):
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

    def inside(self, coord=Point.null) -> bool:
        """Verify if a given point is inside the 2-node Line or not.
        r  = (x, y, z),     is the given point.
        r1 = (x1, y1, z1),  is the first node of the line.
        r2 = (x2, y2, z2),  is the second node of the line.
        Compute: aux = (r-r1) x (r-r2), with 'x' the cross product.
        if aux = (0,0,0) the point is inside, outside otherwise.

        :param coord: 3-D coordinates of the evaluation point
            (if it's inside the line or not), default to [0,0,0]
        :type coord: 1-D :class:`numpy.ndarray`, :class:`Point`, :class:`Node`

        :raises TypeError: Mismatch of input argument type

        :return: True if the point is inside, False if not
        :rtype: bool
        """

        if isinstance(coord, (Point, Node)):
            r = coord.r
        elif isinstance(coord, np.ndarray):
            r = coord
        else:
            raise TypeError("Mismatch, non correct type in 'coord' argument.")

        r1 = self.nodes[0].r
        r2 = self.nodes[1].r

        return all(np.cross(r - r1, r - r2) == np.zeros(3))

    def minDistance(self, p0=np.array([0, 0, 0])) -> float:
        p1 = self.nodes[0].r
        p2 = self.nodes[1].r
        p12 = p2 - p1
        p10 = p0 - p1
        p20 = p0 - p2

        if np.dot(p12, p20) > 0:
            return np.linalg.norm(p20)
        elif np.dot(p12, p10) < 0:
            return np.linalg.norm(p10)
        else:
            return np.linalg.norm(np.cross(p12, p10)) / np.linalg.norm(p12)

    @classmethod
    def refVol(cls) -> float:
        """Reference coordinates: 
        u = [-1, 1]
        Volume = 2.
        """
        return 2
