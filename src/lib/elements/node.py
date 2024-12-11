import numpy as np
from ..geom.point import Point

class Node(Point):
    elemType = 15
    name = "Point"
    dim = 0
    order = 0
    numNodes = 1
    localNodeCoord = np.array([0.])
    numPrimaryNodes = 1

    def __init__(self, tag: int, r: np.ndarray=Point.null) -> None:
        super(Node, self).__init__(r)
        self.tag = tag

    def __str__(self) -> str:
        s = f"1-node Point({self.tag}):\n" + super().__str__()
        return s

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other) -> bool:
        """Overwrite '==' operator.
        Compute equal comparison between the current and another node.
        Nodes are equal if their tags are equal, because nodes are unique 
        based on their tags.
        """
        if isinstance(other, Node):
            return self.tag == other.tag
        else:
            raise TypeError(f"Mismatch type in '==' operator. A comparison \
                            with a {type(other).__name__} is not implemented.")

    def __hash__(self) -> int:
        return self.tag

    def compare(self, other) -> int:
        """Compute a special comparison between the current and
        another node. This function return:
        a)  1	if self.tag == other.tag & self.r == other.r
        b)  0	if self.tag != other.tag & self.r == other.r
        c) -1	if self.r != other.r
        """
        if isinstance(other, Node):
            if (self.tag == other.tag) and (np.allclose(self.r, other.r)):
                return 1
            elif (self.tag != other.tag) and (np.allclose(self.r, other.r)):
                return 0
            else:
                return -1
        else:
            raise TypeError(f"Mismatch type. A comparison with a {type(other).__name__} is not implemented.")

    @classmethod
    def refVol(cls) -> float:
        """An 1-node reference point has zero volume."""
        return 0

    @property
    def vol(self) -> float:
        return 0
