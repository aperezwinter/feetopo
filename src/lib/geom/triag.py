import numpy as np
from .point import Point

class Triangle:
    n_points = 3  # number of points

    def __init__(self, coords: list=[Point.null, Point.ex, Point.ey]):
        self.points = [Point(r) for r in coords]

    def __str__(self):
        s = ["Triangle: "]
        for i in range(Triangle.n_points-1):
            s.append(f"({self.points[i].x:.3g}, {self.points[i].y:.3g}, {self.points[i].z:.3g}) | ")
        s.append(f"({self.points[-1].x:.3g}, {self.points[-1].y:.3g}, {self.points[-1].z:.3g})")
        return "".join(s)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Triangle):
            return all([point in self.points for point in other.points])
        else:
            raise TypeError("Mismatch of argument type in '==' operator.")

    def __ne__(self, other):
        if isinstance(other, Triangle):
            return not self.__eq__(other)
        else:
            raise TypeError("Mismatch of argument type in '!=' operator.")

    def __getitem__(self, subscript):
        return self.points[subscript % Triangle.n_points]

    def __setitem__(self, subscript, item):
        self.points[subscript % Triangle.n_points] = item

    def normal(self):
        p1, p2, p3 = self.points[0], self.points[1], self.points[2]
        return ((p2 - p1).cross(p3 - p1)).axis()

    def shift(self, value=0):
        """Shift the current triangle a distance
        and return a new triangle shifted.

        :param value: numeric value to shift, defaults to 0
        :type value: int, float, :class:`Point`, optional

        :return: a new Triangle
        :rtype: :class:`Triangle`
        """
        points = [point.shift(value) for point in self.points]
        return Triangle(points)

    def rotate(self, angle=0, axis=Point.null):
        """Rotate the current triangle a given angle along
        a given axis, and return a new triangle rotated.

        :param angle: angle to rotate expressed in radians, defaults to 0
        :type angle: int, float, optional
        :param axis: vector of rotation axis, defaults to Point.null
        :type axis: 1-D :class:`numpy.ndarray`, shape=(3, ), optional

        :return: a new Triangle
        :rtype: :class:`Triangle`
        """
        points = [point.rotate(angle, axis) for point in self.points]
        return Triangle(points)

    def ishift(self, value=0) -> None:
        """Shift the current triangle a distance.

        :param value: numeric value to shift, defaults to 0
        :type other: int, float, :class:`Point`, optional

        :return:
        :rtype: None
        """
        for point in self.points:
            point.ishift(value)

    def irotate(self, angle=0, axis=Point.null):
        """Rotate the current triangle a given angle along a given axis.

        :param angle: angle to rotate expressed in radians, defaults to 0
        :type angle: int, float, optional
        :param axis: vector of rotation axis, defaults to Point.null
        :type axis: 1-D :class:`numpy.ndarray`, shape=(3, ), optional

        :return:
        :rtype: None
        """
        for point in self.points:
            point.irotate(angle, axis)

    @classmethod
    def from_arrays(cls, r: np.ndarray):
        points = [Point(ri) for ri in r]
        return Triangle(r)

    @property
    def vol(self):
        a = (self.points[1] - self.points[0]).r
        b = (self.points[2] - self.points[0]).r
        return np.linalg.norm(np.cross(a, b)) / 2
