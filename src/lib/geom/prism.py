import numpy as np
from .point import Point

class Prism:
    n_points = 6  # number of points

    def __init__(
        self,
        coords=
            [
                Point.null,
                Point.ex,
                Point.ey,
                Point.ez,
                Point.ex + Point.ez,
                Point.ey + Point.ez,
            ],
    ):
        self.points = [Point(r) for r in coords]

    def __str__(self):
        s = ["Prism: "]
        for i in range(Prism.n_points-1):
            s.append(f"({self.points[i].x:.3g}, {self.points[i].y:.3g}, {self.points[i].z:.3g}) | ")
        s.append(f"({self.points[-1].x:.3g}, {self.points[-1].y:.3g}, {self.points[-1].z:.3g})")
        return "".join(s)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Prism):
            return all([point in self.points for point in other.points])
        else:
            raise TypeError("Mismatch of argument type in '==' operator.")

    def __ne__(self, other):
        if isinstance(other, Prism):
            return not self.__eq__(other)
        else:
            raise TypeError("Mismatch of argument type in '!=' operator.")

    def __getitem__(self, subscript):
        return self.points[subscript % Prism.n_points]

    def __setitem__(self, subscript, item):
        self.points[subscript % Prism.n_points] = item

    def shift(self, value=0):
        """Shift the current prism a distance
        and return a new prism shifted.

        :param value: numeric value to shift, defaults to 0
        :type value: int, float, :class:`Point`, optional

        :return: a new Prism
        :rtype: :class:`Prism`
        """
        points = [point.shift(value) for point in self.points]
        return Prism(points)

    def rotate(self, angle=0, axis=Point.null):
        """Rotate the current prism a given angle along
        a given axis, and return a new prism rotated.

        :param angle: angle to rotate expressed in radians, defaults to 0
        :type angle: int, float, optional
        :param axis: vector of rotation axis, defaults to Point.null
        :type axis: 1-D :class:`numpy.ndarray`, shape=(3, ), optional

        :return: a new Prism
        :rtype: :class:`Prism`
        """
        points = [point.rotate(angle, axis) for point in self.points]
        return Prism(points)

    def ishift(self, value=0) -> None:
        """Shift the current prism a distance.

        :param value: numeric value to shift, defaults to 0
        :type other: int, float, :class:`Point`, optional

        :return:
        :rtype: None
        """
        for point in self.points:
            point.ishift(value)

    def irotate(self, angle=0, axis=Point.null):
        """Rotate the current prism a given angle along a given axis.

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
        return Prism(r)

    @property
    def vol(self):
        a = (self.points[1] - self.points[0]).r
        b = (self.points[2] - self.points[0]).r
        area = np.linalg.norm(np.cross(a, b)) / 2
        height = abs((self.points[3] - self.points[0]).z)
        return area * height
