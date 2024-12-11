import numpy as np
from .point import Point

class Tetrahedron:
    n_points = 4  # number of points

    def __init__(self, r=np.array([Point.null, Point.ex, Point.ey, Point.ez])):
        self.points = [Point(r_i) for r_i in r]

    def __str__(self):
        s = ["Tetrahedron:"] + ["\n"]
        for i, point in enumerate(self.points):
            s.append(
                "P{} = ({:.2e},{:.2e},{:.2e})".format(i + 1, point.x, point.y, point.z)
            )
            s.append("\n")
        return "".join(s)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Tetrahedron):
            condition = all([other_pi in self.points for other_pi in other.points])
            return condition
        else:
            raise TypeError("Mismatch of argument type in '==' operator.")

    def __ne__(self, other):
        if isinstance(other, Tetrahedron):
            return not self.__eq__(other)
        else:
            raise TypeError("Mismatch of argument type in '!=' operator.")

    def __getitem__(self, subscript):
        return self.points[subscript % Tetrahedron.n_points]

    def __setitem__(self, subscript, item):
        self.points[subscript % Tetrahedron.n_points] = item

    def shift(self, value=0):
        """Shift the current tetrahedron a distance
        and return a new tetrahedron shifted.

        :param value: numeric value to shift, defaults to 0
        :type value: int, float, :class:`Point`, optional

        :return: a new Tetrahedron
        :rtype: :class:`Tetrahedron`
        """
        return Tetrahedron.frompoints([point.shift(value) for point in self.points])

    def rotate(self, angle=0, axis=Point.null):
        """Rotate the current tetrahedron a given angle along
        a given axis, and return a new tetrahedron rotated.

        :param angle: angle to rotate expressed in radians, defaults to 0
        :type angle: int, float, optional
        :param axis: vector of rotation axis, defaults to Point.null
        :type axis: 1-D :class:`numpy.ndarray`, shape=(3, ), optional

        :return: a new Tetrahedron
        :rtype: :class:`Tetrahedron`
        """
        return Tetrahedron.frompoints(
            [point.rotate(angle, axis) for point in self.points]
        )

    # METHODS IN-PLACE
    def ishift(self, value=0) -> None:
        """Shift the current tetrahedron a distance.

        :param value: numeric value to shift, defaults to 0
        :type other: int, float, :class:`Point`, optional

        :return:
        :rtype: None
        """
        for point in self.points:
            point.ishift(value)

    def irotate(self, angle=0, axis=Point.null):
        """Rotate the current tetrahedron a given angle along a given axis.

        :param angle: angle to rotate expressed in radians, defaults to 0
        :type angle: int, float, optional
        :param axis: vector of rotation axis, defaults to Point.null
        :type axis: 1-D :class:`numpy.ndarray`, shape=(3, ), optional

        :return:
        :rtype: None
        """
        for point in self.points:
            point.irotate(angle, axis)

    # CLASS METHODS
    @classmethod
    def frompoints(cls, points=None):
        if points is None:
            r = np.array([Point.null, Point.ex, Point.ey])
        elif isinstance(points, list) and all(
            [isinstance(point, Point) for point in points]
        ):
            r = np.array([point.r for point in points])
        else:
            raise TypeError(
                f"Tetrahedron class, 'frompoints' class method. Variable 'points' must be of type \
                    {list.__name__} of {Point.__name__} instead of {type(points).__name__}."
            )
        return Tetrahedron(r)

    # DECORATORS: @Getters
    @property
    def vol(self):
        # r = (x,y,z)
        drdu = (self.points[0] - self.points[2]).r
        drdv = (self.points[1] - self.points[2]).r
        drdw = (self.points[3] - self.points[2]).r
        j = np.linalg.det(np.array([drdu, drdv, drdw]).T)
        return j / 6
