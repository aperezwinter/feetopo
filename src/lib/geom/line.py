import numpy as np
from .point import Point

class Line:
    n_points = 2  # number of points

    def __init__(self, coords=[Point.null, Point.ex]):
        self.points = [Point(r) for r in coords]

    def __str__(self):
        s = ["Line: "]
        s.append(f"({self.points[0].x:.3g}, {self.points[0].y:.3g}, {self.points[0].z:.3g}) | ")
        s.append(f"({self.points[1].x:.3g}, {self.points[1].y:.3g}, {self.points[1].z:.3g})")
        return "".join(s)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other) -> bool:
        if isinstance(other, Line):
            return all([other_point in self.points for other_point in other.points])
        else:
            raise TypeError(
                f"mismatch type. Other object must be of type \
                    {Line.__name__} instead of {type(other).__name__}."
            )

    def __ne__(self, other) -> bool:
        if isinstance(other, Line):
            return not self.__eq__(other)
        else:
            raise TypeError(
                f"mismatch type. Other object must be of type \
                    {Line.__name__} instead of {type(other).__name__}."
            )

    def __getitem__(self, subscript):
        return self.points[subscript % Line.n_points]

    def __setitem__(self, subscript, item):
        self.points[subscript % Line.n_points] = item

    def vector(self) -> np.ndarray:
        return (self.points[0] - self.points[1]).r

    def tang(self) -> np.ndarray:
        return self.points[0].axis(self.points[1])

    def shift(self, value=0):
        """Shift the current line a distance ``value``
        and return a new line shifted.
        """
        return Line.frompoints([point.shift(value) for point in self.points])

    def rotate(self, angle=0, axis=Point.null):
        """Rotate the current line a given ``angle`` along
        a given ``axis``, and return a new line rotated.
        """
        return Line.frompoints([point.rotate(angle, axis) for point in self.points])

    def ishift(self, value: float = 0) -> None:
        """Shift the current line a distance."""
        for point in self.points:
            point.ishift(value)

    def irotate(self, angle=0, axis=Point.null) -> None:
        """Rotate the current line a given angle along a given axis."""
        for point in self.points:
            point.irotate(angle, axis)

    @classmethod
    def from_arrays(cls, r: np.ndarray):
        points = [Point(ri) for ri in r]
        return Line(r)

    @property
    def vol(self):
        return self.points[0].dist(self.points[1])
