import numpy as np

class Point:
    ex = np.array([1, 0, 0])    # unit vector on x-axis
    ey = np.array([0, 1, 0])    # unit vector on y-axis
    ez = np.array([0, 0, 1])    # unit vector on z-axis
    null = np.array([0, 0, 0])  # null vector

    def __init__(self, r: np.ndarray=np.zeros(3)):
        self.x = r[0]   # coordinate along x-axis
        self.y = r[1]   # coordinate along y-axis
        self.z = r[2]   # coordinate along z-axis
        self.r = r      # Cartesian coordinate r = [x,y,z]

    def __str__(self) -> str:
        return f"Point: ({self.x:.3g}, {self.y:.3g}, {self.z:.3g})"

    def __repr__(self) -> str:
        return self.__str__

    def __add__(self, other):
        if isinstance(other, (int, float)):
            r = self.r + other
        elif isinstance(other, Point):
            r = self.r + other.r
        else:
            raise TypeError(f"No such {type(other).__name__} type in '+' method.")
        return Point(r)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            r = self.r - other
        elif isinstance(other, Point):
            r = self.r - other.r
        else:
            raise TypeError(f"No such {type(other).__name__} type in '-' method.")
        return Point(r)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            r = other * self.r
        elif isinstance(other, Point):
            r = np.multiply(self.r, other.r)
        else:
            raise TypeError(f"No such {type(other).__name__} type in '*' method.")
        return Point(r)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            r = self.r / other
        elif isinstance(other, Point):
            r = [self.x / other.x, self.y / other.y, self.z / other.z]
        else:
            raise TypeError(f"No such {type(other).__name__} type in '/' method.")
        return Point(r)
    
    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            self.x += other
            self.y += other
            self.z += other
            self.r += other
        elif isinstance(other, Point):
            self.x += other.x
            self.y += other.y
            self.z += other.z
            self.r += other.r
        else:
            raise TypeError(f"No such {type(other).__name__} type in '+=' method.")
        return self

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            self.x -= other
            self.y -= other
            self.z -= other
            self.r -= other
        elif isinstance(other, Point):
            self.x -= other.x
            self.y -= other.y
            self.z -= other.z
            self.r -= other.r
        else:
            raise TypeError(f"No such {type(other).__name__} type in '-=' method.")
        return self

    def __imul__(self, other):
        """Overwrite '*=' operator.
        Multiply the current point by another (element wise) or a constant.
        """
        if isinstance(other, (int, float)):
            self.x *= other
            self.y *= other
            self.z *= other
            self.r *= other
        elif isinstance(other, Point):
            self.r = np.multiply(self.r, other.r)
            self.x *= other.x 
            self.y *= other.y
            self.z *= other.z
        else:
            raise TypeError(f"No such {type(other).__name__} type in '*=' method.")
        return self

    def __itruediv__(self, other):
        """Overwrite '/=' operator.
        Compute the division between two points (element wise),
        or divide the current point by a constant.
        """
        if isinstance(other, (int, float)):
            self.x /= other
            self.y /= other
            self.z /= other
            self.r /= other
        elif isinstance(other, Point):
            self.x /= other.x
            self.y /= other.y
            self.z /= other.z
            self.r /= other.r 
        else:
            raise TypeError(f"No such {type(other).__name__} type in '/=' method.")
        return self
    
    def dist(self, other=None) -> float:
        """Compute Euclidean distance between the current an other point.
        Distance = ||p_2-p_1||,  ||p_i|| = sqrt(x_i² + y_i² + z_i²).

        .. note::
                If ``other`` es empty, then compute the distance
                to the origin (Euclidean norm of current point).
        """
        if other is None:
            return np.sqrt(np.sum(self.r**2))
        elif isinstance(other, Point):
            return np.sqrt(np.sum((other.r - self.r) ** 2))
        else:
            raise TypeError(f"No such {type(other).__name__} type in 'dist' method.")

    def axis(self, other=None) -> np.ndarray:
        """Compute the unit vector along the current and other point.
        Distance = (p_2-p_1)/ ||p_2-p_1||,  ||p_i|| = sqrt(x_i² + y_i² + z_i²).

        .. note::
                If ``other`` is the default case, then compute the axis along
                the current point and the origin (unit vector of current point).
        """
        if isinstance(other, Point):
            p = other - self
            p /= p.dist() if p.dist() != 0 else 1
        elif other is None:
            p = self / self.dist() if self.dist() != 0 else self
        else:
            raise TypeError(f"No such {type(other).__name__} type in 'axis' method.")
        return p.r

    def cylindricalCoord(self) -> np.ndarray:
        """Cartesian to cylindrical coordinates.
        (x,y,z) --> (r,phi,z)
        """
        r = np.sqrt(self.x**2 + self.y**2)
        if self.x == 0 and self.y == 0:
            return np.array([0, 0, self.z])
        elif self.x >= 0:
            return np.array([r, np.arcsin(self.y / r), self.z])
        elif self.x > 0:
            return np.array([r, np.arctan(self.y / self.x), self.z])
        else:
            return np.array([r, -np.arcsin(self.y / r) + np.pi, self.z])

    def sphericalCoord(self) -> np.ndarray:
        """Cartesian to spherical coordinates.
        (x,y,z) --> (r,theta,phi)
        """
        if self.x == 0 and self.y == 0 and self.z == 0:
            return np.array([0, 0, 0])
        elif self.z == 0:
            r = self.dist()
            phi = np.arctan(self.y / self.x)
            return np.array([r, np.pi / 2, phi])
        else:
            r = self.dist()
            theta = np.arccos(self.z / r)
            phi = np.arctan(self.y / self.x)
            return np.array([r, theta, phi])

    def inner(self, other=None) -> float:
        """Compute the inner product between the current point and another.
        Product: [u]_i = x_i,  [v]_i = x'_i,  inner(u,v) = x_i.x'_i

        .. note::
                If ``other`` is the default case, then
                compute the inner product with himself.
        """
        if other is None:
            return np.inner(self.r, self.r)
        elif isinstance(other, Point):
            return np.inner(self.r, other.r)
        else:
            raise TypeError(f"No such {type(other).__name__} type in 'inner' method.")

    def cross(self, other=None):
        """Compute the cross product between the current point and another.
        Product: [u x v]_i =  e_ijk u_j v_k, e_ijk: Levi-Civita symbol

        .. note::
                If other point is the default case, then return a null
                point (cross product with himself is zero vector).
        """
        if other is None:
            return Point(Point.null)
        elif isinstance(other, Point):
            return Point(np.cross(self.r, other.r))
        else:
            raise TypeError(f"No such {type(other).__name__} type in 'cross' method.")

    def outer(self, other=None) -> np.ndarray:
        """Compute the outer product between the current point and another.
        Product: [u x v]_ij =  u_i v_j

        .. note::
                If ``other`` is the default case, then
                compute the outer product with himself.
        """
        if other is None:
            return np.outer(self.r, self.r)
        if isinstance(other, Point):
            return np.outer(self.r, other.r)
        else:
            raise TypeError(f"No such {type(other).__name__} type in 'outer' method.")

    def shift(self, value: float = 0):
        """Shift the current point a distance ``value``
        and return a new point shifted.
        """
        if isinstance(value, (int, float, Point)):
            return self + value
        else:
            raise TypeError(f"No such {type(value).__name__} type in 'shift' method.")

    def rotate(self, angle: float = 0, axis: np.ndarray = None):
        """Rotate the current point a given ``angle`` along
        a given ``axis``, and return a new point rotated.
        """
        # Set default value for axis variable.
        axis = Point.null if axis is None else axis

        # Initial reviews and conversions.
        if not isinstance(angle, (int, float)):
            raise TypeError(
                f"Point class, 'rotate' function. Variable 'angle' must be of type \
                    {int.__name__}, {float.__name__} instead of {type(angle).__name__}."
            )
        if isinstance(axis, list):
            axis = np.array(axis)
        elif isinstance(axis, np.ndarray):
            pass
        else:
            raise TypeError(
                f"Point class, 'rotate' function. Variable 'axis' must be of type \
                    {int.__name__}, {float.__name__} instead of {type(axis).__name__}."
            )

        # Check if input 'axis' tensor is of order 1.
        if len(axis.shape) != 1:
            return ValueError(
                f"Point class, 'irotate' function. Order of input \
				'axis' variable is {len(axis.shape)}, higher than 1."
            )

        # [u_x] = (u ^ e_i)*e_i.T, auxiliary calculus for rotation matrix (Einstein sum notation).
        u_x = np.outer(np.cross(axis, Point.ex), Point.ex)
        u_x += np.outer(np.cross(axis, Point.ey), Point.ey)
        u_x += np.outer(np.cross(axis, Point.ez), Point.ez)

        # R(alpha) = cos(alpha)*I + sin(alpha)*[u_x] + (1-cos(alpha))*u*u.T, R:rotation matrix, u:axis.
        R = (
            np.cos(angle) * np.identity(3)
            + (1 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * u_x
        )

        return Point(R.dot(self.r))

    def icross(self, other=None) -> None:
        """Compute the cross product between the current point and
        another. Overwrite the result into the current point.
        Product: [u x v]_i =  e_ijk u_j v_k, e_ijk: Levi-Civita symbol

        .. note::
                If ``other`` is the default case, then overwrite the
                current point with a null point (cross product with
                himself is zero vector).
        """
        if other is None:
            self = Point(Point.null)
        elif isinstance(other, Point):
            self = Point(np.cross(self.r, other.r))
        else:
            raise TypeError(
                f"Point class, 'icross' function. Variable 'other' must be of \
                    type {Point.__name__} instead of {type(other).__name__}."
            )

    def ishift(self, value=0):
        """Shift the current point a distance value and overwrite it."""
        if isinstance(value, (int, float, Point)):
            self += value
        else:
            raise TypeError(
                f"Point class, 'ishift' function. Variable 'value' must be of type \
                    {int.__name__}, {float.__name__} or {Point.__name__} instead of {type(value).__name__}."
            )

    def irotate(self, angle=0, axis: np.ndarray = None):
        """Rotate the current point a given angle
        along a given axis, and overwrite it.
        """

        #   Set default value for axis variable.
        axis = Point.null if axis is None else axis
        #   Initial reviews and conversions.
        if not isinstance(angle, (int, float)):
            raise TypeError(
                f"Point class, 'irotate' function. Variable 'angle' must be of type \
                    {int.__name__}, {float.__name__} instead of {type(angle).__name__}."
            )
        if isinstance(axis, list):
            axis = np.array(axis)
        elif isinstance(axis, np.ndarray):
            pass
        else:
            raise TypeError(
                f"Point class, 'irotate' function. Variable 'axis' must be of type \
                    {np.ndarray.__name__}, {list.__name__} instead of {type(axis).__name__}."
            )
        #   Check if input 'axis' tensor is of order 1.
        if len(axis.shape) != 1:
            raise ValueError(
                f"Point class. Error in 'rotate' function. Order of input \
                    'axis' variable is {len(axis.shape)}, higher than 1."
            )
        #   [u_x] = (u ^ e_i)*e_i.T, auxiliary calculus for rotation matrix (Einstein sum notation).
        u_x = np.outer(np.cross(axis, Point.ex), Point.ex)
        u_x += np.outer(np.cross(axis, Point.ey), Point.ey)
        u_x += np.outer(np.cross(axis, Point.ez), Point.ez)
        #   R(alpha) = cos(alpha)*I + sin(alpha)*[u_x] + (1-cos(alpha))*u*u.T, R:rotation matrix, u:axis.
        R = (
            np.cos(angle) * np.identity(3)
            + (1 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * u_x
        )
        #   Refresh object variables.
        self.r = R.dot(self.r)
        self.x = self.r[0]
        self.y = self.r[1]
        self.z = self.r[2]

    @classmethod
    def from_coord(cls, x: float=0., y: float=0., z: float=0.):
        """Initialize a new Point from its coordinates."""
        condition = all(
            [
                isinstance(x, (int, float)),
                isinstance(y, (int, float)),
                isinstance(z, (int, float)),
            ]
        )
        if not condition:
            raise TypeError("Mismatch type on input variables.")
        return cls(r=np.array([x, y, z]))

    @classmethod
    def rotationMatrix(cls, angle=0, axis: np.ndarray = np.array([0, 0, 0])):
        """Compute the rotation matrix for a given angle
        and along a given axis(direction), and return it.

        :param angle: angle to rotate expressed in radians, defaults to 0
        :type angle: int, float, optional
        :param axis: rotation axis, defaults to Point.null, shape=(3, )
        :type axis: 1-D :class:`numpy.ndarray`, optional

        :raises ValueError: Mismatch of input argument type
        :raises ValueError: Tensor order of axis is higher than 1

        :return: rotation matrix, shape=(3, 3)
        :rtype: 2-D :class:`numpy.ndarray`
        """
        #   Initial reviews and conversions.
        if not isinstance(angle, (int, float)):
            raise TypeError(
                f"Point class, 'rotationMatrix' function. Variable 'angle' must \
				be of type {int}, {float} instead of {type(angle).__name__}."
            )
        if isinstance(axis, list):
            axis = np.array(axis)
        elif isinstance(axis, np.ndarray):
            pass
        else:
            raise TypeError(
                f"Point class, 'rotationMatrix' function. Variable 'axis' must be of type \
                    {np.ndarray.__name__}, {list.__name__} instead of {type(axis).__name__}."
            )

        #   Check if input 'axis' tensor is of order 1.
        if len(axis.shape) != 1:
            return ValueError(
                f"Point class. Error in 'rotationMatrix' function. Order of \
                    input 'axis' variable is {len(axis.shape)}, higher than 1."
            )

        #   Auxiliary calculus for rotation matrix (Einstein sum notation).
        #   [u_x] = (u ^ e_i)*e_i.T
        u_x = np.outer(np.cross(axis, Point.ex), Point.ex)
        u_x += np.outer(np.cross(axis, Point.ey), Point.ey)
        u_x += np.outer(np.cross(axis, Point.ez), Point.ez)

        #   R: rotation matrix, u: rotation axis.
        #   R(alpha) = cos(alpha)*I + sin(alpha)*[u_x] + (1-cos(alpha))*u*u.T
        return (
            np.cos(angle) * np.identity(3)
            + (1 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * u_x
        )

    @property
    def vol(self) -> float:
        return 0
