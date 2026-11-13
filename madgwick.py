import math

class Quaternion:
    
    def __init__(self, w = 0.0, v:tuple = (0.0, 0.0, 0.0)):
        # 'v' is a read-only property (below), built from x, y, z: don't assign it here
        self.w = float(w)
        self.x, self.y, self.z = float(v[0]), float(v[1]), float(v[2])

    @property
    def v(self):
        return (self.x, self.y, self.z)

    def __repr__(self):
        return f"Quaternion(w={self.w}, v={self.v})"

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(
                self.w * other, 
                (
                    self.x * other, 
                    self.y * other,
                    self.z * other
                )
            )
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.w, self.x, self.y, self.z
            w2, x2, y2, z2 = other.w, other.x, other.y, other.z
            return Quaternion(
                (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2),  # w
                (
                    (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2),  # x
                    (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2),  # y
                    (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2)   # z
                )
            )
        return NotImplemented

    def __rmul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Quaternion(
            other * self.w, 
            (
                other * self.x, 
                other * self.y, 
                other * self.z
            )
        )

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(
            self.w + other.w,
            (
                self.x + other.x,
                self.y + other.y,
                self.z + other.z
            )
        )

    def __neg__(self):
        return Quaternion(
            -self.w,
            (
                -self.x,
                -self.y,
                -self.z
            )
        )

    def __sub__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(
            self.w - other.w,
            (
                self.x - other.x,
                self.y - other.y,
                self.z - other.z
            )
        )

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Quaternion(
            self.w / other,
            (
                self.x / other,
                self.y / other,
                self.z / other
            )
        )

    def conjugate(self):
        return Quaternion(self.w, (-self.x, -self.y, -self.z))

    @property
    def norm_squared(self):
        return (self.w ** 2) + (self.x ** 2) + (self.y ** 2) + (self.z ** 2)

    @property
    def norm(self):
        return math.sqrt(self.norm_squared)

    def inverse(self):
        return self.conjugate() / self.norm_squared

    @classmethod
    def from_axis_angle(cls, axis, angle):
        ax, ay, az = axis
        n = math.sqrt((ax ** 2) + (ay ** 2) + (az ** 2))
        if n == 0:
            raise ValueError("axis must not be the zero vector")

        # Normalize to unit size
        ax, ay, az = ax / n, ay / n, az / n
        cos = math.cos(angle / 2)
        sin = math.sin(angle / 2)
        return cls(
            cos, 
            (
                sin * ax, 
                sin * ay, 
                sin * az
            )
        )

    def normalize(self):
        return self / self.norm

class Madgwick:

    def __init__(self):
        self._quaternion = Quaternion(1, (0, 0, 0))

    @property
    def quaternion(self):
        return self._quaternion

    def acceleration_error(self, sample:tuple):
        if sample is None:
            return

        n = math.sqrt(sum(a ** 2 for a in sample))
        if n == 0:
            return

        a_norm = Quaternion(0, (a / n for a in sample))
        d_norm = Quaternion(0, (0, 0, 1))
        estimat = self.quaternion.conjugate() * d_norm * self.quaternion
        return estimat - a_norm

    def magnetic_error(self, sample:tuple):
        if sample is None:
            return

        n = math.sqrt(sum(m ** 2 for m in sample))
        if n == 0:
            return

        m_norm = Quaternion(0, (m / n for m in sample))
        h = self.quaternion * m_norm * self.quaternion.conjugate()
        b = Quaternion(0, (math.sqrt(h.x ** 2 + h.y ** 2), 0.0, h.z))

        estimat = self.quaternion.conjugate() * b * self.quaternion
        return estimat - m_norm


    def approximate(self, G:tuple = None, A:tuple = None, M:tuple = None):
        # Predict with gyroscope reading #####################
        q_gyro = (1/2) * self.quaternion * Quaternion(0, tuple(G))

        acc_error = self.acceleration_error(A)
        mag_error = self.magnetic_error(M)

        # A missing reading must not stop the update:
        #   acc_error is None -> no correction this sample, use the gyro prediction alone
        #   mag_error is None -> 6-axis mode, correct with gravity only

        # Calculate the gradient of F using the Jacobian 
        # of acceleration error and Jacobian of magnetic error
        # 1. Run Jacobian on acceleration_error
        # 2. Run Jacobian on magnetic_error
        # 3. Gradient of F = Transposed Jacobian of Fg + Transposed Jacobian of Fb

        # Fuse
