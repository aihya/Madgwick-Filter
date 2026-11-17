import math

class Quaternion:
    
    def __init__(self, w = 0.0, v:tuple = (0.0, 0.0, 0.0)):
        # 'v' is a read-only property (below), built from x, y, z: don't assign it here
        self.w = float(w)
        v = tuple(v)  # a generator can only be read once: collect it before checking the length
        if len(v) != 3:
            raise ValueError(f"v must have exactly 3 components, got {len(v)}")
        self.x, self.y, self.z = (float(c) for c in v)

    @property
    def v(self):
        return (self.x, self.y, self.z)

    def __repr__(self):
        return f"Quaternion(w={self.w}, v={self.v})"

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.w * other, (_ * other for _ in self.v))
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.w, *self.v
            w2, x2, y2, z2 = other.w, *other.v
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
        return Quaternion(other * self.w, (other * _ for _ in self.v))

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(
            self.w + other.w, (s+o for s, o in zip(self.v, other.v))
        )

    def __neg__(self):
        return Quaternion(-self.w, (-_ for _ in self.v))

    def __sub__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(
            self.w - other.w, (s-o for s, o in zip(self.v, other.v))
        )

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Quaternion(
            self.w / other, (_ / other for _ in self.v)
        )

    @property
    def conjugate(self):
        return Quaternion(self.w, (-_ for _ in self.v))

    @property
    def norm_squared(self):
        return (self.w ** 2) + sum(_ ** 2 for _ in self.v)

    @property
    def norm(self):
        return math.sqrt(self.norm_squared)

    @property
    def inverse(self):
        return self.conjugate / self.norm_squared

    @classmethod
    def from_axis_angle(cls, axis, angle):
        n = math.sqrt(sum(_ ** 2 for _ in axis))
        if n == 0:
            raise ValueError("axis must not be the zero vector")

        # Normalize to unit size
        axis_norm = (_ / n for _ in axis)
        cos = math.cos(angle / 2)
        sin = math.sin(angle / 2)
        return cls(cos, (sin * _ for _ in axis_norm))

    def normalize(self):
        return self / self.norm

    def sandwich(self, other:Quaternion):
        if not isinstance(other, Quaternion):
            raise TypeError(
                f"sandwich expects a Quaternion, got {type(other).__name__}"
            )
        return self * other * self.conjugate

class Madgwick:

    def __init__(self):
        self._quaternion = Quaternion(1, (0, 0, 0))

    @property
    def quaternion(self):
        return self._quaternion

    @quaternion.setter
    def quaternion(self, value:Quaternion):
        if not isinstance(value, Quaternion):
            raise TypeError(
                f"quaternion must be a Quaternion, got {type(value).__name__}"
            )
        # Replace the object (never edit the old one in place) and keep it a unit quaternion
        self._quaternion = value.normalize()

    def acceleration_error(self, sample:tuple):
        if sample is None:
            return

        n = math.sqrt(sum(_ ** 2 for _ in sample))
        if n == 0:
            return

        a_norm = Quaternion(0, (_ / n for _ in sample))
        d_norm = Quaternion(0, (0, 0, 1))
        estimate = self.quaternion.conjugate * d_norm * self.quaternion

        return estimate - a_norm

    def magnetic_error(self, sample:tuple):
        if sample is None:
            return

        n = math.sqrt(sum(_ ** 2 for _ in sample))
        if n == 0:
            return

        m_norm = Quaternion(0, (_ / n for _ in sample))
        h = self.quaternion * m_norm * self.quaternion.conjugate
        b = Quaternion(0, (math.sqrt(h.x ** 2 + h.y ** 2), 0.0, h.z))

        estimate = self.quaternion.conjugate * b * self.quaternion
        return estimate - m_norm

    def approximate(self, G:tuple = None, A:tuple = None, M:tuple = None, dt:float = 0.01):
        # dt: time between samples in seconds. 0.01 (100 Hz) is a placeholder
        # until load_data.py works out the real sampling rate.

        # Predict with gyroscope reading #####################
        # q_gyro is the rate of change of the estimate (q̇), not a rotation:
        # it is integrated over dt below, never normalized on its own
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
        if acc_error is None:
            # Integrate: q <- q + q̇·dt (the setter normalizes the result)
            self.quaternion = self.quaternion + q_gyro * dt
        elif mag_error is None:
            self.quaternion = self.quaternion