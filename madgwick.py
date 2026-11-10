import math

class Quaternion:
    
    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self.w, self.x, self.y, self.z = float(w), float(x), float(y), float(z)

    @property
    def v(self):
        return (self.x, self.y, self.z)

    def __repr__(self):
        return f"Quaternion(w={self.w}, v={self.v})"

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(
                self.w * other, 
                self.x * other, 
                self.y * other,
                self.z * other
            )
        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.w, self.x, self.y, self.z
            w2, x2, y2, z2 = other.w, other.x, other.y, other.z
            return Quaternion(
                (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2),  # w
                (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2),  # x
                (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2),  # y
                (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2)   # z
            )
        return NotImplemented

    def __rmul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Quaternion(
            other * self.w, 
            other * self.x, 
            other * self.y, 
            other * self.z
        )

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Quaternion(
            self.w / other,
            self.x / other,
            self.y / other,
            self.z / other
        )

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

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
            sin * ax, 
            sin * ay, 
            sin * az
        )

    def normalize(self):
        return self / self.norm

class Madgwick:

    def __init__(self):
        self._quaternion = Quaternion()

    @property
    def quaternion(self):
        return self._quaternion

    def approximate(self, G: tuple, A: tuple, M=None):
        # Predict with gyroscope reading
        q_gyro = (1/2) * self.quaternion * Quaternion(0, *G.v)

        # Correction with gravity
        

        # Fuse phase
