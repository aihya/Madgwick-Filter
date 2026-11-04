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
            self.w *= other
            self.x *= other
            self.y *= other
            self.z *= other
        elif isinstance(other, Quaternion):
            w = (self.w * other.w) - (self.x * other.x) - (self.y * other.y) - (self.z * other.z)
            x = (self.w * other.x) + (self.x * other.w) + (self.y * other.z) - (self.z * other.y)
            y = (self.w * other.y) + (self.x * other.z) + (self.y * other.w) - (self.z * other.x)
            z = (self.w * other.z) + (self.x * other.y) + (self.y * other.x) - (self.z * other.w)
            self.w, self.x, self.y, self.z = w, x, y, z
        else:
            raise ValueError("Right-hand side must be either an integer, float or an instance of Quaternion")

    def __add__(self, other):
        if not isinstance(other, Quaternion):
            raise ValueError("Right-hand side must be either an integer, float or an instance of Quaternion")
        self.w += other.w
        self.x += other.x
        self.y += other.y
        self.z += other.z

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    @property
    def norm(self):
        return math.sqrt(self.w ** 2, self.x ** 2, self.y ** 2, self.z ** 2)

    def inverse(self):
        return self.conjugate() / (self.norm ** 2)

class Madgwick:

    def __init__(self):
        self._quaternion = Quaternion()

    @property
    def quaternion(self):
        return self._quaternion

    def approximate(self, G, A, M=None):
        # Predict with gyroscope reading
        qw = 0.5 * (self.quaternion * Quaternion(0, *G.v))

        # Correction with gravity

        # Fuse phase