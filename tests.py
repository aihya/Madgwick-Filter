"""Unit tests for the Quaternion and Madgwick classes.

Run from the repository root with either of:

    python -m unittest tests -v
    python tests.py

Floating-point results are compared with a tolerance (TOL), never with ==,
because values like 1/30 can't be represented exactly (see q * q.inverse).
"""

import math
import unittest

try:
    from quaternion import Quaternion  # planned home of the class
except ImportError:
    from madgwick import Quaternion  # where the class lives for now

from madgwick import Madgwick

TOL = 1e-12

# A few quaternions used by several tests: integer ones, a unit one, and a
# real orientation taken from the first row of data/IMU_Data_1(No Mag).csv.
SAMPLES = [
    (1, 2, 3, 4),
    (4, 3, 2, 1),
    (0.5, -0.5, 0.5, -0.5),
    (-2.5, 0.0, 1.25, -3.0),
    (-0.99884, -0.00787354, -0.0298462, 0.0371704),
]


def components(q):
    return (q.w, q.x, q.y, q.z)


def quat(c):
    """Build a Quaternion from a (w, x, y, z) tuple: Quaternion(w, (x, y, z))."""
    return Quaternion(c[0], tuple(c[1:]))


def cross(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def rotate(q, vector):
    """Rotate a 3D vector by a unit quaternion: q ⊗ (0, v) ⊗ q*."""
    return (q * Quaternion(0, tuple(vector)) * q.conjugate).v


def about_axis(axis, angle_deg):
    """Unit quaternion for a rotation of angle_deg around a unit axis."""
    half = math.radians(angle_deg) / 2
    return Quaternion(math.cos(half), tuple(math.sin(half) * a for a in axis))


class QuaternionTestCase(unittest.TestCase):

    def assertQuatAlmostEqual(self, q, expected, tol=TOL):
        """Compare a Quaternion with another one or with a (w, x, y, z) tuple."""
        if isinstance(expected, Quaternion):
            expected = components(expected)
        got = components(q)
        if not all(math.isclose(a, b, abs_tol=tol) for a, b in zip(got, expected)):
            self.fail(f"{got} != {tuple(expected)} (tolerance {tol})")

    def assertVecAlmostEqual(self, v, expected, tol=TOL):
        if not all(math.isclose(a, b, abs_tol=tol) for a, b in zip(v, expected)):
            self.fail(f"{tuple(v)} != {tuple(expected)} (tolerance {tol})")


class TestConstruction(QuaternionTestCase):

    def test_components_are_stored_in_order(self):
        q = Quaternion(1, (2, 3, 4))
        self.assertEqual(components(q), (1.0, 2.0, 3.0, 4.0))

    def test_components_are_converted_to_float(self):
        q = Quaternion(1, (2, 3, 4))
        for value in components(q):
            self.assertIsInstance(value, float)

    def test_default_is_zero_quaternion(self):
        self.assertEqual(components(Quaternion()), (0.0, 0.0, 0.0, 0.0))

    def test_w_and_v_are_given_separately(self):
        q = Quaternion(0.5, (1.5, -2.0, 3.25))
        self.assertEqual(components(q), (0.5, 1.5, -2.0, 3.25))

    def test_v_is_the_vector_part(self):
        self.assertEqual(Quaternion(1, (2, 3, 4)).v, (2.0, 3.0, 4.0))


class TestHamiltonProduct(QuaternionTestCase):

    def setUp(self):
        self.one = Quaternion(1, (0, 0, 0))
        self.i = Quaternion(0, (1, 0, 0))
        self.j = Quaternion(0, (0, 1, 0))
        self.k = Quaternion(0, (0, 0, 1))

    def test_squares_of_basis_elements_are_minus_one(self):
        for name, e in (("i", self.i), ("j", self.j), ("k", self.k)):
            with self.subTest(name):
                self.assertQuatAlmostEqual(e * e, (-1, 0, 0, 0))

    def test_ijk_is_minus_one(self):
        self.assertQuatAlmostEqual(self.i * self.j * self.k, (-1, 0, 0, 0))

    def test_cyclic_products(self):
        self.assertQuatAlmostEqual(self.i * self.j, self.k)
        self.assertQuatAlmostEqual(self.j * self.k, self.i)
        self.assertQuatAlmostEqual(self.k * self.i, self.j)

    def test_reversed_products_flip_sign(self):
        self.assertQuatAlmostEqual(self.j * self.i, (0, 0, 0, -1))
        self.assertQuatAlmostEqual(self.k * self.j, (0, -1, 0, 0))
        self.assertQuatAlmostEqual(self.i * self.k, (0, 0, -1, 0))

    def test_known_product(self):
        # Worked out by hand from the Hamilton product formula.
        p = Quaternion(1, (2, 3, 4)) * Quaternion(5, (6, 7, 8))
        self.assertQuatAlmostEqual(p, (-60, 12, 30, 24))

    def test_identity_is_neutral_on_both_sides(self):
        for s in SAMPLES:
            with self.subTest(q=s):
                q = quat(s)
                self.assertQuatAlmostEqual(q * self.one, s)
                self.assertQuatAlmostEqual(self.one * q, s)

    def test_product_is_not_commutative(self):
        q, p = Quaternion(1, (2, 3, 4)), Quaternion(5, (6, 7, 8))
        self.assertNotEqual(components(q * p), components(p * q))

    def test_product_is_associative(self):
        q, p, r = (quat(s) for s in SAMPLES[:3])
        self.assertQuatAlmostEqual((q * p) * r, q * (p * r))

    def test_pure_quaternions_give_dot_and_cross_product(self):
        # (0, u) ⊗ (0, v) = (−u·v, u×v)
        u, v = (1.0, -2.0, 0.5), (3.0, 0.25, -1.0)
        p = Quaternion(0, tuple(u)) * Quaternion(0, tuple(v))
        self.assertQuatAlmostEqual(p, (-dot(u, v), *cross(u, v)))

    def test_norm_of_product_is_product_of_norms(self):
        q, p = quat(SAMPLES[0]), quat(SAMPLES[3])
        self.assertAlmostEqual((q * p).norm, q.norm * p.norm, places=10)

    def test_operands_are_not_modified(self):
        q, p = Quaternion(1, (2, 3, 4)), Quaternion(5, (6, 7, 8))
        q * p
        self.assertEqual(components(q), (1, 2, 3, 4))
        self.assertEqual(components(p), (5, 6, 7, 8))


class TestScalarOperations(QuaternionTestCase):

    def test_multiply_by_scalar_on_the_right(self):
        self.assertQuatAlmostEqual(Quaternion(1, (2, 3, 4)) * 2, (2, 4, 6, 8))

    def test_multiply_by_scalar_on_the_left(self):
        self.assertQuatAlmostEqual(2 * Quaternion(1, (2, 3, 4)), (2, 4, 6, 8))

    def test_multiply_by_float(self):
        self.assertQuatAlmostEqual(0.5 * Quaternion(1, (2, 3, 4)), (0.5, 1, 1.5, 2))

    def test_scalar_then_quaternion_product(self):
        # The form used by the filter's prediction step: ½ · q ⊗ (0, ω).
        q, omega = Quaternion(1, (0, 0, 0)), Quaternion(0, (2, 4, 6))
        self.assertQuatAlmostEqual(0.5 * q * omega, (0, 1, 2, 3))

    def test_divide_by_scalar(self):
        self.assertQuatAlmostEqual(Quaternion(2, (4, 6, 8)) / 2, (1, 2, 3, 4))

    def test_divide_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            Quaternion(1, (2, 3, 4)) / 0

    def test_scalar_operations_do_not_modify_the_operand(self):
        q = Quaternion(1, (2, 3, 4))
        q * 2
        2 * q
        q / 2
        self.assertEqual(components(q), (1, 2, 3, 4))

    def test_unsupported_types_raise_type_error(self):
        q = Quaternion(1, (2, 3, 4))
        with self.assertRaises(TypeError):
            q * "2"
        with self.assertRaises(TypeError):
            "2" * q
        with self.assertRaises(TypeError):
            q / "2"


class TestAddition(QuaternionTestCase):

    def test_adds_component_by_component(self):
        p = Quaternion(1, (2, 3, 4)) + Quaternion(5, (6, 7, 8))
        self.assertQuatAlmostEqual(p, (6, 8, 10, 12))

    def test_addition_is_commutative(self):
        q, p = Quaternion(1, (2, 3, 4)), Quaternion(-2.5, (0, 1.25, -3))
        self.assertQuatAlmostEqual(q + p, p + q)

    def test_operands_are_not_modified(self):
        q, p = Quaternion(1, (2, 3, 4)), Quaternion(5, (6, 7, 8))
        q + p
        self.assertEqual(components(q), (1, 2, 3, 4))
        self.assertEqual(components(p), (5, 6, 7, 8))

    def test_adding_a_number_raises_type_error(self):
        with self.assertRaises(TypeError):
            Quaternion(1, (2, 3, 4)) + 1


class TestConjugate(QuaternionTestCase):

    def test_negates_the_vector_part(self):
        self.assertQuatAlmostEqual(Quaternion(1, (2, 3, 4)).conjugate, (1, -2, -3, -4))

    def test_conjugate_twice_gives_back_the_quaternion(self):
        for s in SAMPLES:
            with self.subTest(q=s):
                self.assertQuatAlmostEqual(quat(s).conjugate.conjugate, s)

    def test_conjugate_of_product_reverses_the_order(self):
        # (q ⊗ p)* = p* ⊗ q*
        q, p = quat(SAMPLES[0]), quat(SAMPLES[3])
        self.assertQuatAlmostEqual((q * p).conjugate, p.conjugate * q.conjugate)

    def test_product_with_conjugate_is_norm_squared(self):
        for s in SAMPLES:
            with self.subTest(q=s):
                q = quat(s)
                self.assertQuatAlmostEqual(q * q.conjugate, (q.norm_squared, 0, 0, 0))

    def test_does_not_modify_the_quaternion(self):
        q = Quaternion(1, (2, 3, 4))
        q.conjugate
        self.assertEqual(components(q), (1, 2, 3, 4))


class TestNorm(QuaternionTestCase):

    def test_norm_squared(self):
        self.assertEqual(Quaternion(1, (2, 3, 4)).norm_squared, 30)

    def test_norm(self):
        self.assertAlmostEqual(Quaternion(1, (2, 3, 4)).norm, math.sqrt(30), places=12)

    def test_norm_of_negative_components(self):
        self.assertAlmostEqual(Quaternion(-1, (-2, -3, -4)).norm, math.sqrt(30), places=12)

    def test_basis_elements_have_unit_norm(self):
        for e in ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)):
            with self.subTest(e=e):
                self.assertEqual(quat(e).norm, 1.0)

    def test_zero_quaternion_has_zero_norm(self):
        self.assertEqual(Quaternion().norm, 0.0)

    def test_conjugate_has_the_same_norm(self):
        for s in SAMPLES:
            with self.subTest(q=s):
                q = quat(s)
                self.assertAlmostEqual(q.conjugate.norm, q.norm, places=12)


class TestInverse(QuaternionTestCase):

    def test_known_inverse(self):
        self.assertQuatAlmostEqual(Quaternion(1, (2, 3, 4)).inverse,
                                   (1 / 30, -2 / 30, -3 / 30, -4 / 30))

    def test_product_with_inverse_is_identity(self):
        for s in SAMPLES:
            with self.subTest(q=s):
                q = quat(s)
                self.assertQuatAlmostEqual(q * q.inverse, (1, 0, 0, 0))
                self.assertQuatAlmostEqual(q.inverse * q, (1, 0, 0, 0))

    def test_inverse_of_unit_quaternion_is_its_conjugate(self):
        q = Quaternion(0.5, (-0.5, 0.5, -0.5))
        self.assertQuatAlmostEqual(q.inverse, q.conjugate)

    def test_inverse_of_product_reverses_the_order(self):
        # (q ⊗ p)⁻¹ = p⁻¹ ⊗ q⁻¹
        q, p = quat(SAMPLES[0]), quat(SAMPLES[3])
        self.assertQuatAlmostEqual((q * p).inverse, p.inverse * q.inverse)

    def test_zero_quaternion_has_no_inverse(self):
        with self.assertRaises(ZeroDivisionError):
            Quaternion().inverse


class TestRotation(QuaternionTestCase):
    """Rotating vectors with q ⊗ (0, v) ⊗ q*, using only the class's operations."""

    def test_quarter_turn_about_z_maps_x_to_y(self):
        q = about_axis((0, 0, 1), 90)
        self.assertVecAlmostEqual(rotate(q, (1, 0, 0)), (0, 1, 0))

    def test_quarter_turn_about_x_maps_y_to_z(self):
        q = about_axis((1, 0, 0), 90)
        self.assertVecAlmostEqual(rotate(q, (0, 1, 0)), (0, 0, 1))

    def test_vector_on_the_axis_is_unchanged(self):
        q = about_axis((0, 0, 1), 37)
        self.assertVecAlmostEqual(rotate(q, (0, 0, 2.5)), (0, 0, 2.5))

    def test_rotation_preserves_length(self):
        q = about_axis((1 / math.sqrt(3),) * 3, 123)
        v = (0.3, -1.2, 2.0)
        self.assertAlmostEqual(math.dist(rotate(q, v), (0, 0, 0)),
                               math.dist(v, (0, 0, 0)), places=12)

    def test_composing_two_quarter_turns_gives_a_half_turn(self):
        q = about_axis((0, 0, 1), 90)
        self.assertVecAlmostEqual(rotate(q * q, (1, 0, 0)), (-1, 0, 0))

    def test_q_and_minus_q_give_the_same_rotation(self):
        q = about_axis((0, 1, 0), 60)
        v = (1, 2, 3)
        self.assertVecAlmostEqual(rotate(q, v), rotate(-1 * q, v))

    def test_inverse_undoes_the_rotation(self):
        q = about_axis((0, 1 / math.sqrt(2), 1 / math.sqrt(2)), 75)
        v = (0.5, -1.0, 2.0)
        self.assertVecAlmostEqual(rotate(q.inverse, rotate(q, v)), v)


class TestSandwich(QuaternionTestCase):
    """q.sandwich(p) = q ⊗ p ⊗ q*, the product that rotates p by q."""

    def test_quarter_turn_about_z_maps_x_to_y(self):
        q = about_axis((0, 0, 1), 90)
        self.assertQuatAlmostEqual(q.sandwich(Quaternion(0, (1, 0, 0))), (0, 0, 1, 0))

    def test_matches_the_written_out_product(self):
        for s, t in zip(SAMPLES, reversed(SAMPLES)):
            with self.subTest(q=s, p=t):
                q, p = quat(s), quat(t)
                self.assertQuatAlmostEqual(q.sandwich(p), q * p * q.conjugate, tol=1e-9)

    def test_identity_leaves_the_argument_unchanged(self):
        p = Quaternion(0, (0.3, -1.2, 2.0))
        self.assertQuatAlmostEqual(Quaternion(1, (0, 0, 0)).sandwich(p), p)

    def test_operands_are_not_modified(self):
        q, p = about_axis((0, 1, 0), 40), Quaternion(0, (1, 2, 3))
        before = (components(q), components(p))
        q.sandwich(p)
        self.assertEqual((components(q), components(p)), before)

    def test_non_quaternion_argument_raises_type_error(self):
        sandwich = Quaternion(1, (0, 0, 0)).sandwich  # a method, so fetching it must work
        with self.assertRaises(TypeError):
            sandwich((1, 2, 3))


# The tests below cover operations the filter will need but that the class
# doesn't have yet. Each one is skipped until the method exists, then runs
# automatically. They assume the same style as the rest of the class: the
# operation returns a new Quaternion and leaves its operands unchanged.

@unittest.skipUnless(hasattr(Quaternion, "__sub__"), "subtraction not implemented yet")
class TestSubtraction(QuaternionTestCase):

    def test_subtracts_component_by_component(self):
        p = Quaternion(5, (6, 7, 8)) - Quaternion(1, (2, 3, 4))
        self.assertQuatAlmostEqual(p, (4, 4, 4, 4))

    def test_quaternion_minus_itself_is_zero(self):
        q = Quaternion(1, (2, 3, 4))
        self.assertQuatAlmostEqual(q - q, (0, 0, 0, 0))

    def test_operands_are_not_modified(self):
        q, p = Quaternion(5, (6, 7, 8)), Quaternion(1, (2, 3, 4))
        q - p
        self.assertEqual(components(q), (5, 6, 7, 8))
        self.assertEqual(components(p), (1, 2, 3, 4))


@unittest.skipUnless(hasattr(Quaternion, "__neg__"), "negation not implemented yet")
class TestNegation(QuaternionTestCase):

    def test_negates_every_component(self):
        self.assertQuatAlmostEqual(-Quaternion(1, (-2, 3, -4)), (-1, 2, -3, 4))


@unittest.skipUnless(hasattr(Quaternion, "normalize"), "normalize() not implemented yet")
class TestNormalize(QuaternionTestCase):

    def test_result_has_unit_norm(self):
        for s in SAMPLES:
            with self.subTest(q=s):
                self.assertAlmostEqual(quat(s).normalize().norm, 1.0, places=12)

    def test_known_result(self):
        self.assertQuatAlmostEqual(Quaternion(2, (0, 0, 0)).normalize(), (1, 0, 0, 0))
        self.assertQuatAlmostEqual(Quaternion(1, (1, 1, 1)).normalize(), (0.5, 0.5, 0.5, 0.5))

    def test_unit_quaternion_is_unchanged(self):
        self.assertQuatAlmostEqual(Quaternion(0.5, (-0.5, 0.5, -0.5)).normalize(),
                                   (0.5, -0.5, 0.5, -0.5))

    def test_does_not_modify_the_quaternion(self):
        q = Quaternion(1, (2, 3, 4))
        q.normalize()
        self.assertEqual(components(q), (1, 2, 3, 4))

    def test_zero_quaternion_raises(self):
        with self.assertRaises(ZeroDivisionError):
            Quaternion().normalize()


# ---------------------------------------------------------------------------
# Madgwick filter
# ---------------------------------------------------------------------------

GRAVITY = 9.81  # m/s²; the filter only uses the direction of a reading

# Earth's magnetic field where the sensor is used: pointing north (x) and
# down (−z). Made-up values: only the direction matters.
FIELD = (0.6, 0, -0.8)

# Orientations used by several tests: a roll, a pitch, a yaw, and a mix.
ORIENTATIONS = {
    "roll 20°": about_axis((1, 0, 0), 20),
    "pitch -35°": about_axis((0, 1, 0), -35),
    "yaw 30°": about_axis((0, 0, 1), 30),
    "50° about a tilted axis": about_axis((2 / 3, -1 / 3, 2 / 3), 50),
}


def body_frame(q, vector):
    """What a sensor with orientation q reads for an earth-frame direction.

    This is Rᵀ · vector, with R the rotation matrix of q, written out so that
    it doesn't depend on the class's own products. (The filter computes the
    same thing as q* ⊗ (0, v) ⊗ q.)
    """
    w, x, y, z = components(q)
    r = ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
         (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
         (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))
    return tuple(sum(r[row][col] * vector[row] for row in range(3)) for col in range(3))


def filter_at(q):
    """A Madgwick filter whose current estimate is q.

    The class has no public way to set the estimate yet, so this sets the
    attribute behind the read-only quaternion property directly.
    """
    f = Madgwick()
    f._quaternion = q
    return f


class TestAccelerationError(QuaternionTestCase):
    """f_g = q* ⊗ (0, 0, 0, 1) ⊗ q − (0, â): predicted minus measured "up"."""

    def test_missing_reading_returns_none(self):
        self.assertIsNone(Madgwick().acceleration_error(None))

    def test_all_zero_reading_returns_none(self):
        self.assertIsNone(Madgwick().acceleration_error((0, 0, 0)))

    def test_level_and_still_gives_no_error(self):
        error = Madgwick().acceleration_error((0, 0, GRAVITY))
        self.assertQuatAlmostEqual(error, (0, 0, 0, 0))

    def test_known_error_for_a_20_degree_roll(self):
        # The estimate is level, but the sensor is rolled 20° about x.
        s, c = math.sin(math.radians(20)), math.cos(math.radians(20))
        error = Madgwick().acceleration_error((0, GRAVITY * s, GRAVITY * c))
        self.assertQuatAlmostEqual(error, (0, 0, -s, 1 - c))  # (0, 0, −0.342, 0.060)

    def test_size_of_the_reading_does_not_matter(self):
        f = filter_at(ORIENTATIONS["50° about a tilted axis"])
        reading = (0.3, -1.2, 9.5)
        expected = f.acceleration_error(reading)
        for scale in (0.01, 100):
            with self.subTest(scale=scale):
                scaled = tuple(scale * a for a in reading)
                self.assertQuatAlmostEqual(f.acceleration_error(scaled), expected)

    def test_correct_estimate_gives_no_error(self):
        for name, q in ORIENTATIONS.items():
            with self.subTest(name):
                reading = tuple(GRAVITY * a for a in body_frame(q, (0, 0, 1)))
                self.assertQuatAlmostEqual(filter_at(q).acceleration_error(reading),
                                           (0, 0, 0, 0))

    def test_does_not_change_the_estimate(self):
        f = filter_at(about_axis((0, 1, 0), 30))
        before = components(f.quaternion)
        f.acceleration_error((1, 2, 3))
        self.assertEqual(components(f.quaternion), before)


class TestMagError(QuaternionTestCase):
    """f_b = q* ⊗ (0, b) ⊗ q − (0, m̂), with h = q ⊗ (0, m̂) ⊗ q* and
    b = (√(hx² + hy²), 0, hz)."""

    def test_missing_reading_returns_none(self):
        self.assertIsNone(Madgwick().magnetic_error(None))

    def test_all_zero_reading_returns_none(self):
        self.assertIsNone(Madgwick().magnetic_error((0, 0, 0)))

    def test_facing_north_gives_no_error(self):
        self.assertQuatAlmostEqual(Madgwick().magnetic_error(FIELD), (0, 0, 0, 0))

    def test_known_error_for_a_90_degree_heading_mistake(self):
        # The estimate faces north, but the sensor sees north along its y axis.
        # b keeps the reading's dip, so the error has no vertical part.
        error = Madgwick().magnetic_error((0, 0.6, -0.8))
        self.assertQuatAlmostEqual(error, (0, 0.6, -0.6, 0))

    def test_size_of_the_reading_does_not_matter(self):
        f = filter_at(ORIENTATIONS["50° about a tilted axis"])
        reading = (0.2, 0.5, -0.4)
        expected = f.magnetic_error(reading)
        for scale in (0.01, 50):  # gauss or microtesla, it shouldn't matter
            with self.subTest(scale=scale):
                scaled = tuple(scale * m for m in reading)
                self.assertQuatAlmostEqual(f.magnetic_error(scaled), expected)

    def test_correct_estimate_gives_no_error_whatever_the_dip(self):
        # b is rebuilt from every reading, so the local dip angle doesn't matter:
        # flat (magnetic equator), the field above, and almost vertical.
        for field in ((1, 0, 0), FIELD, (0.2, 0, -0.98)):
            for name, q in ORIENTATIONS.items():
                with self.subTest(field=field, orientation=name):
                    reading = body_frame(q, field)
                    self.assertQuatAlmostEqual(filter_at(q).magnetic_error(reading), (0, 0, 0, 0))

    def test_does_not_change_the_estimate(self):
        f = filter_at(about_axis((0, 1, 0), 30))
        before = components(f.quaternion)
        f.magnetic_error((1, 2, 3))
        self.assertEqual(components(f.quaternion), before)


class TestApproximateWithoutAccelerometer(QuaternionTestCase):
    """approximate() when the accelerometer reading is missing: gyroscope only.

    These checks hold whatever Δt, β and gyroscope units end up being.
    """

    def test_no_rotation_keeps_the_estimate(self):
        f = Madgwick()
        f.approximate(G=(0, 0, 0), A=None)
        self.assertQuatAlmostEqual(f.quaternion, (1, 0, 0, 0))

    def test_magnetometer_is_ignored(self):
        # A magnetometer-only correction tilts the estimate, so without the
        # accelerometer there must be no correction at all, even though this
        # magnetometer reading disagrees with the estimate's heading.
        f = Madgwick()
        f.approximate(G=(0, 0, 0), A=None, M=(0, 0.6, -0.8))
        self.assertQuatAlmostEqual(f.quaternion, (1, 0, 0, 0))

    def test_estimate_stays_a_unit_quaternion(self):
        f = Madgwick()
        for _ in range(100):
            f.approximate(G=(0.3, -0.2, 0.5), A=None)
        self.assertAlmostEqual(f.quaternion.norm, 1.0, places=12)

    def test_turning_about_z_only_changes_the_heading(self):
        f = Madgwick()
        f.approximate(G=(0, 0, 1), A=None)
        w, x, y, z = components(f.quaternion)
        self.assertAlmostEqual(x, 0, places=12)
        self.assertAlmostEqual(y, 0, places=12)
        self.assertGreater(z, 0)


if __name__ == "__main__":
    unittest.main()
