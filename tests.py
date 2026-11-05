"""Unit tests for the Quaternion class.

Run from the repository root with either of:

    python -m unittest tests -v
    python tests.py

Floating-point results are compared with a tolerance (TOL), never with ==,
because values like 1/30 can't be represented exactly (see q * q.inverse()).
"""

import math
import unittest

try:
    from quaternion import Quaternion  # planned home of the class
except ImportError:
    from madgwick import Quaternion  # where the class lives for now

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


def cross(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def rotate(q, vector):
    """Rotate a 3D vector by a unit quaternion: q ⊗ (0, v) ⊗ q*."""
    return (q * Quaternion(0, *vector) * q.conjugate()).v


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
        q = Quaternion(1, 2, 3, 4)
        self.assertEqual(components(q), (1.0, 2.0, 3.0, 4.0))

    def test_components_are_converted_to_float(self):
        q = Quaternion(1, 2, 3, 4)
        for value in components(q):
            self.assertIsInstance(value, float)

    def test_default_is_zero_quaternion(self):
        self.assertEqual(components(Quaternion()), (0.0, 0.0, 0.0, 0.0))

    def test_v_is_the_vector_part(self):
        self.assertEqual(Quaternion(1, 2, 3, 4).v, (2.0, 3.0, 4.0))


class TestHamiltonProduct(QuaternionTestCase):

    def setUp(self):
        self.one = Quaternion(1, 0, 0, 0)
        self.i = Quaternion(0, 1, 0, 0)
        self.j = Quaternion(0, 0, 1, 0)
        self.k = Quaternion(0, 0, 0, 1)

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
        p = Quaternion(1, 2, 3, 4) * Quaternion(5, 6, 7, 8)
        self.assertQuatAlmostEqual(p, (-60, 12, 30, 24))

    def test_identity_is_neutral_on_both_sides(self):
        for s in SAMPLES:
            with self.subTest(q=s):
                q = Quaternion(*s)
                self.assertQuatAlmostEqual(q * self.one, s)
                self.assertQuatAlmostEqual(self.one * q, s)

    def test_product_is_not_commutative(self):
        q, p = Quaternion(1, 2, 3, 4), Quaternion(5, 6, 7, 8)
        self.assertNotEqual(components(q * p), components(p * q))

    def test_product_is_associative(self):
        q, p, r = (Quaternion(*s) for s in SAMPLES[:3])
        self.assertQuatAlmostEqual((q * p) * r, q * (p * r))

    def test_pure_quaternions_give_dot_and_cross_product(self):
        # (0, u) ⊗ (0, v) = (−u·v, u×v)
        u, v = (1.0, -2.0, 0.5), (3.0, 0.25, -1.0)
        p = Quaternion(0, *u) * Quaternion(0, *v)
        self.assertQuatAlmostEqual(p, (-dot(u, v), *cross(u, v)))

    def test_norm_of_product_is_product_of_norms(self):
        q, p = Quaternion(*SAMPLES[0]), Quaternion(*SAMPLES[3])
        self.assertAlmostEqual((q * p).norm, q.norm * p.norm, places=10)

    def test_operands_are_not_modified(self):
        q, p = Quaternion(1, 2, 3, 4), Quaternion(5, 6, 7, 8)
        q * p
        self.assertEqual(components(q), (1, 2, 3, 4))
        self.assertEqual(components(p), (5, 6, 7, 8))


class TestScalarOperations(QuaternionTestCase):

    def test_multiply_by_scalar_on_the_right(self):
        self.assertQuatAlmostEqual(Quaternion(1, 2, 3, 4) * 2, (2, 4, 6, 8))

    def test_multiply_by_scalar_on_the_left(self):
        self.assertQuatAlmostEqual(2 * Quaternion(1, 2, 3, 4), (2, 4, 6, 8))

    def test_multiply_by_float(self):
        self.assertQuatAlmostEqual(0.5 * Quaternion(1, 2, 3, 4), (0.5, 1, 1.5, 2))

    def test_scalar_then_quaternion_product(self):
        # The form used by the filter's prediction step: ½ · q ⊗ (0, ω).
        q, omega = Quaternion(1, 0, 0, 0), Quaternion(0, 2, 4, 6)
        self.assertQuatAlmostEqual(0.5 * q * omega, (0, 1, 2, 3))

    def test_divide_by_scalar(self):
        self.assertQuatAlmostEqual(Quaternion(2, 4, 6, 8) / 2, (1, 2, 3, 4))

    def test_divide_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            Quaternion(1, 2, 3, 4) / 0

    def test_scalar_operations_do_not_modify_the_operand(self):
        q = Quaternion(1, 2, 3, 4)
        q * 2
        2 * q
        q / 2
        self.assertEqual(components(q), (1, 2, 3, 4))

    def test_unsupported_types_raise_type_error(self):
        q = Quaternion(1, 2, 3, 4)
        with self.assertRaises(TypeError):
            q * "2"
        with self.assertRaises(TypeError):
            "2" * q
        with self.assertRaises(TypeError):
            q / "2"


class TestAddition(QuaternionTestCase):

    def test_adds_component_by_component(self):
        p = Quaternion(1, 2, 3, 4) + Quaternion(5, 6, 7, 8)
        self.assertQuatAlmostEqual(p, (6, 8, 10, 12))

    def test_addition_is_commutative(self):
        q, p = Quaternion(1, 2, 3, 4), Quaternion(-2.5, 0, 1.25, -3)
        self.assertQuatAlmostEqual(q + p, p + q)

    def test_operands_are_not_modified(self):
        q, p = Quaternion(1, 2, 3, 4), Quaternion(5, 6, 7, 8)
        q + p
        self.assertEqual(components(q), (1, 2, 3, 4))
        self.assertEqual(components(p), (5, 6, 7, 8))

    def test_adding_a_number_raises_type_error(self):
        with self.assertRaises(TypeError):
            Quaternion(1, 2, 3, 4) + 1


class TestConjugate(QuaternionTestCase):

    def test_negates_the_vector_part(self):
        self.assertQuatAlmostEqual(Quaternion(1, 2, 3, 4).conjugate(), (1, -2, -3, -4))

    def test_conjugate_twice_gives_back_the_quaternion(self):
        for s in SAMPLES:
            with self.subTest(q=s):
                self.assertQuatAlmostEqual(Quaternion(*s).conjugate().conjugate(), s)

    def test_conjugate_of_product_reverses_the_order(self):
        # (q ⊗ p)* = p* ⊗ q*
        q, p = Quaternion(*SAMPLES[0]), Quaternion(*SAMPLES[3])
        self.assertQuatAlmostEqual((q * p).conjugate(), p.conjugate() * q.conjugate())

    def test_product_with_conjugate_is_norm_squared(self):
        for s in SAMPLES:
            with self.subTest(q=s):
                q = Quaternion(*s)
                self.assertQuatAlmostEqual(q * q.conjugate(), (q.norm_squared, 0, 0, 0))

    def test_does_not_modify_the_quaternion(self):
        q = Quaternion(1, 2, 3, 4)
        q.conjugate()
        self.assertEqual(components(q), (1, 2, 3, 4))


class TestNorm(QuaternionTestCase):

    def test_norm_squared(self):
        self.assertEqual(Quaternion(1, 2, 3, 4).norm_squared, 30)

    def test_norm(self):
        self.assertAlmostEqual(Quaternion(1, 2, 3, 4).norm, math.sqrt(30), places=12)

    def test_norm_of_negative_components(self):
        self.assertAlmostEqual(Quaternion(-1, -2, -3, -4).norm, math.sqrt(30), places=12)

    def test_basis_elements_have_unit_norm(self):
        for e in ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)):
            with self.subTest(e=e):
                self.assertEqual(Quaternion(*e).norm, 1.0)

    def test_zero_quaternion_has_zero_norm(self):
        self.assertEqual(Quaternion().norm, 0.0)

    def test_conjugate_has_the_same_norm(self):
        for s in SAMPLES:
            with self.subTest(q=s):
                q = Quaternion(*s)
                self.assertAlmostEqual(q.conjugate().norm, q.norm, places=12)


class TestInverse(QuaternionTestCase):

    def test_known_inverse(self):
        self.assertQuatAlmostEqual(Quaternion(1, 2, 3, 4).inverse(),
                                   (1 / 30, -2 / 30, -3 / 30, -4 / 30))

    def test_product_with_inverse_is_identity(self):
        for s in SAMPLES:
            with self.subTest(q=s):
                q = Quaternion(*s)
                self.assertQuatAlmostEqual(q * q.inverse(), (1, 0, 0, 0))
                self.assertQuatAlmostEqual(q.inverse() * q, (1, 0, 0, 0))

    def test_inverse_of_unit_quaternion_is_its_conjugate(self):
        q = Quaternion(0.5, -0.5, 0.5, -0.5)
        self.assertQuatAlmostEqual(q.inverse(), q.conjugate())

    def test_inverse_of_product_reverses_the_order(self):
        # (q ⊗ p)⁻¹ = p⁻¹ ⊗ q⁻¹
        q, p = Quaternion(*SAMPLES[0]), Quaternion(*SAMPLES[3])
        self.assertQuatAlmostEqual((q * p).inverse(), p.inverse() * q.inverse())

    def test_zero_quaternion_has_no_inverse(self):
        with self.assertRaises(ZeroDivisionError):
            Quaternion().inverse()


class TestRotation(QuaternionTestCase):
    """Rotating vectors with q ⊗ (0, v) ⊗ q*, using only the class's operations."""

    @staticmethod
    def about_axis(axis, angle_deg):
        """Unit quaternion for a rotation of angle_deg around a unit axis."""
        half = math.radians(angle_deg) / 2
        return Quaternion(math.cos(half), *(math.sin(half) * a for a in axis))

    def test_quarter_turn_about_z_maps_x_to_y(self):
        q = self.about_axis((0, 0, 1), 90)
        self.assertVecAlmostEqual(rotate(q, (1, 0, 0)), (0, 1, 0))

    def test_quarter_turn_about_x_maps_y_to_z(self):
        q = self.about_axis((1, 0, 0), 90)
        self.assertVecAlmostEqual(rotate(q, (0, 1, 0)), (0, 0, 1))

    def test_vector_on_the_axis_is_unchanged(self):
        q = self.about_axis((0, 0, 1), 37)
        self.assertVecAlmostEqual(rotate(q, (0, 0, 2.5)), (0, 0, 2.5))

    def test_rotation_preserves_length(self):
        q = self.about_axis((1 / math.sqrt(3),) * 3, 123)
        v = (0.3, -1.2, 2.0)
        self.assertAlmostEqual(math.dist(rotate(q, v), (0, 0, 0)),
                               math.dist(v, (0, 0, 0)), places=12)

    def test_composing_two_quarter_turns_gives_a_half_turn(self):
        q = self.about_axis((0, 0, 1), 90)
        self.assertVecAlmostEqual(rotate(q * q, (1, 0, 0)), (-1, 0, 0))

    def test_q_and_minus_q_give_the_same_rotation(self):
        q = self.about_axis((0, 1, 0), 60)
        v = (1, 2, 3)
        self.assertVecAlmostEqual(rotate(q, v), rotate(-1 * q, v))

    def test_inverse_undoes_the_rotation(self):
        q = self.about_axis((0, 1 / math.sqrt(2), 1 / math.sqrt(2)), 75)
        v = (0.5, -1.0, 2.0)
        self.assertVecAlmostEqual(rotate(q.inverse(), rotate(q, v)), v)


# The tests below cover operations the filter will need but that the class
# doesn't have yet. Each one is skipped until the method exists, then runs
# automatically. They assume the same style as the rest of the class: the
# operation returns a new Quaternion and leaves its operands unchanged.

@unittest.skipUnless(hasattr(Quaternion, "__sub__"), "subtraction not implemented yet")
class TestSubtraction(QuaternionTestCase):

    def test_subtracts_component_by_component(self):
        p = Quaternion(5, 6, 7, 8) - Quaternion(1, 2, 3, 4)
        self.assertQuatAlmostEqual(p, (4, 4, 4, 4))

    def test_quaternion_minus_itself_is_zero(self):
        q = Quaternion(1, 2, 3, 4)
        self.assertQuatAlmostEqual(q - q, (0, 0, 0, 0))

    def test_operands_are_not_modified(self):
        q, p = Quaternion(5, 6, 7, 8), Quaternion(1, 2, 3, 4)
        q - p
        self.assertEqual(components(q), (5, 6, 7, 8))
        self.assertEqual(components(p), (1, 2, 3, 4))


@unittest.skipUnless(hasattr(Quaternion, "__neg__"), "negation not implemented yet")
class TestNegation(QuaternionTestCase):

    def test_negates_every_component(self):
        self.assertQuatAlmostEqual(-Quaternion(1, -2, 3, -4), (-1, 2, -3, 4))


@unittest.skipUnless(hasattr(Quaternion, "normalize"), "normalize() not implemented yet")
class TestNormalize(QuaternionTestCase):

    def test_result_has_unit_norm(self):
        for s in SAMPLES:
            with self.subTest(q=s):
                self.assertAlmostEqual(Quaternion(*s).normalize().norm, 1.0, places=12)

    def test_known_result(self):
        self.assertQuatAlmostEqual(Quaternion(2, 0, 0, 0).normalize(), (1, 0, 0, 0))
        self.assertQuatAlmostEqual(Quaternion(1, 1, 1, 1).normalize(), (0.5, 0.5, 0.5, 0.5))

    def test_unit_quaternion_is_unchanged(self):
        self.assertQuatAlmostEqual(Quaternion(0.5, -0.5, 0.5, -0.5).normalize(),
                                   (0.5, -0.5, 0.5, -0.5))

    def test_does_not_modify_the_quaternion(self):
        q = Quaternion(1, 2, 3, 4)
        q.normalize()
        self.assertEqual(components(q), (1, 2, 3, 4))

    def test_zero_quaternion_raises(self):
        with self.assertRaises(ZeroDivisionError):
            Quaternion().normalize()


if __name__ == "__main__":
    unittest.main()
