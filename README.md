# Madgwick Filter

A from-scratch Python implementation of the **Madgwick orientation filter**: it estimates the 3D orientation of a sensor, as a quaternion, from inertial measurements:

- **IMU mode (6-axis):** gyroscope + accelerometer
- **MARG mode (9-axis):** gyroscope + accelerometer + magnetometer

The goal is to understand the algorithm by building every piece by hand (quaternion algebra included), then to check the estimates against reference orientations recorded with real sensor data.

## How the filter works

At each sample the filter does three things:

1. **Predict with the gyroscope.** It integrates the angular rate through the quaternion derivative `q̇ = ½ · q ⊗ (0, ωx, ωy, ωz)`. This is accurate over short periods but drifts over time.
2. **Correct with gravity (and the magnetic field).** It runs one gradient-descent step that rotates the estimate so the measured acceleration lines up with gravity. In MARG mode, the measured magnetic field must also line up with the Earth's field. This removes drift in roll and pitch (and in yaw, for MARG).
3. **Fuse.** It blends the two using a single gain, `β`, then integrates over `dt` and normalizes the quaternion.

The appeal of Madgwick's method is that it is cheap: there is no covariance matrix as in a Kalman filter, and there is one tunable parameter.

## Repository layout

| File | Purpose | Status |
|---|---|---|
| `madgwick.py` | `Quaternion` class and `Madgwick` filter class | In progress |
| `filter.py` | `Filter` base class (common interface for orientation filters) | Empty stub, not committed yet |
| `quaternion.py` | Intended home of the `Quaternion` class | Empty |
| `load_data.py` | CSV loading and preprocessing of the datasets | Empty |
| `run.py` | Entry point: run the filter on a dataset and compare it with the reference | Empty |
| `tests.py` | Unit tests | Empty |
| `data/` | IMU recordings (see below) | Done |

## Dataset

The data comes from the Kaggle dataset [IMU Data for AI](https://www.kaggle.com/datasets/banaankiamanesh/imu-for-ai). It was chosen because each sample contains the raw 9-axis readings *and* reference orientations (Euler angles and quaternion), which gives a ground truth to compare against.

| File | Columns | Samples |
|---|---|---|
| `IMU_Data_1.csv` … `IMU_Data_5.csv` | Gyro, Acc, Mag, Euler, Quat (16 columns) | 30,000 each |
| `IMU_Data_1(No Mag).csv` … `IMU_Data_3(No Mag).csv` | Gyro, Acc, Euler, Quat (13 columns) | 30,000 each |

The "No Mag" files are separate recordings, not copies of the full files with the magnetometer columns removed. They are useful for testing IMU (6-axis) mode on its own.

Column meanings and units, **inferred from the values** (to be confirmed):

| Columns | Quantity | Likely unit |
|---|---|---|
| `Gyro_x/y/z` | Angular rate | °/s (values reach ~1500, so they must be converted to rad/s) |
| `Acc_x/y/z` | Acceleration | m/s² (≈ 9.6 at rest) |
| `Mag_x/y/z` | Magnetic field | µT |
| `Euler_x/y/z` | Reference orientation | degrees (`Euler_z` spans 0–360) |
| `Quat_0..3` | Reference orientation | unitless; `Quat_0` is the scalar part `w` |

Points to handle when loading the data:

- **Header formatting:** the header of `IMU_Data_1.csv` is padded with spaces (`Gyro_x  , Gyro_y  , …`), so column names need stripping.
- **Sampling rate:** the files don't state it. It is needed for `dt` and must be confirmed from the dataset's documentation.
- **Reference quaternion norms:** the reference quaternions are only approximately unit-norm (≈ 0.96–1.04). `IMU_Data_5.csv` contains at least one near-zero quaternion. Normalize the references and drop invalid samples before comparing.
- **Sign ambiguity:** `q` and `−q` describe the same rotation (e.g. `IMU_Data_1.csv` starts near `w = −1`), so comparisons must not depend on the sign.

## Current state

The project is at an early stage: the structure, dataset and quaternion groundwork are in place, and the filter itself isn't implemented yet.

**Done**
- Project skeleton and dataset
- First draft of the `Quaternion` class: product, sum, conjugate, norm, inverse
- `Madgwick` class skeleton: holds the orientation quaternion. `approximate(G, A, M=None)` has the gyroscope prediction step started; the correction and fusion steps are placeholders.

**Known issues in the current `Quaternion` draft** (to fix before building the filter on it)
- `__mul__` and `__add__` modify `self` in place and return `None`, so expressions like `q * p` or `0.5 * (q * p)` don't work. They should return a new `Quaternion`. A `__rmul__` is needed for `scalar * q`.
- The Hamilton product has sign errors in the `y` and `z` components. They should be:
  - `y = w1·y2 − x1·z2 + y1·w2 + z1·x2`
  - `z = w1·z2 + x1·y2 − y1·x2 + z1·w2`
- `norm` passes four arguments to `math.sqrt`. It should be `sqrt(w² + x² + y² + z²)`.
- `inverse` divides by a scalar, but there is no `__truediv__`.
- There is no `normalize()` method yet.

## Development roadmap

### 1. Quaternion foundations
- [ ] Move `Quaternion` to `quaternion.py` and fix the issues above
- [ ] Make the operations return new objects; add `__rmul__`, `__sub__`, `__truediv__`, `__neg__`
- [ ] Add `normalize()`, rotating a vector (`q ⊗ v ⊗ q*`) and conversions to and from Euler angles
- [ ] Unit tests in `tests.py`: identities, `q ⊗ q⁻¹ = 1`, known rotations, Euler round-trips

### 2. Data loading (`load_data.py`)
- [ ] Parse the CSVs into arrays, with stripped headers and optional magnetometer columns
- [ ] Convert the gyroscope from °/s to rad/s
- [ ] Determine the sampling period `dt`
- [ ] Clean the reference quaternions (normalize them and drop invalid rows)

### 3. Madgwick: IMU mode (6-axis)
- [ ] Gyroscope integration (prediction)
- [ ] Gradient-descent correction using the accelerometer (objective function and Jacobian for gravity)
- [ ] Fusion with gain `β`, integration over `dt`, normalization
- [ ] Validate on the `(No Mag)` recordings

### 4. Madgwick: MARG mode (9-axis)
- [ ] Add the magnetometer term to the objective function
- [ ] Compensate for magnetic distortion (estimate the Earth-field reference direction `b` online)
- [ ] Optional: gyroscope bias drift compensation (`ζ` gain)

### 5. Evaluation (`run.py`)
- [ ] Run the filter over every dataset
- [ ] Align the initial orientation and frame convention with the reference
- [ ] Metrics: angular error `2·acos(|⟨q_est, q_ref⟩|)` and per-axis Euler RMSE
- [ ] Plots of estimated vs. reference orientation over time
- [ ] Tune `β` and study its effect on convergence speed vs. noise

### 6. Possible extensions
- [ ] Use `filter.py` as a common interface to compare other filters, such as a complementary filter, the Mahony filter or an EKF
- [ ] Vectorize with NumPy and benchmark performance
- [ ] Add a CLI for `run.py` (choice of dataset, mode and `β`)

## Requirements

- Python 3. The current code uses only the standard library (`math`).
- NumPy and Matplotlib will probably be added for data handling and plotting.

The filter can't be run yet. The planned entry point is:

```bash
python run.py
```

## References

- S. O. H. Madgwick, *An efficient orientation filter for inertial and inertial/magnetic sensor arrays*, technical report, University of Bristol, 2010.
- S. O. H. Madgwick, A. J. L. Harrison, R. Vaidyanathan, *Estimation of IMU and MARG orientation using a gradient descent algorithm*, IEEE International Conference on Rehabilitation Robotics (ICORR), 2011.
- Dataset: [IMU Data for AI, Kaggle](https://www.kaggle.com/datasets/banaankiamanesh/imu-for-ai)
