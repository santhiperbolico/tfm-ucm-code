import numpy as np
import pytest

from hyper_velocity_stars_detection.sources.metrics import (
    convert_mas_yr_in_km_s,
    get_l_b_velocities,
)


def test_convert_mas_yr_in_km_s():
    parallax = np.array([0.254, 0.10, 0.5])
    pm = np.array([5.661751789124557, 2.10, 3.10])
    expected = np.array([105.65, 99.54, 29.38])
    result = convert_mas_yr_in_km_s(parallax, pm)
    assert result == pytest.approx(expected, abs=1e-1)


def test_get_l_b_velocities():
    ra = np.array([10.6846, 83.633, 161.265, 201.365, 266.417, 290.917, 344.413, 192.86])
    dec = np.array([41.2692, -5.416, 59.6844, -43.0192, -29.0078, 14.5167, -29.6222, 27.1283])
    pm_ra = np.array(
        [
            -12.54598812,
            45.07143064,
            23.19939418,
            9.86584842,
            -34.39813596,
            -34.40054797,
            -44.19163878,
            36.61761458,
        ]
    )
    pm_dec = np.array(
        [
            10.11150117,
            20.80725778,
            -47.94155057,
            46.99098522,
            33.24426408,
            -28.76608893,
            -31.81750328,
            -31.65954901,
        ]
    )

    pm_l = np.array(
        [
            -12.17047278,
            1.83379983,
            19.84693374,
            16.32790115,
            10.4575766,
            -41.61059705,
            -32.8096937,
            -35.22514405,
        ]
    )
    pm_b = np.array(
        [
            10.56048608,
            49.60859819,
            49.42371275,
            45.1540397,
            46.68031645,
            16.71657221,
            43.46007884,
            -33.20189705,
        ]
    )

    l_result, b_result = get_l_b_velocities(ra, dec, pm_ra, pm_dec)
    assert b_result == pytest.approx(pm_b, abs=1e-3)
    assert l_result == pytest.approx(pm_l, abs=1e-3)
