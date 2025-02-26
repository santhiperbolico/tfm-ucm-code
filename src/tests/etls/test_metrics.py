import numpy as np
import pytest

from hyper_velocity_stars_detection.etls.metrics import convert_mas_yr_in_km_s, get_l_b_velocities


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
            -7.47850727,
            49.62968943,
            2.17742277,
            27.77132817,
            -18.30470321,
            -43.01332523,
            -53.20948091,
            20.97178387,
        ]
    )
    pm_b = np.array(
        [
            14.27291851,
            1.12683782,
            -53.21525149,
            39.16938841,
            44.19672717,
            -12.67830533,
            -11.57607853,
            -43.62752597,
        ]
    )

    l_result, b_result = get_l_b_velocities(ra, dec, pm_ra, pm_dec)
    assert b_result == pytest.approx(pm_b, abs=1e-3)
    assert l_result == pytest.approx(pm_l, abs=1e-3)
