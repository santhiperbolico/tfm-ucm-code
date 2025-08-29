import numpy as np
from astropy import units as u
from astropy.coordinates import Galactic, SkyCoord

ARCS_RAD = np.pi / (180 * 3600)
PC_KM = 3.0857 * 1e13
YR_S = 60 * 60 * 24 * 365


def convert_mas_yr_in_km_s(parallax_data: np.ndarray, pm_data: np.ndarray) -> np.ndarray:
    """
    Función que calcula la velocidad tangencial en km/s usando el movimiento
    propio en mas/yr y el paraláje en mas.

    Parameters
    ----------
    parallax_data: np.ndarray
        Datos de paralaje en miliarcos de segun por año.
    pm_data: np.ndarray
        Datos del movimiento propio en miliarcos de segun por año.

    Returns
    -------
    vt_data: np.ndarray
        Velocidad tangencial en km/s
    """
    distance_pc = 1 / (parallax_data / 1000)
    factor = ARCS_RAD * PC_KM / YR_S
    vt_data = factor * (pm_data / 1000) * distance_pc
    return vt_data


def get_l_b_velocities(
    ra: np.ndarray,
    dec: np.ndarray,
    pm_ra_cosdec: np.ndarray,
    pm_dec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Función que calcula las componentes l y b de la velocidad en mas/yr

    Parameters
    ----------
    ra: np.ndarray
        Array con la ascensión recta en grados.
    dec: np.ndarray
        Array con la declinación en grados.
    pm_ra_cosdec: np.ndarray
        Array con el movimiento propio en ascensión recta por el coseno de la declinación.
        Las unidades son mas/yr
    pm_dec: np.ndarray
        Array con el movimiento propio en la declinación. Las unidades son mas/yr.

    Returns
    -------
    pm_l_cosb: np.ndarray
        Array con el movimiento propio en longitud por el coseno de la latitud.
        Las unidades son mas/yr
    pm_dec: np.ndarray
        Array con el movimiento propio en la latitud. Las unidades son mas/yr.

    """

    velocity_coord = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        pm_ra_cosdec=pm_ra_cosdec * u.mas / u.yr,
        pm_dec=pm_dec * u.mas / u.yr,
        frame="icrs",
    )

    velocity_coord_gal = velocity_coord.transform_to(Galactic())

    return velocity_coord_gal.pm_l_cosb.value, velocity_coord_gal.pm_b.value
