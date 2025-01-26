import numpy as np


ARCS_RAD = np.pi / (180 * 3600)
PC_KM = 3.0857*1e13
YR_S = 60*60*24*365

def convert_mas_yr_in_km_s(
        parallax_data: np.ndarray,
        pm_data: np.ndarray
) -> np.ndarray:
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
