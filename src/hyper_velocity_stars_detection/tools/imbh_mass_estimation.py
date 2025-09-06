from typing import Optional

import numpy as np


def v_ejection(a: float, m_bin: float, m_imbh: float) -> float:
    """
    Función que calcula la velocidad de ejección considerando el escenario de eyección
    tipo Hills [Hills, 1988], en el que un sistema binario de masa total m_bin = m1 + m2 y
    semieje mayor a interactúa con un IMBH de masa m_imbh.

    Parameters
    ----------
    a: float
        Semije mayor del sistema binario en UA.
    m_bin: float
        Masa total de la binaria en Msun.
    m_imbh: float
        Masa total del IMBH en Msun.

    Returns
    -------
    v_ej: float
        Media de la velocidad de ejección esperada.
    """
    return (
        460.0
        * (a / 0.1) ** (-0.5)
        * (m_bin / 2.0) ** (1.0 / 3.0)
        * (m_imbh / 1.0e3) ** (1.0 / 6.0)
    )


def get_v_ejection(
    a: float,
    m_bin: float,
    m_imbh: float,
    sigma_f: float,
    size: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Función que extrae un vector aleatorio con las velocidades de ejección esperadas
    teniendo en cuenta un mecanismo de Hill (ver v_ejection) y una distribución
    de N(v_ej, sigma_f * v_ej).

    Parameters
    ----------
    a: float
        Semije mayor del sistema binario en UA.
    m_bin: float
        Masa total de la binaria en Msun.
    m_imbh: float
        Masa total del IMBH en Msun.
    sigma_f: float,
        Proporción de la desviación según el trabajo de Fragione 2018.
    size: int = 1
        Tamaño de la muestra
    seed: Optional[int] = None
        Semilla utilizada.
    Returns
    -------
    v_ej_array: np.ndarray
        Array con las velocidades de ejección
    """
    rng = np.random.default_rng()
    if seed:
        rng = np.random.default_rng(seed)
    v_ej = v_ejection(a, m_bin, m_imbh)
    v_ej_array = rng.normal(loc=v_ej, scale=sigma_f * v_ej, size=size)
    return v_ej_array


def estimate_log10_mbh_from_sigma(sigma_kms: float) -> tuple[float, float]:
    """
    Método que estima la masa del IMBH usando la dispersión de velocidades del cúmulo
    usando el método de Lützgendorf, N., Kissler-Patig, M., Neumayer, N., Baumgardt, H., Noyola,
    E., de Zeeuw, P. T., ... & Feldmeier, A. (2013). M•− σrelation for intermediate-mass black
    holes in globular clusters. Astronomy & Astrophysics, 555, A26.

    Parameters
    ----------
    sigma_kms: float
        Dispersión de velocidades.

    Returns
    -------
    log10_mbh: float
        Logarítmo base 10 de la masa del BH.
    scatter_dex: float
        Dispersión esperada.
    """
    if sigma_kms <= 0:
        raise ValueError("No se admiten valores negativos para sigma.")
    scatter_dex = 0.36
    x = np.log10(sigma_kms / 200.0)
    log10_mbh = 7.41 + 3.28 * x
    return log10_mbh, scatter_dex


def sample_mbh(
    log10_mbh, scatter_dex, size: int = 20000, seed: Optional[int] = None
) -> np.ndarray:
    """
    FUnción que extrae una muestra de la masa esperada para un IMBH de masa log10_mbh y
    dispersión scatter_dex.

    Parameters
    ----------
    log10_mbh: float
        Logarítmo base 10 de la masa del BH.
    scatter_dex: float
        Dispersión esperada.
    size: int = 20000
        Tamaño de la muestra.
    seed: Optional[int] = None
        Semilla utilizada.


    Returns
    -------
    sample: np.ndarray
        Array con lasmasas del BH.

    """
    rng = np.random.default_rng()
    if seed:
        rng = np.random.default_rng(seed)
    mu_ln = log10_mbh * np.log(10.0)
    sigma_ln = scatter_dex * np.log(10.0)
    ln_samples = rng.normal(loc=mu_ln, scale=sigma_ln, size=size)
    return np.exp(ln_samples)


def v_ejections_sample(
    a_space: tuple[float, float],
    m_bin_space: tuple[float, float],
    sigma_kms: float,
    sigma_f: float,
    size: int = 20000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Función que extrae una muestra de
    Parameters
    ----------
    a_space: tuple[float, float],
        Intervalo de búsqueda en el eje semimayo del sistema binario.
    m_bin_space: tuple[float, float],
        Intervalo de búsqueda en la masa total del sistema binario.
    sigma_kms: float,
        Dispersión de velocidades del cúmulo
    sigma_f: float,
        Proporción de la desviación según el trabajo de Fragione 2018.
    size: int = 20000
        Tamaño de la muestra
    Semilla para generar la muestra.

    Returns
    -------
    v_ejection_sample: np.ndarray
        Array con la muestra de velocidades de eyección esperadas.
    """
    rng = np.random.default_rng(seed)
    log10_mbh, scatter_dex = estimate_log10_mbh_from_sigma(sigma_kms)
    imbh_sample = sample_mbh(log10_mbh=log10_mbh, scatter_dex=scatter_dex, size=size, seed=seed)
    a_sample = rng.uniform(low=a_space[0], high=a_space[1], size=size)
    m_bin_sample = rng.uniform(low=m_bin_space[0], high=m_bin_space[1], size=size)
    v_ejection_sample = np.zeros(size)
    for i_sample in range(size):
        v_ejection_sample[i_sample] = get_v_ejection(
            a=a_sample[i_sample],
            m_bin=m_bin_sample[i_sample],
            m_imbh=imbh_sample[i_sample],
            sigma_f=sigma_f,
            size=1,
        )
    return v_ejection_sample
