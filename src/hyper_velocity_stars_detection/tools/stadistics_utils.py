import numpy as np
import pandas as pd
from attr import attrs
from pingouin import multivariate_normality
from tqdm import tqdm

MAX_SAMPLE = 5000


@attrs(auto_attribs=True)
class ResultMVN:
    """
    Resultados del test estadístico que comprueba la normalidad.

    Attributes
    ----------
    is_multivariate_normal: bool
        Indica si los datos siguen la normal multivariante.
    statistic_name: str
        Nombre del test estadístico
    statistic: float
        Valor del estadístico
    pval: float
        P-valor obtenido con el test.
    """

    is_multivariate_normal: bool
    statistic_name: str
    statistic: float
    pval: float

    @property
    def values(self) -> np.ndarray:
        return np.array([self.statistic, self.pval])

    def is_normal(self, alpha: float | None = None) -> bool:
        if alpha:
            return self.pval > alpha
        return self.is_multivariate_normal


def is_multivariate_normality_hz(
    df_data: pd.DataFrame | np.ndarray, alpha: float = 0.05
) -> ResultMVN:
    """
    Función que indica si los datos siguen una distribución normal multivariante, aplicando
    el test Henze-Zirkler.

    Parameters
    ----------
    df_data: pd.DataFrame | np.ndarray
        Datos a comprobar su distribción.
    alpha: float, default 0.05
        Nivel de significancia.

    Returns
    -------
    results: ResultMVN
        Reusltados del test estadístico
    """
    hz_results = multivariate_normality(df_data, alpha=alpha)
    results = dict(zip(["statistic", "pval", "is_multivariate_normal"], list(hz_results)))
    results.update({"statistic_name": "Henze-Zirkler"})
    return ResultMVN(**results)


def is_multivariate_normality(
    df_data: pd.DataFrame | np.ndarray, alpha: float = 0.05, max_sample: int = MAX_SAMPLE
) -> ResultMVN:
    """
    Función que indica si los datos siguen una distribución normal multivariante.

    Parameters
    ----------
    df_data: pd.DataFrame | np.ndarray
        Datos a comprobar su distribción.
    alpha: float, default 0.05
        Nivel de significancia.
    max_sample: int, default MAX_SAMPLE
        Muestra máxima que puede analizar d euna vez. Cuando supera este umbral aplica
        el método de monte carlo.

    Returns
    -------
    results: ResultMVN
        Reusltados del test estadístico
    """
    if df_data.shape[0] <= max_sample:
        return is_multivariate_normality_hz(df_data, alpha)

    n_simuls = max(10 * (df_data.shape[0] // max_sample + 1), 100)
    pval_array = np.zeros(n_simuls)
    statistics_array = np.zeros(n_simuls)
    for simul in tqdm(
        range(n_simuls), total=n_simuls, desc="Analizando la muestra", unit="simulación"
    ):
        hz_result = is_multivariate_normality_hz(df_data.sample(max_sample, replace=True))
        pval_array[simul] = hz_result.pval
        statistics_array[simul] = hz_result.statistic
    pval = np.quantile(pval_array, 1 - alpha)
    results = {
        "is_multivariate_normal": pval > alpha,
        "pval": pval,
        "statistic": statistics_array.mean(),
        "statistic_name": "MC - Henze-Zirkler",
    }
    return ResultMVN(**results)
